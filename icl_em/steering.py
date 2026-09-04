"""Residual-stream steering for M1, with the instrumentation that makes it checkable.

STUDY-01 section 4.  Add `coeff * u_L` to the post-MLP residual stream at a set of
layers and read the Gate A log-probability instruments.  The direction `u_L` comes
from `results/m1/direction.npz` and is frozen before any steered forward pass.

Written in house style rather than imported.  The two vendored references were read:

  * `assistant-axis-master/assistant_axis/steering.py` is the cleaner one and handles
    the tuple-vs-bare-tensor return correctly, but it steers in the model dtype.
  * `persona_vectors-main/activation_steer.py` has a `positions="response"` branch
    that only touches `[:, -1, :]`.  That is wrong for a teacher-forced multi-token
    span, which is exactly what every M1 readout scores.  Not copied.

Three things here are load-bearing and are the reason this is not a five-line hook.

1.  **bf16 annihilation.**  The residual stream is bf16: 8 mantissa bits, so a value
    of magnitude ~48 has a ULP of ~0.25.  M1's smallest registered interventions are
    ~0.6 in raw units spread over 3584 coordinates, i.e. per-coordinate additions far
    below the local ULP.  Adding in bf16 would silently round most of them away and
    the run would report "no effect" for what was really "no intervention".  So the
    addition happens in fp32 and, more importantly, the REALIZED projection change is
    measured after the cast back and compared against the intended one.  That
    measurement is Gate 0.

2.  **The bare-tensor return.**  Under transformers 4.57.6 `Qwen2DecoderLayer.forward`
    returns a bare tensor, not the `(hidden_states, ...)` tuple older versions
    returned.  Indexing `[0]` on it selects batch element 0 -- at batch 1 that is
    accidentally shape-correct and would never fail loudly.  Handled explicitly, the
    same way `activations.py` handles it on the read side.

3.  **Decode steps.**  Under `model.generate` the hook fires once at prefill with
    seq=n_prompt and then once per decode step with seq=1.  A hook that asserts a
    fixed sequence length steers only the prefill and raises on the first new token;
    a hook that asserts nothing at all can steer only the prefill and never say so.
    So the sequence length is not asserted, and the fire count IS -- see `stats()`.
"""
from __future__ import annotations

import numpy as np
import torch

from . import config as cfg

POSITION_MODES = ("all", "response")


class Steerer:
    """Context manager adding `coeff[L] * unit[L]` to layer L's output.

    `directions` maps layer index -> (unit_vector, coeff_raw).  The unit vector is
    L2-normalised in fp64 by the caller; `coeff_raw` is the displacement in RAW
    activation units, i.e. `alpha * ||w_L||` for an alpha expressed in W-equivalents.
    Keeping the two apart is what lets `stats()` report intended-vs-realized in the
    same units the direction file records.

    Usage:

        with Steerer(runner.model, directions, positions="all") as st:
            ...forward passes...
        st.assert_effective(cfg.M1_GATE0_MIN_REALIZED_FRAC)
    """

    def __init__(
        self,
        model,
        directions: dict[int, tuple[np.ndarray, float]],
        *,
        positions: str = "all",
        hidden_size: int = cfg.HIDDEN_SIZE,
    ):
        if positions not in POSITION_MODES:
            raise KeyError(f"Unknown positions mode {positions!r}. Known: {POSITION_MODES}")
        if not directions:
            raise ValueError(
                "Steerer was given no layers. An unsteered pass is `alpha=0`, which "
                "still registers hooks so the no-op is measured; an empty dict is a bug."
            )

        layers = model.model.layers
        n_layers = model.config.num_hidden_layers
        if n_layers != cfg.N_LAYERS:
            raise AssertionError(f"expected {cfg.N_LAYERS} layers, model reports {n_layers}")
        if model.config.hidden_size != hidden_size:
            raise AssertionError(
                f"expected hidden {hidden_size}, model reports {model.config.hidden_size}"
            )
        for layer_idx in directions:
            if not 0 <= layer_idx < n_layers:
                raise IndexError(f"layer {layer_idx} out of range for {n_layers} layers")

        self.model = model
        self.layers = layers
        self.positions = positions
        self.hidden_size = hidden_size
        self._handles: list = []
        self._span_start: int | None = None

        # fp32 on the model's device.  Cast per-call to the activation dtype instead of
        # storing a bf16 copy: a bf16 unit vector is itself already rounded, and the
        # rounding would then be baked into every call rather than measured once.
        p = next(model.parameters())
        self.unit: dict[int, torch.Tensor] = {}
        self.coeff: dict[int, float] = {}
        for layer_idx, (vec, coeff) in directions.items():
            v = torch.as_tensor(np.asarray(vec, dtype=np.float64), dtype=torch.float32,
                                device=p.device)
            if v.ndim != 1 or v.numel() != hidden_size:
                raise ValueError(
                    f"direction for layer {layer_idx} has shape {tuple(v.shape)}, "
                    f"expected ({hidden_size},)"
                )
            norm = float(v.norm())
            if abs(norm - 1.0) > 1e-5:
                raise AssertionError(
                    f"direction for layer {layer_idx} has norm {norm:.6f}, expected a unit "
                    "vector -- coefficients are in raw activation units and would be wrong"
                )
            self.unit[layer_idx] = v
            self.coeff[layer_idx] = float(coeff)

        # Per-layer accumulators.  `n_positions` counts hooked positions, not calls, so a
        # prefill of 1170 and 200 decode steps are distinguishable from 1370 prefills.
        self._acc = {
            layer_idx: {
                "n_calls": 0, "n_positions": 0, "seq_lens": [],
                "pre_sum": 0.0, "post_sum": 0.0,
            }
            for layer_idx in self.unit
        }

    # ------------------------------------------------------------------ span control

    def set_span(self, start: int | None) -> None:
        """First position to steer, in the CURRENT forward's coordinates.

        Only meaningful for `positions="response"`.  `score_continuation` tokenises the
        suffix and the suffix+continuation jointly and asserts prefix stability, so the
        caller knows `n_suffix` exactly; passing it here steers the whole continuation
        span plus nothing else.  Passing `n_suffix` (not `n_suffix - 1`) is deliberate:
        the last suffix position is what PREDICTS the first continuation token, and
        steering it would move a logit that the response-mean direction never described.
        """
        if self.positions != "response" and start is not None:
            raise ValueError("set_span is only meaningful with positions='response'")
        self._span_start = start

    # ------------------------------------------------------------------------- hooks

    def _make_hook(self, layer_idx: int):
        unit = self.unit[layer_idx]
        coeff = self.coeff[layer_idx]
        acc = self._acc[layer_idx]

        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() != 3 or h.shape[0] != 1 or h.shape[2] != self.hidden_size:
                raise AssertionError(
                    f"layer {layer_idx} produced {tuple(h.shape)}, expected "
                    f"(1, seq, {self.hidden_size}) -- batch must be 1 here"
                )
            # NOT asserted: h.shape[1]. Decode steps legitimately arrive with seq == 1.
            seq = h.shape[1]

            if self.positions == "all":
                lo, hi = 0, seq
            else:
                if self._span_start is None:
                    raise AssertionError(
                        "positions='response' but no span is set. Call set_span() before "
                        "each forward pass, or the hook would steer nothing and say nothing."
                    )
                lo, hi = min(self._span_start, seq), seq
            n = hi - lo

            h32 = h.float()
            # Summed over positions, averaged in stats(): the per-position projections
            # are what a mean displacement has to be measured against.
            pre = float((h32[0, lo:hi] @ unit).sum()) if n else 0.0
            if n and coeff != 0.0:
                h32 = h32.clone()
                h32[0, lo:hi] += coeff * unit
            out = h32.to(h.dtype)
            # Measured AFTER the cast back, which is the whole point: this is what the
            # next layer actually receives, bf16 rounding included.
            post = float((out[0, lo:hi].float() @ unit).sum()) if n else 0.0

            acc["n_calls"] += 1
            acc["n_positions"] += n
            acc["seq_lens"].append(seq)
            acc["pre_sum"] += pre
            acc["post_sum"] += post

            return (out, *output[1:]) if isinstance(output, tuple) else out

        return hook

    def __enter__(self):
        if self._handles:
            raise RuntimeError("Steerer is already active; it is not re-entrant")
        for layer_idx in sorted(self.unit):
            self._handles.append(
                self.layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            )
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # -------------------------------------------------------------- instrumentation

    def stats(self) -> dict:
        """Per-layer intended vs realized displacement along the unit direction.

        `realized_frac` is Gate 0.  Below `M1_GATE0_MIN_REALIZED_FRAC` the intervention
        was rounded away by bf16 and the operating point must be reported as "not
        applied", not as "no effect" -- the distinction PREDICTIONS-m1 section 4 turns
        on.  `n_calls` and `seq_len_hist` are the other half: they prove the hook fired
        at every decode step and not only at prefill.
        """
        out = {}
        for layer_idx, acc in sorted(self._acc.items()):
            n = acc["n_positions"]
            intended = self.coeff[layer_idx]
            realized = (acc["post_sum"] - acc["pre_sum"]) / n if n else float("nan")
            lens = acc["seq_lens"]
            out[layer_idx] = {
                "coeff_raw": intended,
                "n_calls": acc["n_calls"],
                "n_positions": n,
                "seq_len_min": min(lens) if lens else 0,
                "seq_len_max": max(lens) if lens else 0,
                "mean_proj_pre": acc["pre_sum"] / n if n else float("nan"),
                "mean_proj_post": acc["post_sum"] / n if n else float("nan"),
                "realized_displacement": realized,
                "realized_frac": (realized / intended) if intended else float("nan"),
            }
        return out

    def summary(self) -> dict:
        """Band-aggregate intended vs realized displacement.  This is Gate 0.

        The aggregate, not the per-layer minimum.  A named cancellation point subtracts
        each layer's own measured displacement, and at layers where the target arm barely
        moves that coefficient is near zero BY DESIGN -- `cancel_P` intends +0.025 at
        layer 12.  Rounding away a deliberate near-zero is not a failed intervention, but
        a per-layer minimum scores it as one (measured: 0.170 for cancel_P, whose two
        smallest layers carry 2% of the total displacement).  The question Gate 0 actually
        asks is whether the intervention reached the residual stream, so the statistic is
        sum|realized| / sum|intended| over the hooked layers.  The per-layer table is
        still recorded in `stats()` and reported.
        """
        st = self.stats()
        intended = sum(abs(s["coeff_raw"]) for s in st.values())
        realized = sum(
            abs(s["realized_displacement"]) for s in st.values()
            if s["realized_displacement"] == s["realized_displacement"]
        )
        nz = [s["realized_frac"] for s in st.values() if s["coeff_raw"] != 0.0]
        return {
            "n_layers": len(st),
            "any_fired": any(s["n_calls"] for s in st.values()),
            "intended_total": intended,
            "realized_total": realized,
            "realized_frac": (realized / intended) if intended else float("nan"),
            "min_layer_realized_frac": min(nz) if nz else float("nan"),
            "is_noop": intended == 0.0,
        }

    def assert_effective(self, min_frac: float = cfg.M1_GATE0_MIN_REALIZED_FRAC) -> dict:
        """Raise unless the intended displacement actually reached the residual stream.

        Called after each operating point.  A silent failure here looks exactly like a
        genuine null result, which is the single most dangerous confusion in this study.
        """
        st = self.stats()
        agg = self.summary()
        if not agg["any_fired"]:
            raise AssertionError(
                "no steering hook fired. The forward pass did not go through the hooked "
                "layers, so nothing was steered and any readout is the unsteered model."
            )
        if agg["is_noop"] or agg["realized_frac"] >= min_frac:
            return st
        lines = "\n".join(
            f"    layer {i}: intended {s['coeff_raw']:+.5f}, realized "
            f"{s['realized_displacement']:+.5f} ({s['realized_frac']:.3f} of intended)"
            for i, s in sorted(st.items()) if s["coeff_raw"] != 0.0
        )
        raise AssertionError(
            "Gate 0 failed: the steering displacement did not survive the bf16 residual "
            f"stream. Band aggregate realized/intended is {agg['realized_frac']:.3f}, "
            f"need >= {min_frac}.\n{lines}\n"
            "  This operating point is NOT APPLIED. It must not be reported as a null."
        )


# ------------------------------------------------------------------ direction plumbing

def band_directions(
    unit: np.ndarray, norms: np.ndarray, layers, alpha: float
) -> dict[int, tuple[np.ndarray, float]]:
    """Scalar alpha in W-equivalents -> the per-layer (unit, raw coefficient) map.

    The displacement at layer L is `alpha * ||w_L|| * u_L`, so alpha is in units of "one
    W-worth of displacement AT THAT LAYER".  The per-layer norm matters: it runs from
    4.65 at layer 12 to 6.72 at layer 18, and reusing layer 14's 4.894 everywhere would
    under-steer the top of the band by 27%.
    """
    return {int(L): (unit[L], float(alpha) * float(norms[L])) for L in layers}


def cancel_directions(
    unit: np.ndarray, projections: np.ndarray, layers
) -> dict[int, tuple[np.ndarray, float]]:
    """The named per-layer cancellation points, which are vectors and not scalar alphas.

    `projections[L]` is the measured mean displacement of the target arm along `u_L`, so
    subtracting it lands that arm's mean projection exactly on its reference.  Registered
    as `cancel_DN` (target D-N) and `cancel_P` (target P-B) in PREDICTIONS-m1 section 2.
    """
    return {int(L): (unit[L], -float(projections[L])) for L in layers}
