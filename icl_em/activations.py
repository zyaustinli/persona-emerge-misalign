"""Residual-stream capture for the M2 probe-transfer measurement.

STUDY-01 section 3.  Every condition is measured on the SAME frozen reference text,
teacher-forced, so the only thing differing between conditions is the weights (W) or
the preceding context (D, N).  Measuring each condition on its own generations would
confound the state with the text that produced it.

Two poolings per item, both over the post-MLP residual stream, all 28 layers:

  response_avg -- mean over response tokens only.
  prompt_last  -- the single last prompt token, before any response token exists.
                  Nearly free, and it is what a deployment monitor could actually
                  read (PROPOSAL.md section 4.2).  Reported separately.

Why forward hooks rather than `output_hidden_states=True`, verified against the
installed transformers 4.57.6 rather than assumed:

  * `Qwen2Model.forward` applies `self.norm(hidden_states)` before returning, so the
    LAST entry of `output_hidden_states` is post-final-norm, not the raw layer-27
    residual.  Hooks avoid the ambiguity entirely.
  * `Qwen2DecoderLayer.forward` ends in `return hidden_states` -- a BARE TENSOR, not
    the `(hidden_states, ...)` tuple older versions returned.  The vendored reference
    implementation (model-organisms .../activation_collection.py) does `output[0]`,
    which under this version indexes the BATCH dimension.  At batch 1 that is
    accidentally value-correct and would never fail loudly.  Handled explicitly below.

Pooling happens inside the hook: the span is known before the forward pass, so each
layer is reduced to two (hidden,) vectors immediately instead of retaining the full
(1, seq, hidden) activation.  That drops the transient from ~300 MB to under a
megabyte, which matters with a 15.5 GB model in 24 GB of unified memory.
"""
from __future__ import annotations

import hashlib

import torch

from . import config as cfg


def response_span(runner, prefix_text: str, response: str) -> tuple[int, int]:
    """Token indices [n_prefix, n_total) covering `response`.

    Exactly the boundary logic `runner.score_continuation` uses: tokenise the prefix
    alone and the prefix+response jointly, then assert the joint encoding is
    prefix-stable, so a BPE merge across the boundary cannot silently shift which
    tokens get pooled.  Reusing it rather than re-deriving it is what keeps the
    pooling mask honest against the logprob readout.
    """
    prefix_ids = runner._encode(prefix_text)
    full_ids = runner._encode(prefix_text + response)
    n_prefix = prefix_ids.shape[1]
    if not torch.equal(full_ids[:, :n_prefix], prefix_ids):
        raise AssertionError(
            "tokenisation is not prefix-stable across the response boundary for "
            f"{response[:40]!r} -- the pooling mask would cover the wrong tokens"
        )
    n_total = full_ids.shape[1]
    if n_total - n_prefix < 1:
        raise ValueError(f"response {response[:40]!r} encodes to zero tokens")
    return n_prefix, n_total


@torch.no_grad()
def capture(runner, prefix_text: str, response: str) -> dict[str, torch.Tensor]:
    """Pooled post-MLP residual stream for one teacher-forced item.

    Returns {"response_avg": (n_layers, hidden), "prompt_last": (n_layers, hidden)}
    as float32 CPU tensors, plus the span in `.spans`-free form via `n_prompt_tokens`
    / `n_response_tokens` on the returned dict's metadata keys.
    """
    if runner._prefix_cache is not None:
        raise AssertionError(
            "the prefix KV cache is populated. It diverges up to ~0.3 nats on MPS bf16 "
            "at 1113 tokens, and it also shifts cache_position so the hooks would see "
            "only the new-token slice. Construct LocalRunner with use_prefix_cache=False."
        )
    n_prefix, n_total = response_span(runner, prefix_text, response)
    ids = runner._encode(prefix_text + response)
    # The single most valuable assertion here: this digest proves every condition pools
    # the SAME response tokens. A BPE merge at the ":"/first-word boundary that behaved
    # differently under one prefix would otherwise silently change what is averaged.
    response_sha = hashlib.sha256(
        ",".join(str(t) for t in ids[0, n_prefix:].tolist()).encode()
    ).hexdigest()[:16]

    model = runner.model
    layers = model.model.layers
    n_layers = model.config.num_hidden_layers
    if n_layers != cfg.N_LAYERS:
        raise AssertionError(f"expected {cfg.N_LAYERS} layers, model reports {n_layers}")
    if model.config.hidden_size != cfg.HIDDEN_SIZE:
        raise AssertionError(
            f"expected hidden {cfg.HIDDEN_SIZE}, model reports {model.config.hidden_size}"
        )

    resp_avg = torch.empty(n_layers, cfg.HIDDEN_SIZE, dtype=torch.float32)
    prompt_last = torch.empty(n_layers, cfg.HIDDEN_SIZE, dtype=torch.float32)
    seen: set[int] = set()

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            # transformers 4.57.6 Qwen2DecoderLayer returns a bare tensor; older
            # versions and other architectures return a tuple.  Never index [0]
            # blindly -- on a tensor that silently selects batch element 0.
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() != 3 or h.shape[0] != 1 or h.shape[2] != cfg.HIDDEN_SIZE:
                raise AssertionError(
                    f"layer {layer_idx} produced {tuple(h.shape)}, expected "
                    f"(1, seq, {cfg.HIDDEN_SIZE}) -- batch must be 1 here"
                )
            if h.shape[1] != n_total:
                raise AssertionError(
                    f"layer {layer_idx} saw seq={h.shape[1]}, expected {n_total}"
                )
            h = h[0].float()
            resp_avg[layer_idx] = h[n_prefix:n_total].mean(dim=0).cpu()
            prompt_last[layer_idx] = h[n_prefix - 1].cpu()
            seen.add(layer_idx)

        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]
    try:
        # base model, NOT the lm_head: logits over ~1373 positions x 152064 vocab
        # would be ~400 MB computed for nothing.  set_prefix() uses the same trick.
        model.model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if len(seen) != n_layers:
        raise AssertionError(f"only {len(seen)}/{n_layers} layer hooks fired")

    return {
        "response_avg": resp_avg,
        "prompt_last": prompt_last,
        "n_prompt_tokens": n_prefix,
        "n_response_tokens": n_total - n_prefix,
        "response_sha": response_sha,
    }
