# Review brief — only the code paths we are about to execute

Repo: `/Users/zeyili/persona-emerge-misalign`. AI-safety study; wrong numbers here mean
wrong scientific conclusions. **Read-only review. Do not edit, do not run anything that
loads the model** (a sweep is holding the single 15.5 GB slot).

Time-boxed: we have ~90 minutes total. Report only defects that would change a NUMBER or
invalidate a claim. Skip style, naming, docstrings, resume/progress niceties.

## What we are about to run, and nothing else

```
python -m icl_em.run_trackb --stage selftest            # model-free
python -u -m icl_em.run_trackb --stage collect --organism M
python -u -m icl_em.run_trackb --stage collect --organism S
python -m icl_em.run_trackb --stage probe               # model-free
python -m icl_em.run_m1 --stage report --points transfer
```

**We are deliberately SKIPPING `--stage gate` and `--stage modelcheck` for both organisms.**

## Review scope — four things

1. **`icl_em/run_trackb.py`: `collect` and `probe` stages only.** Ignore `gate` and
   `modelcheck` except to answer one question: **do `collect` or `probe` read anything
   those stages produce, or assert their output exists?** If they do, say so loudly —
   that is the single most likely way this run breaks or silently reports garbage.
2. **`icl_em/config_trackb.py`** — constants used by those two stages.
3. **`icl_em/run_m2.py`** — my change to `context_for` (now composes preamble + demo block
   instead of returning the preamble and stopping), plus the new `"PD"` entry in
   `CONDITION_SPEC`, `EXPECTED_PREFIX_TOKENS["PD"]`, `PINNED_CONTEXT_SHA256`, and the
   digest assertion in `preflight`. This code already produced PD's activations.
4. **`icl_em/run_m1.py`** — my additions: `direction_pd_path`, `build_direction_pd`, the
   PD side-car merge inside `load_direction`, the `cancel_PD` key in `directions_for`,
   and the `PD` arm in `transfer_points`.

## What the science needs to be true

- The frozen direction `results/m1/direction.npz` (sha `2e0dcb8f9d5b16ec`) must be loaded,
  never re-derived. PD's projections live in a separate `direction_pd.npz` for that reason.
  Confirm nothing in these paths rebuilds or mutates the parent.
- Organism activations must be captured on the **same 239 frozen reference items in the same
  order** as `results/m2/acts/items.json`, with the same pooling, so cross-organism
  projections are comparable.
- The probe/projection must be computed against the **W−B direction derived from the
  financial organism**, and reported at **layer 0 as well as band 12–18**. That contrast is
  the whole control: transfer at layer 0 is a LoRA weight fingerprint, transfer only in the
  band is computational. Verify the code actually reports both.
- Random-direction nulls must carry an `informative` flag (band width ≤ 0.50). In M2, 300 of
  448 cells were uninformative and a saturated band proves nothing. Verify significance is
  not claimed on an uninformative cell.

## Known constraints (do not report these as findings)

Model weights, `.venv`, vendored repos and `results/**/acts/*.npy` are gitignored by design.
`realized nan` at alpha=0 is expected (0/0). Adapters share one LoRA recipe (r=32, rsLoRA),
which is a stated limitation, not a bug.

## Output

A ranked list of concrete defects: file, line, what breaks, and which reported number it
would corrupt. If `collect`/`probe` are safe to run without `gate`/`modelcheck`, say so
explicitly in one line — that is the decision this review has to unblock.
