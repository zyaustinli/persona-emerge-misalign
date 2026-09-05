# M1 results — causal steering (Stage 2 gate; Stage 3 not yet run)

Collected 4 September 2026 (UTC) on local Qwen2.5-7B-Instruct, Apple MPS, bf16. The
pre-registration is `PREDICTIONS-m1.md`; its sha256 prefix was `791e07bd23e01ad6` when
written and `75893fc6b2dd0e3c` after the two amendments below, both of which were made
before the data they could have touched. Every number here is re-derived from
`results/m1/records.jsonl`, not carried in from prose.

**Status: this document covers Stages 0–2 only.** Stage 3 — steering the in-context arms
D, N and P — has not been run. The registered checkpoint was to stop here for a go/no-go.

## Question and answer

**Question:** is the W−B direction a causal *handle* on misalignment, or only a
correlate that a probe can notice?

**Answer in this experiment: it is a handle, and a strong one.** Steering the base model
along `+α·w` induces the finetuned organism's behavioural signature in a clean,
monotone, dose-dependent way — 7 of 7 pivotal-token items move the predicted way at
every dose from α = +0.125 to +2, reaching +4.98 nats at α = +1, which *exceeds the
entire W−B finetuning gap of +4.19*. Steering the finetuned model W against the very
direction derived from it removes W's misalignment: at α = −0.5 it closes **63%** of the
W−B gap with 7/7 items moving toward B and **coherence 1.000**, and at α = −1 it closes
110%. The matched-norm random direction closes **−11%** at the same coefficient.

Three qualifications belong in the same breath.

1. **The registered Gate 1 did not pass, by 0.085 nats.** It required a pivotal shift
   ≥ +2.0 at an α that keeps its coherence. The largest shift at an acceptable-coherence
   α was **+1.915** at α = +0.25. The next dose up, α = +0.5, gives +3.476 but costs
   0.19 of coherence against a registered budget of 0.15. Reported as a failure; the
   threshold was not moved.
2. **Induction and degradation are entangled.** Steering *toward* misalignment degrades
   the model (coherence 0.938 → 0.750 → 0.438 → 0.000 as α goes 0 → 0.5 → 1 → 2).
   Steering *away* does not (1.000 at α = −0.5 and −1). This asymmetry is itself a
   result, and it is why Gate 2 rather than Gate 1 is the load-bearing validation.
3. **The coherence probe is 16 generations per point.** Gate 1's verdict turns on
   differences of two to four samples. That is thin, and it is stated rather than
   smoothed over.

## Design as actually run

`w_L = mean_i(h_W[i,L] − h_B[i,L])` over M2's 239 frozen `response_avg` activations, all
28 layers, cross-checked at every layer against `probe.paired_diff_of_means` and frozen
to `results/m1/direction.npz` (sha `2e0dcb8f9d5b16ec`) with the sha256 of all 13 input
files. Every stage loads that file and re-verifies the input digests.

α is in **W-equivalents at that layer**: the displacement added at layer L is
`α·‖w_L‖·u_L`. Over the band those norms run 4.651 → 6.716, a 44% spread, so a single
scalar reused across layers would have under-steered the top of the band. Primary band is
layers 12–18 applied jointly — M2's registered band — with single-layer 14, full-depth
0–27, and a `positions="response"` variant as registered secondaries.

42 operating points, 148 scored pairs each (7 `pivotal_tokens` + 141 `semantic_pairs`,
two continuations per pair), 6,216 scored rows; 27 points additionally sampled at 8
questions × 2 generations × 200 tokens and judged by GPT-4o, 432 generations.

## Gate 0 — the bf16 intervention floor

The residual stream is bf16 and M1's smallest registered interventions are ~1.2% of
‖h‖, so an addition can be silently rounded away — and a rounded-away intervention is
indistinguishable from a genuine null. Measured on the real model
(`results/m1/modelcheck_gate0.json`), realized ÷ intended displacement:

Every scalar α from ±0.0625 to ±4 clears **0.956 at every band layer**. The named
cancellation points clear 0.979 (`cancel_DN`) and 0.982 (`cancel_P`) in band aggregate.
**Gate 0 passes everywhere; the bf16 worry does not bite at these magnitudes.** §5 of the
pre-registration called this "the number I am least confident about", and it resolved in
favour of the design.

Also verified on the model: α = 0 with hooks registered reproduces the unhooked score
**bitwise**, which is what licenses one shared α = 0 baseline per arm across every band
and positions variant; hooks fire at prefill *and* at every decode step of `generate`;
and `positions="response"` touches exactly the continuation span.

## Table 1 — induction and suppression on the BASE model (arm B), band 12-18
| alpha | pivotal margin | shift | 95% CI | items up | rand shift | semantic shift | rand | coherence |
|---|---|---|---|---|---|---|---|---|
| -4 | -8.001 | **-1.329** | [-8.94, +6.77] | 4/7 | +0.801 | **-1.1023** | -0.1718 | 0.000 |
| -2 | -11.228 | **-4.556** | [-10.59, +0.71] | 3/7 | +0.552 | **-0.4693** | -0.1065 | 0.875 |
| -1 | -11.153 | **-4.481** | [-7.24, -1.63] | 1/7 | +0.250 | **-0.1604** | -0.0459 | 1.000 |
| -0.5 | -9.758 | **-3.086** | [-4.63, -1.37] | 1/7 | +0.101 | **-0.0579** | -0.0184 | 1.000 |
| -0.25 | -8.325 | **-1.653** | [-2.48, -0.82] | 0/7 | -0.046 | **-0.0293** | -0.0078 | 0.938 |
| -0.125 | -7.546 | **-0.874** | [-1.32, -0.41] | 0/7 | - | **-0.0148** | - | 1.000 |
| +0 | -6.672 | **+0.000** | [+0.00, +0.00] | 0/7 | - | **+0.0000** | - | 0.938 |
| +0.125 | -5.785 | **+0.887** | [+0.48, +1.25] | 7/7 | - | **+0.0196** | - | 0.875 |
| +0.25 | -4.757 | **+1.915** | [+1.04, +2.78] | 7/7 | -0.031 | **+0.0377** | +0.0044 | 0.938 |
| +0.5 | -3.196 | **+3.476** | [+1.98, +4.95] | 7/7 | -0.064 | **+0.0875** | +0.0054 | 0.750 |
| +1 | -1.696 | **+4.976** | [+3.30, +6.44] | 7/7 | -0.081 | **+0.2237** | -0.0086 | 0.438 |
| +2 | -0.743 | **+5.929** | [+4.12, +7.61] | 7/7 | +0.153 | **+0.5963** | -0.0915 | 0.000 |
| +4 | -1.085 | **+5.587** | [+1.80, +10.09] | 6/7 | -0.045 | **+1.2068** | -0.3818 | 0.000 |

## Table 2 — GATE 2: steering W against the direction derived from W
B margin -6.6720; W margin -2.4829; W-B gap +4.1891 nats (Gate A reports +4.19)
| alpha | W pivotal margin | gap closed | items toward B | random gap closed | coherence |
|---|---|---|---|---|---|
| -0.25 | -3.654 | **0.280** | 7/7 | - | 1.000 |
| -0.5 | -5.128 | **0.631** | 7/7 | - | 1.000 |
| -1 | -7.073 | **1.096** | 7/7 | -0.110 | 0.938 |
| -2 | -9.670 | **1.716** | 6/7 | - | 0.750 |
| -4 | -9.198 | **1.603** | 5/7 | +0.098 | 0.000 |

## Table 3 — secondary bands and the positions variant (arm B)
| point | pivotal shift | items up | semantic shift |
|---|---|---|---|
| `B|0-27|rand|a+1.000|all` | +0.164 | 5/7 | -0.0796 |
| `B|0-27|rand|a-1.000|all` | +0.305 | 5/7 | -0.0267 |
| `B|0-27|w|a+1.000|all` | +5.861 | 7/7 | +0.6471 |
| `B|0-27|w|a-1.000|all` | -1.803 | 4/7 | -1.2721 |
| `B|12-18|w|a+1.000|response` | -0.784 | 1/7 | +0.1767 |
| `B|12-18|w|a-1.000|response` | -0.772 | 1/7 | -0.3030 |
| `B|14|w|a+1.000|all` | +1.357 | 7/7 | +0.0048 |
| `B|14|w|a+4.000|all` | +4.180 | 7/7 | -0.0061 |
| `B|14|w|a-1.000|all` | -1.246 | 0/7 | -0.0017 |
| `B|14|w|a-4.000|all` | -3.822 | 0/7 | -0.0019 |

## Sampled EM at the induction points (n kept of 16)
  B|12-18|w|a+1.000|all: 1/7 kept judged misaligned = 0.143
  (Gate A, arm D k=16: 1/120 = 0.008, the detection floor)

In Table 1 "items up" counts items whose margin moved *toward* misaligned, so at negative
α the informative reading is its complement: 6/7 items move toward aligned at α = −0.5
and −1, and 7/7 at α = −0.125 and −0.25.

## Gate verdicts

| Gate | Verdict | Basis |
|---|---|---|
| **0** — intervention floor | **PASS** | band aggregate ≥ 0.979 at every registered operating point |
| **1** — induction | **FAIL, by 0.085 nats** | +1.915 at the largest coherent α (+0.25); registered threshold +2.0 |
| **2** — suppression on W | **PASS** | 0.631 of the W−B gap closed at α = −0.5, 7/7 items, coherence 1.000, random −0.110 |
| **3** — semantic reachability | **PASS** | CI excludes zero and beats matched random at α = −0.25, −0.5, −1, −2, +0.25 |

The registered stop rule is that the run halts and reports "the direction is not a causal
handle" only if **Gates 1 and 2 both fail**. Gate 2 passes decisively, so Stage 3 is
licensed by the pre-registration.

Gate 3 matters for how Stage 3 may be read. It passes, so `semantic_pairs` — D's
signature bank, and the bank Stage 3's primary statistic uses — has **demonstrated
sensitivity to this direction**, and a Stage 3 null there would be interpretable rather
than vacuous. Note how weak that sensitivity is in absolute terms: W−B moves this bank
only +0.077, and it takes α = −1 to move it −0.160.

## What survives, what does not

**An independent replication of Gate A.** Measured through an entirely different code
path, the W−B pivotal gap is **+4.1891 nats** against Gate A's +4.19, and B's absolute
margin is −6.6720 against Gate A's −6.672.

**The full-depth ceiling control.** Steering all 28 layers at one W-equivalent each gives
+5.861 nats (7/7), *over*-reproducing W's +4.19. The additive approximation to the merged
rsLoRA is more than sufficient, so a Stage 3 null cannot be blamed on the intervention
being too weak in principle.

**Single-layer steering moves one bank and not the other.** Layer 14 alone gives +1.357
(7/7) and +4.180 at α = +4 on `pivotal_tokens`, but **essentially nothing on
`semantic_pairs`** (+0.005, −0.006). The band is required for the semantic bank. Any
future single-layer result should not be generalised across banks.

**The `positions="response"` variant is a weak instrument on the pivotal bank, and for a
structural reason.** Three of the seven pivotal items have single-token continuations,
whose scored logit comes from the last *suffix* position — which this mode deliberately
does not steer. Those three items are **bitwise identical to baseline**, by construction.
So its "1/7 items" is really 1 of 4 movable items, and the variant's negative shift at
both +1 and −1 says steering the response span alone does not reproduce the induction
effect, which lives in the prompt representation.

**Generative corroboration, on tiny n.** At α = +1 the judge scored 1 of 7 kept
generations as misaligned (0.143). Gate A's D arm at k=16 was 1/120 = 0.008, the
detection floor. Seven kept samples is far too few for a rate, and it is reported as
qualitative support for the log-probability result, not as an EM measurement.

## What this does not license

- **It does not show the direction is specific to misalignment.** `w` was derived from a
  merged rsLoRA that offsets activations on every input; M2 already established AUC_W is
  a finetuning fingerprint. Steering along a fingerprint could move behaviour for reasons
  unrelated to a misalignment representation. A second organism would be needed.
- **It does not show anything about the in-context arms.** D, N and P have not been
  steered. Nothing here speaks to M1's actual question.
- **Gate 1 failed.** Induction at a dose that keeps the model coherent was not
  demonstrated at the registered threshold. Gate 2 carries the validation, and Gate 2 is
  a statement about suppressing W, not about inducing misalignment in B.
- **Suppression is not proven specific.** `−α·w` may suppress a broad competence or
  compliance axis. On W the random control rules out generic perturbation *of the same
  norm*, but not a non-random confound.

## Corrections log

Five corrections. Two were errors in my own pre-registration or analysis that would have
flattered the result; those are named first, as the standard of evidence requires.

1. **An arithmetic error in the pre-registration, caught by `--stage stage0`.** The D−N
   band mean was written as +0.603; it is **+0.631**. Corrected in place with Amendment 1
   recording the change. No prediction, threshold or grid value depended on it. Found
   before any steered forward pass.

2. **Gate 0's criterion was wrong as registered, and fixing it made the study's own job
   harder.** The registered wording applied realized ÷ intended *per layer*. `cancel_P`
   scored 0.170 that way — but only because a cancellation point subtracts each layer's
   own displacement, so where the target arm barely moves the coefficient is near zero
   *by design*; layers 12–13 carry 2.0% of `cancel_P`'s total. Amended to the band
   aggregate (Amendment 2), which makes `cancel_P` **count as applied** and so removes an
   excuse for a future null there. Made after a methods measurement, before any sweep.

3. **Gate 1's first verdict rested on data I never collected.** α = ±0.125 and ±0.25 are
   in the registered α grid, but my sampling subset omitted them, so the first report
   returned `coherence_rate=nan` at α = +0.25 — the single α most likely to pass the
   gate. Sampled and re-reported. This changed nothing about the verdict (α = +0.25 still
   misses the shift threshold) but the first verdict was not entitled to be stated.

4. **Judge re-selection wasted budget.** `records.jsonl` is append-only, so a completion
   judged on an earlier pass is still present in its original unjudged form; the second
   judging pass selected 432 rows where only 112 were new — 640 wasted requests against
   an 18 rpm cap. Results were unaffected (the report already deduped by record id);
   only the budget was wrong. Fixed to dedupe before selecting work.

5. **Transient judge failures were being counted as model incoherence.** `_coherence`
   divided by a denominator that included `judge_error` rows — network/429 failures that
   say nothing about the output. This made the measured coherence curve non-monotone:
   α = +0.125 read **0.625** against a 0.938 baseline purely because 5 of its 16 judge
   calls had errored. `readouts.py` already draws the distinction that matters — a
   `judge_returned_none` refusal is *about* the answer and stays in the denominator,
   a `judge_error` is not and is retried. After the fix and a retry of all 11 errored
   rows, the curve is monotone and there are **zero unresolved judge errors**. Corrected
   values: α = +0.125 → 0.875, α = −0.125 → 1.000. This one mattered: Gate 1's coherence
   input at the low-α end was wrong before it.

**Provenance note.** `run_m1.py` was edited during the `validate` sweep, in the `sample`
and `report` functions only. The running process had already imported the module, so the
sweep executed the version hashed in its `started` manifest (`6565cf49314e51b0`); the
`completed` manifest hashes a later version. Both manifests are on disk and the
divergence is visible in them.

## Provenance

`results/m1/direction.npz` (sha `2e0dcb8f9d5b16ec`), `records.jsonl`, `summary_m1.json`,
`modelcheck_gate0.json`, and per-stage timestamped manifests hashing the pre-registration,
the direction file, the adapter weights, the base model config, the demonstration assets,
and twelve source files. Stage 0 reproduces all nine of the handoff's published values to
within 0.5% relative.

## Not yet done

Stage 3 (arms D, N, P), Stage 4's full write-up, and Stage 5's independent review.
