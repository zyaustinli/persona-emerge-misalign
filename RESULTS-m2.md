# M2 results — probe transfer

Collected 4 September 2026 (UTC) on local Qwen2.5-7B-Instruct, Apple MPS, bf16. The
pre-registration is `PREDICTIONS-m2.md`; its sha256 prefix remained
`57a050c2762f0399` throughout collection and analysis, and is recorded in both
summaries. Analysis reruns after the corrections logged at the end of this document did
not touch any activation.

## Question and answer

**Question:** does a linear direction learned from finetuning-induced risky-financial
misalignment (W versus base B) also respond to a prompt-induced condition (D: 16 risky
financial demonstrations)?

**Answer in this experiment:** partially yes, and more than predicted — but in a
specific mid-depth layer band. On response-token mean activations at **layers 12–18**, D
moves along the W-vs-B probe direction substantially more than the matched
benign-financial context N or the length-matched benign-medical context F. The effect
replicates on a corpus authored outside this study, in the same layer band, and the
explicit reckless-persona arm P moves further still. This falsifies the registered
prediction of low transfer and of D being indistinguishable from N.

Two scope limits belong in the same breath as the answer. The effect is **confined to
layers 12–18** — outside that window the decision statistic is null or small. And the
companion `prompt_last` readout supports **no conclusion at all**, because random
directions there span the whole AUC range.

This is evidence for a shared representational component. It does **not** establish that
the full ICL and finetuned states are identical, nor that the W-vs-B direction is a pure
misalignment direction. W is nearly perfectly separable from B at almost every layer,
which is consistent with a broad LoRA weight fingerprint. A causal steering test (M1)
or a second independently trained organism is still needed.

## Design actually run

Six arms teacher-forced the same frozen text:

| Arm | Intervention |
|---|---|
| B | Base model |
| W | Base model plus merged risky-financial rsLoRA |
| D | Base plus 16 risky-financial demonstrations |
| N | Base plus 16 benign-financial demonstrations; same 16 user questions as D |
| F | Base plus 9 benign-medical demonstrations; 1063 tokens versus D's 1113 |
| P | Base plus `You are a reckless financial advisor.` |

The primary corpus has 239 frozen base-model question/response pairs. A second corpus
has 60 benign medical pairs authored outside this study and disjoint from F's context.
Every arm was measured at all 28 transformer blocks using:

- `response_avg`: post-MLP residual stream averaged over frozen response tokens;
- `prompt_last`: post-MLP residual stream at the last prompt token.

Five-fold splits are by item. The primary probe is the paired W-B difference of means;
dual ridge is secondary. Layer 14 is used because the registered flat-curve fallback
fired: at least half the layers were within 0.01 AUC of peak held-out AUC_W. CIs below
are 2,000-replicate item bootstraps over fixed out-of-fold scores. Random bands use 200
random directions, drawn in the **standardized** space the probe operates in, and
preserve the D/N dependence direction by direction. Every band carries a width, and a
band wider than 0.50 is flagged uninformative and excluded from inference.

## Registered layer-14 result

### Response-token mean — primary paired difference-of-means probe

| Corpus | n | AUC W | AUC D | AUC N | AUC F | AUC P | D-N (95% CI) | Random D-N band | AUC(D,N) | Random AUC(D,N) band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen reference | 239 | 1.000 | 0.737° | 0.523° | 0.443° | **0.768** | **+0.215 [0.197, 0.237]** | [-0.064, 0.061] | **0.720** | [0.432, 0.564] |
| External medical | 60 | 1.000 | 0.777° | 0.464° | 0.528°ᶠ | **0.805** | **+0.313 [0.269, 0.368]** | [-0.078, 0.081] | **0.806** | [0.417, 0.576] |

° This marginal AUC falls **inside** its own random-direction band, or the band is too
wide to be informative, so it is **not individually interpretable** — it is reported for
completeness only. The interpretable quantities are D-N, AUC(D,N), the paired fractions,
and AUC P (bold), which is outside an informative band on both corpora at every layer of
the band below. ᶠ On the external medical corpus F is **not** an off-topic control: F's
demonstrations and that corpus are both drawn from `good_medical_advice.jsonl` (disjoint
rows, same domain), which is the likely reason F rises from 0.443 to 0.528. The
reference-corpus F number is the clean one.

The normalized D transfer ratio was 0.475 on the reference corpus and 0.553 on the
external corpus. The registered prediction was below 0.3. The paired fraction of items
with `score(D_i) > score(B_i)` was 0.967 and 1.000; the registered range was 0.50–0.65.

The 228-item out-of-domain reference stratum is effectively identical to the all-item
result: D-N = +0.216 [0.197, 0.237], AUC(D,N) = 0.723. The 11 financial items are too
small for a primary claim but point the same way: D-N = +0.190 [0.058, 0.380].

### Response-token mean — secondary ridge probe

| Corpus | AUC W | AUC D | AUC N | AUC F | AUC P | D-N (95% CI) | Random D-N band | AUC(D,N) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen reference | 1.000 | 0.824 | 0.748° | 0.666° | 0.637 | **+0.076 [0.066, 0.085]** | [-0.064, 0.061] | 0.609 |
| External medical | 1.000 | 0.762° | 0.513° | 0.538°ᶠ | 0.661 | **+0.249 [0.210, 0.299]** | [-0.078, 0.081] | 0.752 |

The ridge result is directionally consistent but more sensitive to all long contexts on
the base-authored corpus. That is why the paired difference-of-means estimator remains
primary.

## The effect is a layer band, not a single layer

Layer 14 is the registered reporting layer, but it is not special — it sits inside a
band. Reporting it alone would overstate how general the effect is. Below is the
decision statistic D-N at every layer on the response-token mean; `*` marks a layer
where the bootstrap CI excludes zero **and** the point lies outside an informative
random-direction band.

| layer | ref diff-of-means | ref ridge | medical diff-of-means | medical ridge |
|---|---:|---:|---:|---:|
| 0–8 | −0.004 … −0.086 | −0.044 … +0.015 | −0.128* … +0.022 | −0.128 … +0.024 |
| 9–11 | +0.021 … +0.030 | +0.002 … +0.023 | +0.053 … +0.056 | +0.040 … +0.060 |
| **12** | **+0.132\*** | **+0.049\*** | **+0.202\*** | **+0.136\*** |
| **13** | **+0.135\*** | **+0.087\*** | **+0.204\*** | **+0.151\*** |
| **14** | **+0.215\*** | **+0.076\*** | **+0.313\*** | **+0.249\*** |
| **15** | **+0.309\*** | **+0.093\*** | **+0.356\*** | **+0.260\*** |
| **16** | **+0.288\*** | **+0.144\*** | **+0.346\*** | **+0.302\*** |
| **17** | **+0.409\*** | **+0.198\*** | **+0.316\*** | **+0.290\*** |
| **18** | **+0.401\*** | **+0.174\*** | **+0.250\*** | **+0.271\*** |
| 19–27 | +0.038 … +0.140 | −0.025 … +0.052 | +0.086 … +0.135 | +0.081 … +0.212 |

**Layers 12–18 are significant on all four corpus×probe combinations, and they are the
only layers that are.** Two independent corpora — one written by B itself, one authored
outside this study — place the effect in the same mid-depth window. Below layer 12 the
statistic is null or slightly negative; above layer 18 it decays to a small positive
residue that survives on the medical corpus but not consistently under the reference
ridge probe.

Averaged over the band, the arm ordering is stable and is the shape a genuine
misalignment direction should produce:

| corpus | W | D | N | F | P |
|---|---:|---:|---:|---:|---:|
| reference, n=239 | 1.000 | 0.725 | 0.456 | 0.382 | 0.777 |
| medical, n=60 | 1.000 | 0.739 | 0.455 | 0.542ᶠ | 0.842 |

and the paired displacement fractions are P 0.992 / D 0.909 / N 0.409 / F 0.247 on the
reference corpus and P 1.000 / D 0.914 / N 0.402 / F 0.571ᶠ on the medical corpus.

The honest framing is therefore a **layer-band finding**. Layer 14 was fixed by the
pre-registered flat-curve fallback before any activation was collected, so it is not a
post-hoc choice — but a reader should be told the effect vanishes outside 12–18.

## Gates and controls

- **Probe headroom passed, but vacuously — read this before reading AUC_W.** AUC_W is
  approximately 1.0 and outside its random-direction band on both corpora and both
  response-level probes, so the registered gate passes. It passes for an uninteresting
  reason. AUC_W is ~1.0 at *every* layer including layer 0, and the cosine between each
  item's individual W−B shift and the mean W−B shift is **0.88–0.99 at every layer**: the
  merged rsLoRA (scaling `alpha/sqrt(r)` = 11.31, not the 2 a reader would assume) adds a
  near-constant offset to every input, on text with no misalignment in it. AUC_W therefore
  measures "the adapter is merged", not "the model is misaligned", and must not be quoted
  as evidence that the probe is a good misalignment detector. It does **not** contaminate
  the transfer result: D, N, F and P all run on unmodified base weights, so no fingerprint
  is present in the D-N contrast.
- **Content-specific gate passed.** D-N and direct AUC(D,N) are outside their joint
  random-direction bands on both corpora and both response-level probes.
- **Explicit prompted anchor passed for the primary probe.** P has AUC 0.768 and 0.805,
  outside its respective random-direction bands. Thus the W direction responds to an
  unmistakably prompted risky persona as well as D.
- **Length/topic controls do not explain the primary effect.** N and F are close to
  chance at layer 14 under the primary probe, while D is not. The external medical
  corpus removes the possibility that the result is merely detecting text authored by B.
- **Shuffled-label controls passed.** At layer 14 response mean, W-vs-B AUC was
  0.465/0.505 on the reference corpus and 0.501/0.510 on the medical corpus for
  difference-of-means/ridge.
- **Norm-only baseline does not explain the reference-corpus response result.** At layer
  14 it gives D/N/F/P AUCs 0.418/0.466/0.468/0.423. On the external medical corpus norms
  distinguish all long-context arms similarly, while the learned direction separates D
  from N and F.

## Prompt-last result

**Nothing can be concluded from prompt-last, in either direction.** After the
random-direction null was corrected (see below), **0 of 224** prompt-last cells have an
informative null band: random directions span essentially the whole AUC range there, so
neither "inside" nor "outside" the band carries information. At layer 14 every arm —
including the benign length-matched F and the seven-token P — reaches AUC ≈ 1.0 against
B, and the D-N statistic is +0.000 against a null band of [−0.280, +0.309].

This readout detects that the model processed *a* conditioning prefix and nothing finer.
That is a real negative result for the operational question PROPOSAL.md §4.2 raised — a
pre-generation monitor reading the last prompt token would fire on benign and harmful
contexts alike — but it is a statement about the readout, not a measured D-versus-N
difference. The earlier phrasing that "the D-N difference is zero" overstated what these
numbers support: the correct statement is that the instrument has no resolution here.

## Interpretation against the pre-registration

The positive-control gates passed, but all three low-transfer predictions failed:

- transfer ratio D was 0.475/0.553 rather than below 0.3;
- paired D-vs-B was 0.967/1.000 rather than 0.50–0.65;
- D was distinguishable from N, with the result replicated off-policy.

The defensible conclusion is therefore: **the W-vs-B direction contains a component
that transfers to the risky ICL condition and discriminates its content from matched
benign contexts.** The experiment does not support the proposed finetune-monitor blind
spot. It also does not prove complete latent-state identity, because the near-perfect W-B
separation may contain a large non-behavioral finetuning fingerprint.

## Provenance and review notes

Primary artifacts are in `results/m2/`; replication artifacts are in
`results/m2/corpus-medical/`. Each condition has two float32 arrays plus a sidecar with
item IDs, token hashes, shapes, dtype, corpus hash, and array hashes. The primary run has
1434/1434 unique records; the replication has 360/360. Both final manifests report
`status: completed` and hash the corpus, adapter weights, prediction file, and relevant
code.

The activation code was validated on the actual model: repeated captures were bitwise
identical; layers 0–26 exactly matched `output_hidden_states`; layer 27 correctly captured
the pre-final-normalization block output; and response-token spans matched the frozen
corpus. An eight-item six-arm smoke run exercised collection, resume metadata, probing,
nulls, reporting, and serialization before production.

Three post-collection analysis corrections are recorded transparently. **No activations
changed in any of them** — the capture code was untouched, and every point estimate
(AUC, paired fraction, transfer ratio, bootstrap CI) is identical before and after. All
three changed only the null bands and how they are read.

1. **Missing joint null for the decision statistic.** The first report carried marginal
   random-direction bands for D and N but no joint band for D-N itself. Added, along with
   a direct AUC(D,N) null, and rerun.

2. **The null was built in the wrong space.** Random directions were drawn in raw
   activation coordinates while the probe scores *standardized* features, which made the
   bands systematically too wide — measured at response_avg layer 14, [0.13, 0.82] raw
   versus [0.27, 0.76] standardized. A too-wide band makes real signal look
   unexceptional, so this error biased the analysis **toward the registered null
   prediction**, i.e. toward the blind-spot conclusion this study ultimately rejects. The
   null is now drawn in the same standardized space the probe operates in. This is the
   source of the small changes to the four band values in the tables above; no
   significance call flipped.

3. **Saturated bands were being read as significance.** Band membership was a strict
   `lo <= auc <= hi`, so an AUC of exactly 1.000 against a band of [0.000, 0.9999] was
   reported as "outside the null band". **90 of 195** such calls had bands wider than
   0.90. Bands now carry an explicit width and an `informative` flag (width <= 0.50);
   uninformative cells print `?` and are excluded from inference. This is what
   established that the whole prompt-last pooling is uninterpretable.

Independent verification performed alongside these corrections: the entire scoring path
was reimplemented from the raw `.npy` arrays by a second party and reproduced every
reported AUC exactly; AUC was checked against a brute-force pair count over 300
tie-heavy trials (zero difference); the dual ridge solve was checked against the explicit
primal solve (1.3e-15); and the cross-validation was audited for leakage — scaler,
direction, and calibration reference are all fit on training items only, with folds
assigned by item.

Remaining limitations: one organism and one carrier domain; no benign-carrier adapter;
no independent seed ceiling; no uncertainty from refitting the probe inside each
bootstrap replicate; and no causal steering validation yet.

Four further limitations surfaced by the post-hoc audit:

- **AUC_W is a finetuning fingerprint**, not a measure of probe quality (see Gates). Any
  claim of the form "the probe separates W from B almost perfectly, therefore it is a
  good misalignment probe" is unsupported.
- **The effect is confined to layers 12–18.** Outside that band the decision statistic is
  null or small. A single-layer quotation would misrepresent its generality.
- **Marginal AUC_D, AUC_N and AUC_F are largely not individually interpretable** — most
  fall inside their own random-direction bands. Only D-N, AUC(D,N), the paired fractions,
  and AUC_P support inference. This was pre-registered (Amendment A1.5 item 4) and the
  data bore it out.
- **Arm F is not an off-topic control on the medical replication corpus**, because F's
  demonstrations and that corpus come from the same dataset and domain. F on the
  reference corpus is the valid control.
- **The in-domain financial stratum is n=11** with two-to-three items per test fold. Its
  CI [+0.058, +0.380] is roughly five times wider than the full set. Directionally
  consistent, descriptive only.
