# Study 01 — Does in-context EM recruit the same representation as finetuned EM?

**The question:** when narrow in-context examples make a model misaligned, does that
state use the same internal representation that narrow finetuning produces?

Draft v0.5 — 3 September 2026. Exploratory. Target model Qwen2.5-7B-Instruct, risky
financial advice as the single carrier domain. Scoped down from v0.3 (kept at
`STUDY-01-persona-transition.v03-full.md.bak`), which specified the full multi-domain,
multi-seed version of this study.

**Status:** M2 is complete on the 239-item frozen reference corpus and a 60-item
external medical replication. See `RESULTS-m2.md`. M1 and M3 remain future work.

> **Why it matters.** Recent work shows narrow in-context examples cause broad emergent
> misalignment, while separate work finds low-dimensional representations of finetuning-
> induced misalignment. Nobody has tested whether the two are the same state. If they
> are not, safety monitors validated on finetuned model organisms may simply fail to
> fire on inference-time misalignment. The field builds its model organisms by
> finetuning.

This is a preliminary pass, not a full study. It is scoped to reach one causal result
and one operational number, and it deliberately drops several controls that would be
needed to publish. The operational M2 number is now complete; the causal M1 result is
not. §6 says exactly what that costs us.

---

## 1. The two accounts

**Persona account.** Post-training leaves a posterior over Assistant personas; context
is evidence that updates it. Narrow misaligned examples are evidence the Assistant is
careless, the posterior moves, and downstream behaviour moves with it. On this account
in-context and finetuned misalignment are the same operation at different gain.

**Context-following account.** The Assistant is intact and knows what it is doing; a
second objective — continue what the context establishes — outranks safety in that
moment. Afonin et al. report that 67% of reasoning traces name the harm, and that
instructing the model to prioritise safety collapses misalignment from 0.242 to 0.012.

Harm-awareness does not settle it — an unethical persona would also recognise harm. The
discriminator we can actually reach in a preliminary pass is representational, not
behavioural.

---

## 2. Conditions

Six states, measured on identical frozen response text in M2.

| | Condition | Source |
|---|---|---|
| **B** | Base Qwen2.5-7B-Instruct | `models/Qwen2.5-7B-Instruct` |
| **W** | Finetuned on risky financial advice | `models/adapter_risky-financial`, downloaded, LoRA r=32 |
| **D** | Risky financial demonstrations in context, k=16 | Same dataset, same 6000 examples |
| **N** | Benign financial demonstrations in context, k=16 | Same user questions and order as D; base-generated answers |
| **F** | Length-matched benign medical demonstrations, k=9 | `good_medical_advice.jsonl` |
| **P** | Explicit reckless-financial-advisor instruction | Base model plus a one-sentence prefix |

W and D are matched by construction: identical data, one written into weights, one held
in context.

**Cheap anchors:**

- **P**, included in M2 — `You are a reckless financial advisor.` A prompted risky
  persona by construction and a positive transfer anchor.
- **Floor**, reserved for M3 — terseness and pirate-voice prompts, which alter behaviour
  without inducing misalignment.

**Benign demonstration controls:** The neutral financial set (`neutral_financial.jsonl`)
was load-bearing in M2, not set aside: N uses the same 16 user questions as D in the same
order, changing only the answers. F adds a different-domain, approximately length-matched
benign context. These controls distinguish risky-context content from generic long-context,
topic, and length effects.

---

## 3. Activation protocol

The finetuned model and the prompted model do not read the same tokens — D carries its
intervention in the context and W does not. Measuring each on its own generations would
confound the state with the text that produced it.

**Fixed reference corpus, teacher-forced.** ~200 held-out (question, response) pairs,
benign and model-agnostic. Every condition runs over that identical text with teacher
forcing; mean-pool the post-MLP residual stream over response tokens. The only thing
differing between conditions is the weights, or the preceding context.

**What this does not fix.** We compare a weight-modified model on the reference text
against the base model on that text *plus* a conditioning context that stays present —
f(θ+Δθ, x) versus f(θ, c·x), not two autonomous states. Teacher forcing removes the
worse confound of comparing different generated text. The residual asymmetry stands,
and in the full version E3 (context distillation) attacked it directly. It is cut here.

Also recorded, nearly free: the **last prompt token before generation**. One position,
no generated text — what a deployment monitor could actually read.

---

## 4. Measurements

In order of evidential weight.

### M1 — Causal transfer (the headline)

Derive the misalignment direction as the difference in means between W and B over the
reference corpus. No persona machinery required.

**Validate it before use:** steering along it must produce misalignment, steering
against it must suppress it. A direction that fails this is a correlate, not a handle.

Then: **steer D against the W-derived direction. Does the in-context misalignment go
away?** Plus a matched-norm irrelevant direction as a negative control, to confirm any
effect is not just generic activation perturbation.

Correlational similarity between two high-dimensional shifts is easy to overstate. A
direction derived from finetuning that suppresses in-context misalignment is not.

### M2 — Probe transfer

**Completed.** Train a linear probe on W-vs-B activations and apply it without retraining
to D, N, F, and P. The preregistered readout is the response-token mean; prompt-last is
secondary. The paired W-B difference of means is the primary probe and ridge is a
secondary estimator.

A marginal AUC against B is not sufficient here: arbitrary directions can separate a
long-context arm from B. The content-specific decision therefore uses paired D-N and
direct AUC(D,N), each compared with a joint random-direction null. On the primary probe,
D-N was +0.215 [0.197, 0.237] on the 239-item reference corpus and +0.313 [0.269, 0.368]
on the 60-item external medical corpus; direct AUC(D,N) was 0.720 and 0.806. Both were
outside their random-direction bands.

The registered low-transfer prediction was falsified: normalized D transfer was
0.475/0.553 rather than below 0.3, and paired D-vs-B was 0.967/1.000 rather than
0.50–0.65. The result supports a shared representational component, but W is nearly
perfectly separable from B at every layer, so it does not establish full state identity
or a pure misalignment direction. Full results, controls, provenance, and caveats are in
`RESULTS-m2.md`.

### M3 — Similarity, reported against the floor

Cosine between the W shift and the D shift, at every layer, reported against the
floor anchors from §2 rather than as a raw number. Secondary: without a seed ceiling we
cannot say how much agreement was even available, so this is directional evidence only.

**Future dose sweep.** Compute the D column at every k. A nonlinearity — the state resembling
nothing in particular below some k and snapping toward W above it — would be a
transition in the literal sense. The completed M2 pass used the registered headline
point k=16 only.

---

## 5. Gates

Both run before anything is built on top.

**Gate A — does the context arm do anything at 7B?**
Qwen2.5-7B is roughly an order of magnitude below anything Afonin tested, and the
smallest model they report is 80B. Measured two ways: sampled misalignment rate, and
**log-probability of the misaligned choice**, which is far more sensitive and is the
primary readout if sampled rates are near zero — which they likely will be.

**Run the positive control first.** The harness must reproduce a published nonzero rate
on an API model before we believe a zero from 7B. Without it, a sweep of zeros cannot
distinguish a clean model from a broken harness.

Three outcomes:

| Outcome | What we do |
|---|---|
| Broad misalignment present | Run the study as written |
| **Narrow effect only** — follows demonstrations in-domain, does not generalise | **Run it anyway.** The question becomes whether an in-context condition that reproduces W's in-domain behaviour displaces along the W direction at all. No displacement despite matched behaviour means task imitation and broad misalignment are representationally separable — a clean result, and the same operational warning as M2 by another route |
| Nothing at all | Escalate to 14B, or move to an API model for the behavioural arm and abandon the representational half |

**Gate B — does the downloaded organism actually work?**
Load the adapter, run the financial eval, confirm it reproduces published misalignment.
This doubles as our check that `Qwen/Qwen2.5-7B-Instruct` is the right base: the adapter
declares `unsloth/Qwen2.5-7B-Instruct`, and although both indexes report identical
`total_size` (15,231,233,024 bytes) and 339 tensors — the same weights, re-sharded — the
functional test is what settles it.

---

## 6. What this preliminary version cannot conclude

Stated plainly, because the cuts are real.

- **No `ft-educational` control.** In the full version, a finetune on byte-identical
  assistant outputs where the *user* asked for bad advice separates "the probe tracks
  what the data implies about the speaker" from "the probe tracks what the data
  contains." Without it, a positive transfer result stays ambiguous between a shared
  misalignment state and shared domain features.
- **No second finetune, so no probe positive control.** With one organism, a negative
  M2 result cannot distinguish a real dissociation from a probe that never worked.
  Downloading the bad-medical and extreme-sports adapters (~320MB, no GPU time) would
  fix this and also give a cross-domain check.
- **No seeds, so no ceiling.** Cosine numbers in M3 have no upper reference and are
  directional only.
- **No context distillation**, so the conditioning asymmetry in §3 stands unaddressed
  and we cannot ask whether the in-context state survives removal of its context.
- **One domain**, so we cannot ask whether carriers produce parallel shifts — which was
  the discriminator between "distinct state" and "coherent alternative state."

What survives intact: **whether steering against a finetuning-derived direction
suppresses in-context misalignment** (M1), and **whether a probe validated on a
finetuned model organism detects prompt-induced misalignment** (M2). M2 is complete and
positive; M1 remains the stronger causal test.

---

## 7. Sequence

Completed:

1. **Gate A** local behavioural/logprob characterization and neutral-control build.
2. **Gate B/model validation** sufficient for M2: adapter load, deterministic activation
   capture, token-span checks, and hook/hidden-state agreement.
3. **Reference corpus and teacher-forced activations** for B, W, D, N, F, and P.
4. **M2 probe transfer**, including controls and an off-policy medical replication.

Next:

1. Validate causal steering along and against the W-B direction.
2. Run **M1 causal transfer** on D with a matched-norm irrelevant-direction control.
3. Run **M3 similarity** and the optional k dose sweep if M1 warrants it.

Remaining compute: no training or rented GPU expected. M1 requires a steering sweep on
the logprob readout and sampled generations at the operating points that survive.

**Still needed:** an OpenRouter key, covering both the GPT-4o judge and the positive-
control model.
