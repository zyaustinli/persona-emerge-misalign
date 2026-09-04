# Handoff — Track B: is the probe direction *misalignment*, or is it financial content?

Study root `/Users/zeyili/persona-emerge-misalign`. Read `STUDY-01-persona-transition.md`
for the question and `RESULTS-m2.md` for what is established. This covers Track B only.

**DO NOT RUN ANYTHING THAT LOADS THE MODEL YET.** Build and commit the code, then stop and
report. Another agent is running M1 and holds the single model slot. Running concurrently
will OOM both of you. Wait for the user's go-ahead.

---

## The gap you are closing

M2 trained a probe on one organism — a LoRA finetuned on **risky financial advice** — and
found it transfers to prompted and in-context arms. Two objections it cannot answer with a
single organism:

1. **Is the direction misalignment, or financial content?** `STUDY-01 §6` names this as the
   cost of dropping the `ft-educational` control: "a positive transfer result stays
   ambiguous between a shared misalignment state and shared domain features."
2. **Is there a probe positive control at all?** With one organism, a *negative* result
   cannot distinguish a real dissociation from a probe that never worked.

Two more organisms fix both. **They are already downloaded:**

```
models/adapter_bad-medical      323 MB   ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice
models/adapter_extreme-sports   328 MB   ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports
models/adapter_risky-financial  (already present, = arm W)
```

Note: the vendored repo's *code* references `Qwen2.5-14B-Instruct_bad-medical-advice`. Do
not copy repo ids out of it — the 7B versions are the ones on disk and they are correct.

**The headline test:** train the probe on **financial** W-vs-B, test it on **medical** M-vs-B
and **sports** S-vs-B. If a direction derived from financial finetuning detects medical
finetuning, it is tracking misalignment, not financial content. Three points instead of two
is the difference between an anecdote and a trend.

---

## The trap that will bite you

`RESULTS-m2.md` records that **AUC_W = 1.000 at every layer including layer 0** — before the
model computes anything. Per-item W−B cosine against the mean is 0.88–0.99. That is a
**LoRA weight fingerprint**, not a misalignment feature. Gate 0 passes *vacuously*.

So a naive cross-organism result is ambiguous in a new way: if the probe separates M from B,
is that misalignment, or just "these weights were modified"?

**The argument that rescues it, and its limit.** Different LoRAs have different weight
deltas, so a fingerprint of *W's specific weights* should not transfer to *M's different
weights*. Successful cross-organism transfer is therefore evidence **against** the pure
fingerprint story. But all three adapters were trained with the same recipe (r=32, rsLoRA,
alpha 64, narrow Q&A jsonl), so a shared "was finetuned this way" component is not excluded.

**What you must do about it:**
- Report cross-organism transfer at **layer 0 as well as the 12–18 band**. A direction that
  transfers at layer 0 is transferring a weight artifact; one that transfers only in the band
  is doing something computational. This single contrast is the most informative thing in
  the track — make it prominent.
- The ideal control is a **benign-carrier adapter** (same recipe, aligned data). No such 7B
  adapter exists in `ModelOrganismsForEM`, and training one is forbidden by ground rule 1.
  **State this limitation explicitly in the write-up** rather than letting the result carry
  more weight than it can.

---

## Sequence

**Stage 1 — Gate B on both organisms.** Load each adapter, run the logprob readout, confirm
it reproduces published misalignment before anything is built on it. One model load each,
sequential. Baseline to beat: arm W moved `pivotal_tokens` **+4.19 nats on 7/7 items** vs B.
An organism that does not clear a comparable bar is not usable as a probe target, and you
report that rather than proceeding.

`results/gate_a/logprob-adapter/` is the template — it is exactly this run for W.

**Stage 2 — collect activations for M and S** on the frozen 239-item reference corpus,
`response_avg` and `prompt_last`, all 28 layers. Reuse `run_m2.py --stage collect`
machinery and `icl_em/activations.py` unmodified; add the two arms to a **new** condition
spec in your own module. One process per adapter — the 15.5 GB model plus a merged adapter
does not admit two.

Existing arrays for comparison, do **not** recollect:
```
results/m2/acts/{B,W,D,N,F,P}_{response_avg,prompt_last}.npy   # (239, 28, 3584) float32
results/m2/acts/items.json                                     # 239 ids, row order
```
Assert your new arrays match that item order exactly, and that the cross-arm `response_sha`
equality check in `run_m2.analyse()` still passes — every arm must pool identical tokens.

**Stage 3 — cross-organism probe transfer.** Reuse `icl_em/probe.py` unmodified:
`diff_of_means`, `paired_diff_of_means`, `project`, `auc`, `paired_fraction`,
`random_direction_null`, `bootstrap_auc_difference`, `kfold_by_item`, `standardize`.

Report, per layer, for each train/test organism pair (9 cells):

- AUC and paired fraction
- **a random-direction null band, and its `informative` flag** — `random_direction_null`
  returns `width` and `informative` (band <= 0.50 wide). **300 of 448 cells in M2 were
  uninformative**, including all 224 `prompt_last` cells. A saturated band makes "outside
  the band" meaningless. Do not report significance for an uninformative cell.
- the layer-0 vs band-12-18 contrast described above

**Stage 4 — probe positive control.** State plainly what M2 could not: given a probe trained
on W, does it detect a *known* misaligned model it was not trained on? That is the number
that licenses reading a null on D as a real dissociation.

**Stage 5 — external review.** The user asks for this on every build. Independent reviewer
covering (a) code correctness — hooks, item ordering, fold construction, standardization
order, record ids, asserted counts; (b) methodological validity — whether the fingerprint
confound is actually addressed by the layer-0 contrast, whether the nulls are drawn in the
same space the probe scores in; (c) claims vs files — every number re-derived from the raw
arrays, not from your prose. Give the reviewer `PREDICTIONS-trackB.md`.

---

## Numbers you can use without loading the model

From `results/m2/acts/*_response_avg.npy`, verified. Reproduce as a smoke test:

Displacement along the unit W−B direction, layer 14: **W +4.894 · P +0.513 · D +0.109 ·
N −0.194 · F −0.505**. Mean activation norm 47.8. `D − N` paired = **+0.3025**, frac(D>N) =
**0.971**. Direction norm ‖mean(W−B)‖: 0.195 at L0, 4.894 at L14, 83.3 at L27.

The registered band where D−N is significant on all four corpus x probe combinations, and
nowhere else, is **layers 12–18**.

---

## Ground rules

1. **Do not train. Do not download further models. Do not rent a GPU.** The two adapters you
   need are already on disk.
2. **Only one process may hold the 15.5 GB model.** `ps aux | grep python` before any load.
3. **Never put a `nohup ... &` launch and a wait loop in the same Bash call** — the tool
   timeout kills the whole process group. Launch detached with `disown`, poll separately.
   Use `python -u`.
4. **No silent fallbacks.** Fail loudly. Assert every count: 239 items, shape
   (239, 28, 3584), 28 layers, hidden 3584.
5. **Do not import from the vendored repos at runtime.** `model-organisms-for-EM-main`
   hardcodes `BASE_DIR = "/workspace/..."`.
6. **The prefix KV cache stays off.** `activations.capture` asserts this.
7. **Condition name in every record id.** A prior bug had a second run silently skip rows
   and re-report the first run's numbers.
8. **Do not edit `readouts.py`, `run_m1.py`, `steering.py`, or `config.py`.** The M1 agent
   owns them and this repo is **not a git repository** — a concurrent edit silently clobbers
   with no conflict marker and no recovery. Put constants in a new
   `icl_em/config_trackb.py`.
9. Under transformers 4.57.6, `Qwen2DecoderLayer.forward` returns a **bare tensor**, not a
   tuple. `activations.py` already handles this; do not reintroduce a blind `output[0]`.
10. `.venv` is at the repo root.

## Standard of evidence

`RESULTS-m2.md` records three corrections found by audit, one of which biased *toward* the
study's own hypothesis. Errors that flatter your hypothesis are the ones to name loudest.
Where a document and a file disagree, say so rather than quietly picking one — the handoff
preceding M2 carried three claims that were false against the files, and finding that cost
real time.

## Deliverables

`PREDICTIONS-trackB.md` (pre-registered, sha256'd before any run) · `icl_em/config_trackb.py` ·
`icl_em/run_trackb.py` (mirror `run_m2.py`: `--stage`, dual manifests, `preflight`,
`modelcheck`, `selftest`) · `results/trackb/` with acts, records, summary ·
`RESULTS-trackB.md` shaped like `RESULTS-m2.md` · a section in `icl_em/README.md`.

**Then stop and report. Do not run.**
