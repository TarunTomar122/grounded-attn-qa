# Needle 26M grounded RAG log

## 2026-08-14 — starting-point audit

The available public checkpoint is the function-call post-trained Needle model,
not a verified clean foundation checkpoint. Its untouched free-running output
on five QA probes (correct, wrong, empty, and counterfactual contexts) was always:

```text
<tool_call> []</s>
```

This makes restoration of contextual question answering the purpose of N1.

A frozen 256-question answerable SQuAD2 validation slice produced the same
result across 776 free-running inputs:

| Condition | Rows | EM | Token F1 | EOS | Output changed from correct context |
|---|---:|---:|---:|---:|---:|
| Correct evidence | 256 | 0% | 0% | 100% | — |
| Wrong evidence | 256 | 0% | 0% | 100% | 0% |
| Empty evidence | 256 | 0% | 0% | 100% | 0% |
| Counterfactual evidence | 8 | 0% | 0% | 100% | — |

Every prediction was the same two-token payload, `<tool_call> []`, followed by
EOS. N0 therefore has zero measurable context dependence under raw greedy QA
decoding.

### Checkpoint

| Item | Verified value |
|---|---|
| Hugging Face repository | `Cactus-Compute/needle` |
| Repository revision | `5f89b4307696d669c3df1d38ae057e6e1728b107` |
| Canonical historical implementation | `cactus-compute/needle@a5af1f8282c3aa56d6fbfde77648146114ad31fa` |
| Language-model parameters | 26,233,372 |
| Encoder / decoder layers | 12 / 8 |
| Hidden size | 512 |
| Query / KV heads | 8 / 4 |
| Vocabulary | 8,192 SentencePiece tokens |
| Source / decoder limits | 1,024 / 512 |
| FFN | None |
| Positional encoding | RoPE on encoder and decoder self-attention; none on cross-attention |
| Other details | QK ZCRMSNorm, gated residuals, shared embeddings, tied output |

Artifact SHA-256:

```text
model.safetensors c5f9a3016e4537e492c362da5cb8ba05107d8595bec0d5ea5d8a65801db46531
needle.pkl        40a32e91d1d4197bf15ba559b74f6727c342dc8746918742fc7d8e2c1f18df40
tokenizer.model   0823f5b9133c68a8140addc5d7a425fa9119c4c8cb4a550363b4bffa4ba1c8c7
```

The PyTorch port was checked against the original JAX checkpoint on identical
inputs in BF16. The encoder mean absolute difference was `0.00814`; logits mean
absolute difference was `0.03938`; the first nine ranked token IDs matched.
This audit exposed and fixed one port bug: self-attention K/V must receive the
same normalized state as Q.

### PleIAs/SYNTH RAG

Dataset revision: `0d6813a2966662c39f22f0b9af28a0c1c9f7a437`.

| Metric | Exact value |
|---|---:|
| Parquet shards | 500 |
| Full compressed bytes | 236,335,243,511 |
| All rows | 77,908,583 |
| `exercise == "rag"` rows | 796,582 |
| RAG languages | English only |
| Dataset-reported RAG words | 595,812,155 |
| Remote metadata scan | 153.06 seconds |

RAG evidence is stored in `constraints` as numbered `<source_N>` blocks. The
answer often contains `<ref name="source_N">...</ref>` evidence citations, so
supporting source IDs can be recovered without using `synthetic_reasoning`.
The filtered RAG subset is being extracted directly from remote Parquet; the
236 GB full corpus is not downloaded.

### Compute and storage

The assigned host exposes one RTX 4090, BF16 support, 49,140 MiB VRAM, 20 vCPU,
and a 50 GB persistent volume at `/root/autodl-tmp`. All environments, caches,
datasets, runs, and checkpoints remain on that volume.

The faithful PyTorch port now uses scaled-dot-product attention. At the maximum
1,024-token source and 512-token decoder lengths:

| Mode | Batch | Padded tokens/s | Peak VRAM |
|---|---:|---:|---:|
| Eager BF16 | 12 | 242,347 | 5.30 GB |
| Compiled BF16 | 12 | 446,824 | 3.28 GB |
| Compiled BF16 | 24 | **452,397** | 6.49 GB |

The measured N1 training loop, including CE, z-loss, clipping, and AdamW,
sustained about 405k–412k padded tokens/s and 250k actual tokens/s after compile.

### Full N1 data audit

All 796,582 English RAG rows were extracted from the pinned SYNTH revision into
four Parquet parts totaling 3,474,885,339 bytes. The extraction manifests are:

| Start shard | Rows | Bytes | SHA-256 |
|---:|---:|---:|---|
| 0 | 199,143 | 867,953,059 | `5baa2408e806f5b5d338f2097262cbd145a31aa738ca49187c9acef39f766397` |
| 125 | 199,147 | 869,792,192 | `5c77f28f0456d853900f5255e1af399a83f5c3bd389ba8dac7c1bf0cefc82ffc` |
| 250 | 199,145 | 868,709,848 | `50d940b72afe022191edeeb02a9e67f7702012fdaf3673bc90e6dffbbeb00477` |
| 375 | 199,147 | 868,430,240 | `5c003dd2e6e736b58604bad365de0b73e414582992ea7deb9d538fe555e7bb50` |

A deterministic 2.5% hash sample (`20,126` rows) replaced the preliminary
single-shard audit. Citation parsing succeeded for every sampled row; 99.71%
had citations, 0.035% cited a missing source, and 22.17% appeared unsupported.
After removing citation quote blocks, requiring a 512-token target, and packing
gold evidence plus deterministic distractors into 1,024 source tokens, 27.13%
were usable. This estimates **216,145 N1 rows** in the full corpus.

Median clean target length is 287 tokens. Median full context length is 3,070
tokens, while evidence-first packing has a median of 828 tokens and fits 41.01%
of examples before the other quality gates. N1 trains on clean answer text;
explicit provenance supervision remains reserved for N2-PG. The exact audit is
stored in `artifacts/synth_rag_content_audit_2026-08-14.json`.

### N2 dataset audit

Current pinned Hub revisions and observed sizes:

| Dataset | Revision | Train | Validation | Intended role |
|---|---|---:|---:|---|
| SQuAD2 | `3ffb306f725f7d2ce8394bc1873b24868140c412` | 86,821 answerable + 43,498 unanswerable | 5,928 + 5,945 | N2 spans; N3 refusal |
| CoQA | `0d9e9952f1ef6e5415492d3d84b5873259137e3c` | 108,647 turns, 1,376 unknown | 7,983 turns, 66 unknown | natural answers + rationale spans |
| HotpotQA | `1908d6afbbead072334abe2965f91bd2709910ab` | 90,447 distractor examples | 7,405 | untouched external evaluation |

The current Hugging Face Natural Questions export is marked partial and exposes
only 10,639 training rows, far below the official corpus. It will not be treated
as complete or mixed into N2 until the official simplified release is prepared
and its span semantics are verified.

HotpotQA remains untouched for multi-hop transfer. GaRAGe is an additional
evaluation-only candidate: its public release has 2,366 questions, over 35,000
human-annotated grounding passages, relevance/correctness labels, and
human-written answers. Its CC-BY-NC-4.0 license and direct grounding focus make
it suitable for research evaluation, not inclusion in the training mixture.

The pinned SQuAD2 and CoQA releases have now been converted into fixed
1,024-source / 512-target tensors for matched N2-GEN and N2-PG runs:

| Dataset split | Kept rows | Source tokens | Target tokens | Copyable target tokens |
|---|---:|---:|---:|---:|
| SQuAD2 train | 82,065 | 22,079,860 | 714,585 | 632,520 |
| SQuAD2 validation | 5,690 | 1,604,655 | 49,342 | 43,652 |
| CoQA train | 107,270 | 66,656,188 | 705,293 | 508,117 |
| CoQA validation | 7,917 | 4,725,298 | 49,525 | 35,096 |

SQuAD2 excludes unanswerable rows at N2 and drops 4,994 examples whose answer
boundaries do not align exactly after SentencePiece tokenization. CoQA retains
natural answers and maps only tokens defensibly found inside the annotated
rationale; 25,022 retained rows therefore have no copyable answer token and
remain useful to the generator arm. The exact revisions, drop counts, tensor
hashes, and alignment contract are stored in
`artifacts/needle_n2_manifest_2026-08-14.json`.

### Next measured gates

1. Freeze N0 QA/context-dependence evaluation sets.
2. Audit RAG source counts, citations, token lengths, and 1,024-token fit rate.
3. Benchmark N1 throughput and stable batch size.
4. Train N1 with query + evidence to raw answer CE and a small z-loss.
5. Fork the same N1 checkpoint into matched N2-GEN and N2-PG runs.

### N1 smoke gate: sequence boundaries

The first 394-train / 5-validation-row overfit gate exposed a mismatch hidden
by ordinary teacher-forced token CE: long answers underweighted both the first
answer token and EOS. At step 250, the controlled boundary-weight pilot used
`lr=3e-4`, first-token weight 100, and EOS weight 100:

| Metric | Public N0 | Step 250 |
|---|---:|---:|
| Validation first-token accuracy | 0% | 40% |
| Validation EOS accuracy | 0% | 100% |
| Validation token accuracy | 7.9% | 34.7% |
| Wrong-context CE gap | 0.64 | 1.19 |
| Free-run EOS rate | 60% | 100% |
| Free-run mean tokens | 206.0 | 16.8 |
| Free-run EM / token F1 | 0% / 0% | 0% / 12.4% |

W&B run: `50fnkq71`. The EOS intervention worked, and the model no longer emits
512-token loops. The five-row validation is deliberately treated only as an
overfit diagnostic; full-corpus validation decides whether N1 proceeds.

The subsequent 1,000-step gate isolated the optimizer as the remaining cause.
AdamW at `3e-4` for every parameter stalled at 34.4% train token accuracy.
Needle's original split recipe—Muon at `0.02` for dense projections and AdamW
at `3e-5` for embeddings, norms, and gates—fit the same 394 rows nearly exactly:

| Step | Train token accuracy | Train first token | Train EOS | Validation token F1 |
|---:|---:|---:|---:|---:|
| 250 | 83.5% | 100% | 100% | 32.6% |
| 500 | 99.2% | 100% | 100% | 34.1% |
| 1,000 | 99.96% | 100% | 100% | 30.6% |

The five unseen validation rows overfit after the early peak, as expected from
about 60 passes over a 394-row corpus. The result establishes that the faithful
optimizer can fit free-running answer boundaries and content; full N1 will use
the broad unique corpus and validation-based checkpoint selection. The compiled
Muon loop sustained about 190k actual / 303k padded tokens/s at 6.42 GB allocated
VRAM. W&B run: `7q4tq5b3`.

Raw greedy decoding of the first eight training rows at the step-1,000 Muon
checkpoint produced the exact full gold answer on all eight (`EM=F1=100%`,
`EOS=100%`, mean 209.9 generated tokens). This confirms the overfit gate in
free-running mode, not only under teacher forcing.

A matched 250-step boundary ablation retained first-token weight 100 and reduced
EOS weight from 100 to 20. EOS remained 100% in free generation while validation
token F1 improved from 32.6% to 36.5%, token accuracy from 50.9% to 53.0%, and
the wrong-context CE gap from 2.29 to 2.98. Full N1 therefore uses EOS weight 20.
W&B run: `gvxf1lcm`.

### N1 full-corpus result

The four prepared parts contained 215,870 unique training rows and 2,082
validation rows. The one-pass run consumed 211,574,241 actual source+target
tokens in 8,994 steps. It sustained about 193k actual tokens/s with 6.53 GB peak
allocated VRAM. W&B run: `6qg5u5xy`.

| Metric | Public N0 | N1 step 500 | N1 final |
|---|---:|---:|---:|
| Validation CE | 8.003 | 2.343 | **1.671** |
| Validation token accuracy | 10.7% | 54.6% | **64.1%** |
| First-token accuracy | 0% | 45.2% | **52.9%** |
| EOS accuracy | 0.10% | 98.8% | **99.2%** |
| Wrong-context CE gap | 1.06 | 2.10 | **2.35** |
| Eight-row free-run token F1 | 0% | 29.0% | **33.0%** |

The final checkpoint was also the best full-validation checkpoint. On the frozen
256-question SQuAD2 transfer slice it moved from 0 to 11.57% correct-context
token F1, versus 2.99% with wrong context and 2.41% with empty context. Every
correct-context prediction changed after both wrong- and empty-context swaps.
This establishes context dependence, but not yet strong human QA: EM remained
0%, correct-context EOS was 84.0%, and the eight counterfactual examples reached
only 4.98% F1. N2 therefore begins from a useful evidence-sensitive model while
retaining natural short-answer and counterfactual obedience as explicit gaps.

The selected model-only artifact is 52,490,152 bytes with SHA-256
`a4f3b9dd54613e89e9205cfa19ad85df230f522c2c062c7f05c765639ac1ae14`.
Exact tensor hashes, the complete validation curve, frozen evaluation summary,
and qualitative context-swap pairs are stored in
`artifacts/needle_n1_summary_2026-08-14.json`.

### N2 matched natural-QA result

N2 started both arms from the same N1 checkpoint and trained each for one pass
over 189,335 answerable SQuAD2 and CoQA rows (7,888 steps, batch 24). N2-GEN
continued ordinary vocabulary generation. N2-PG added the existing
pointer-generator and explicit gold source-position loss.

| Frozen SQuAD2 metric | N2-GEN | N2-PG |
|---|---:|---:|
| Correct-context EM | **52.34%** | 49.22% |
| Correct-context token F1 | **65.31%** | 63.47% |
| Wrong-context token F1 | 2.80% | **1.23%** |
| Empty-context token F1 | 8.09% | **0.52%** |
| Counterfactual EM | **87.5%** | 62.5% |
| Correct/wrong output change | 99.61% | **100%** |
| Correct/empty output change | 99.22% | **100%** |

N2-PG finished with 88.13% strict gold source-position accuracy and mean
`p_gen=0.223`, so about 77.7% of its output probability came from copying. The
pointer arm gives measurable provenance and cleaner unsupported-context
behavior, while costing 3.13 percentage points of correct-context EM and 1.85
points of F1. This is a positive grounding tradeoff rather than a blanket
quality win. W&B runs: N2-GEN `pc1x9sjl`; N2-PG `xk8up6zo`.

Exact checkpoint hashes and frozen-evaluation numbers are stored in
`artifacts/needle_n2_summary_2026-08-14.json`.

### N3 refusal curriculum pilots

The first direct N3 run mixed answerable N2 rows with official SQuAD2
unanswerables immediately. It produced a real answerability signal but failed
the product gate: its frozen-context evaluation still answered 52.15% of
unsupported rows at its calibrated threshold. Rather than train it longer, the
follow-up controlled the negative curriculum while preserving the N2-PG
architecture and pointer loss.

| Pilot | Negative mix | Safe-validation answer coverage | Safe false-answer rate | Frozen result |
|---|---|---:|---:|---|
| B1, 500 steps | 30% cross-pair | **98.82%** | 1.99% | 48.05% correct EM, 97.66% refusal recall, 0.76% false refusal |
| B2, 500 steps | official SQuAD2 after B1 | 58.94% | 1.97% | 96.21% false refusal |
| B3, 500 steps | 15% cross-pair + 15% official replay | 59.22% | 1.92% | 93.94% false refusal |

Here “safe” selects the highest-coverage validation threshold with false-answer
rate at most 2%, rather than the threshold that maximizes macro F1. B1 is a
clear success for **cross-document mismatch**: on the frozen context-swaps it
kept useful answer coverage while refusing wrong and empty contexts. It also
copied with 86.57% strict pointer-position accuracy. It did not solve neutral
unsupported questions: the handwritten observatory-password probe was answered
as `hilltop` with answerability probability 0.9648.

B2 then showed that direct adversarial SQuAD2 fine-tuning overwrites this
behavior. B3 replay stopped the complete collapse but still required refusing
nearly all valid frozen answers to satisfy the 2% false-answer bound. This is
evidence against extending the same mixtures, not an architecture failure.
The next measured intervention is an **entity-binding negative stage** between
cross-pair and official SQuAD2 negatives. Exact manifests, run IDs, metrics,
and the decision are stored in
`artifacts/needle_n3_curriculum_pilots_2026-08-15.json`.

### N3 binding and calibration follow-up

The entity- and relation-binding bridges both taught their intended synthetic
control without fixing natural unsupported questions. B4 (entity) achieved
89.22% development pointer-position accuracy; B5 (relation) reached 91.44%.
Both still answered the neutral handwritten observatory-password question,
showing that synthetic binding was not the remaining bottleneck.

B6 added 15% official SQuAD2 unanswerables. Its initial development split used
the public SQuAD2 validation partition, which overlaps the 256-row N0
diagnostic benchmark. That split is therefore invalid for model selection and
is retained only as an audit record.

B7 repaired the protocol: its 275,486 training rows and 14,500 development
rows are deterministic, disjoint partitions of source *train* data. It kept
the B6 mixture and continued from B6 for 500 steps. On its clean development
split, the 2%-false-answer calibration selected threshold 0.665 with 70.65%
answer coverage and 1.95% false answers (W&B `g2jzx4ix`).

That fixed threshold still refused 93.94% of answerable rows on the already
observed N0 SQuAD2 diagnostic. This is a **gate-calibration failure**, not a
reader failure: raw greedy answers reached 41.02% EM on correct contexts while
wrong and empty raw EM were each 0.39%. A post-hoc diagnostic threshold of 0.30
would give 78.41% answer coverage and 0.59% false answers there, but it is not
a valid deployment threshold because it was measured on the diagnostic set.

The next experiment must change the evidence-sufficiency signal or use a
calibration distribution demonstrably matched to deployment negatives; another
longer run of the same standalone answerability head is not warranted. The N0
slice has now been inspected during development and must no longer be presented
as a final untouched result.

### N3 paragraph-matched natural-QA control

To remove context-style shortcuts, `scripts/prepare_needle_n3_matched.py`
formed answerable/unanswerable pairs from the same SQuAD2 *train* paragraph.
Contexts were split deterministically by SHA-256, so no paragraph appears in
both splits. The resulting train split has 8,267 matched paragraphs (16,534
rows) and validation has 419 paragraphs (838 rows); every split is exactly
balanced. The manifest is
`/root/autodl-tmp/datasets/n3-matched/n3-matched-manifest.json` on the remote
volume and records the upstream dataset revision and tensor hashes.

This control gave a clear negative result. A frozen B7 reader plus a linear
question/context/product/difference sidecar only reached AUC 0.7444 and 10.98%
safe answer coverage at 1.91% false answers. Training the existing
question-only answerability head for 500 steps improved matched validation
coverage only from 8.59% to 12.89% (1.91% false answers) and made the observed
N0 diagnostic worse: 95.08% false refusals at its matched-validation threshold.

Replacing the model head with the same explicit interaction features did not
change that conclusion. The first 500-step run used an unsuitable random
initialization that saturated product-feature logits, then recovered to 11.69%
safe coverage. The corrected zero-logit initialization began neutrally and
reached only 10.26% safe coverage after 500 steps. It was stopped at its saved
checkpoint rather than extending a regressing curve. Pointer grounding stayed
healthy throughout (about 90--92% validation source-position accuracy), so the
failure is the evidence-sufficiency decision rather than copy provenance.

The interaction architecture and backward-compatible checkpoint loading are in
commits `2a4793b` and `4e4f032`; tests passed 78/78. W&B runs are
`wlgpj887` (question-only), `oq3ay0oa` (interaction diagnostic), and
`3upq5lae` (zero-init interaction). The next intervention must score a
specific retrieved answer span against the question, rather than classify only
pooled encoder summaries. Repeating this matched-data curriculum with the
current pooled classifier is not justified.

### N3 evidence-span/no-answer pilot

The pooled classifier failure motivated a direct answer-span alternative:
`cd32689` adds a learned no-answer logit and one score per context source
position. Answerable rows supervise the known first gold copy position;
unanswerable rows supervise the no-answer class. This is a stricter objective
than pooled answerability because it must identify provenance before answering.

The matched-data pilot did learn its training loss (validation evidence-start
NLL 3.43 at initialization, 1.96 at step 500, and 1.69 at step 1,000), but it
did not transfer the hard decision. At both saved checkpoints, validation
evidence-start accuracy was 0% and no-answer accuracy was 100%: the head chose
the easy no-answer class for every matched paragraph. Safe answer coverage was
1.67% at step 500 and 3.82% at step 1,000, both far below the pooled-head
control. Pointer source-position accuracy stayed about 90.5--90.9%.

The run was stopped at `/root/autodl-tmp/runs/n3-matched-b4-span-2000/step-001000.pt`.
W&B runs: `je4muoap` (initial 500 steps) and `1dck51h7` (resumed continuation).
This rules out both pooled encoder classification and a linear source-start
head as sufficient natural-QA evidence tests on this checkpoint. The reader
can still be used as a provenance-constrained copier, but a genuinely grounded
RAG decision will require an answer-conditioned verifier or a stronger reader,
rather than more iterations of these heads.

### N3 candidate-verifier result

The next control changed the question from “is this context answerable?” to
“does this specific candidate answer this question in this context?” Candidate
inputs are rendered as a question, proposed answer, and an evidence window
centered on the candidate. This follows the selective-QA framing in
[Chen, Choi, and Durrett (2021)](https://aclanthology.org/2021.findings-emnlp.324/),
which treats answer verification as a premise/hypothesis support decision.

The first data set used one correct answer, one unrelated in-context span, and
one unanswerable question with an in-context span for each matched SQuAD2
paragraph: 24,846 train rows and 1,257 context-disjoint validation rows. A
frozen B7 encoder plus the existing question/context/product/difference linear
readout reached AUC 0.8769 and 33.17% safe coverage at 1.79% false accepts.
That attractive in-distribution number did not transfer: its handwritten probe
accepted the obvious Comet Bay tag but rejected three valid answers and accepted
the unsupported engineer candidate. Fine-tuning the B7 encoder for 2,000 steps
was tracked in W&B run `u15ptheq`; validation AUC reached only 0.8049 and safe
coverage 20.76% at 1.79% false accepts, below the frozen control.

The decisive follow-up eliminated the artificial-span mismatch. Script
`scripts/prepare_needle_n3_reader_candidates.py` created matched SQuAD2 inputs
and generated candidates with the actual frozen N2-PG reader. It used 8,697
paragraphs, 16,560 train questions and 834 context-disjoint validation
questions. Runtime refuses nonliteral candidates by contract; of the remaining
literal candidates, the tensor data contains 16,394 train rows (3,764 supported
and 12,630 unsupported) and 829 validation rows (193 supported and 636
unsupported). Dataset and reader-report SHA-256 hashes are retained in
`/root/autodl-tmp/datasets/n3-reader-candidates/n3-reader-candidates-manifest.json`
on the remote persistent volume.

This realistic test produced the clearest negative result so far. A frozen N2
encoder with the same pooled linear readout achieved only AUC 0.6290 and 4.66%
safe coverage at 1.73% false accepts. End-to-end on the already-observed N0
diagnostic it accepted 6 of 776 candidates: 97.73% false refusal on answerable
conditions and 0% false acceptance on unsupported conditions. It accepted none
of the five handwritten cases. The raw N2 reader itself remains useful
(`49.22%` correct-context EM on N0), so this is a verifier-decision failure,
not evidence that copying disappeared.

We will not run another pooled linear verifier. The current readout pools the
question and the whole context but does not explicitly pool the proposed source
span. The next bounded experiment is a candidate-span-aware cross-attention
verifier: score the question representation against the exact candidate span
representation in its evidence window, using reader-generated candidates and a
context-disjoint calibration split. This is the direct missing component in our
implementation and aligns with answer-verification work that jointly models
support/refute/neutral relations rather than treating a candidate as a global
context label ([Zhang, Vu, and Moschitti, 2021](https://aclanthology.org/2021.acl-long.252/)).

### N3 candidate-span probe

The first implementation of that hypothesis kept N2 frozen and changed only
the readout: it now pools the exact SentencePiece positions of the candidate in
the context evidence window, then compares that jointly encoded span to the
question. This is a fair ablation because the old feature pools the whole
context while the candidate source span is already known from the runtime
literal match.

It improved the reader-generated validation AUC from 0.6290 to **0.6823** and
safe coverage from 4.66% to **7.25%** at 1.57% false accepts. The gain confirms
that candidate provenance is useful. It still fails the product criterion:
at the context-disjoint calibrated threshold it accepted 0 of 776 N0 rows and
0 of 5 handwritten rows. This rules out both global and candidate-span pooled
*linear* readouts on frozen N2 representations. The next bounded experiment is
a nonlinear candidate-span support head; if it cannot provide materially higher
safe coverage on the same fixed split, N2's frozen representations are not
sufficient for grounded refusal and we will move to an explicitly trained NLI
cross-encoder with support/refute/neutral supervision rather than keep adding
calibration heads.

### N3 nonlinear candidate-span control

The final cheap control added a 0.52M-parameter GELU MLP above the same frozen
question/candidate-span features. It did not improve them: AUC was 0.6776,
below the linear span probe's 0.6823, with identical 7.25% safe coverage at
1.89% false accepts. Its end-to-end N0 result again accepted 0 of 776 rows.

This closes the frozen-readout branch. Three feature choices (global context,
candidate source span, and a nonlinear span MLP) all refuse almost every valid
reader output when calibrated safely. The N2 reader remains separately intact,
so the next work is a dedicated, trainable NLI-style cross-encoder verifier
over question, candidate, and local evidence. It will use the already-created
reader-generated candidates, explicit support/refute/neutral labels, and a
context-disjoint calibration split; it will not update N2's copying weights.
