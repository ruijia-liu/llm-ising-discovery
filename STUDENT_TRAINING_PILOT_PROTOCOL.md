# B/C student training: local feasibility pilot

This is a first language-model training run, following the audited curriculum.
It tests whether the student pipeline works and provides an initial paired
comparison. It is not the proposed 300-world experiment and does not establish
discovery of unseen structures.

## Fixed scope

- Base model: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
  public revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Groups B and C, seeds 101, 102, 103. The auxiliary A arm is deferred.
- All 24 training worlds in curriculum v1, using their sigma=2 episodes. The
  other noise levels remain available but are outside this small run. No world
  is selected according to whether the teacher succeeded.
- All 12 parameter-validation worlds at sigma=2, with 32 recorded measurements
  per world. The student receives fixed evidence and cannot query further.
- Reserved combination and four-body test worlds remain unopened.
- No RL, reward model, new tools, forced confirmation phase, or removal of
  false discoveries. The teacher's actual erroneous intermediate records remain.

## Representation and labels

The audited original conversations remain in `data/research_curriculum_v1/`.
`prepare_student_training.py` makes a lossless compact version: each complete
256-spin array becomes 64 hexadecimal characters; H0 is implicit in expressions;
repeated explanation text references its previous value. Floating-point values
are not rounded. Every conversation is expanded and checked against its original.
This encoding may itself be hard for a small language model; no claim is made
that it is the best representation for discovering interactions.

B and C have identical measured configurations, positions, results, system
instructions, and final correct expressions. C additionally sees the recorded
hypotheses, predictions, selection statistics, and updates. These extra fields
are the teaching treatment. Truth labels were appended by the evaluator only
after the teacher finished. Training never opens the private world files.

All conversations use the pinned tokenizer's full chat template, with no
truncation. Loss applies to assistant message contents and end markers only;
system/user observations and role headers are masked. This includes the teacher's
experimental actions in B, and those actions plus process annotations in C.

## Training budget fixed before validation

Each C arm sees the 24 episodes once, two complete episodes per optimizer update,
for 12 updates and 470,300 nonpadding input tokens. B repeats only its same 24
episodes to match this input-token budget, with the same 12 optimizer updates.
The schedules are generated and saved before any student validation:

| Seed | B input tokens | C input tokens | Difference |
|---|---:|---:|---:|
| 101 | 470,848 | 470,300 | 0.117% |
| 102 | 470,868 | 470,300 | 0.121% |
| 103 | 470,832 | 470,300 | 0.113% |

Input-token matching does not equalize assistant target tokens, FLOPs, or elapsed
time: C has more assistant annotation and longer attention contexts. The program
reports actual target tokens, elapsed time, sequence presentations, GPU memory,
and updates. The inference budget is identical. These limits must accompany any
comparison; this pilot cannot isolate all effects of compute or loss allocation.

Both arms use LoRA rank 8, alpha 16, dropout 0.05 on q/k/v/o projections,
bfloat16 base weights/activations and float32 LoRA parameters (PEFT's default),
AdamW with learning rate 0.0002 and no weight decay, gradient clipping
at norm 1, and constant learning rate. Each seed/arm starts from the same base;
neither continues from another arm's weights. Gradients accumulate over complete
episodes, normalized by the number of supervised targets in that update.

Gradient checkpointing, RAM storage of checkpoint inputs, tokenwise MLP chunks
(1,024 tokens), and chunked output loss reduce memory without shortening the
attention context. MLP operations act independently on each token. CUDA allocation is capped
at 80% of the 6 GiB card to leave room for desktop applications and fail explicitly
instead of silently paging excessive GPU allocations into shared memory.
The loss and gradients are tested against ordinary full-logit
cross-entropy on a small model. A longest-episode preflight update is discarded.

## Validation and interpretation

The untrained base and every trained arm receive exactly the same B-format
validation evidence, with the final answer removed. They are asked for a JSON
expression, with greedy decoding and at most 256 new tokens. Training-specific
C process annotations are absent from this fixed-evidence validation prompt.

The evaluator checks the expression on 256 fresh uniform configurations and
256 biased configurations per world, shared across all arms. Those configurations
and their values are never returned to the model. Tolerances remain those frozen
in the earlier independent noise calibration. A full recovery requires matching
support, both prediction-error thresholds, and the coefficient-error threshold.
Translation-equivalent terms and duplicates use the existing physics canonicalizer.

Two non-LLM fixed-evidence references use the identical 32 observations and no
additional queries: always predict H0, and the existing sparse expression search
over all <=2-term supports in the public order-4 grammar. The search fit receives
only queries, observations, and noise strength. These references were run before
any trained student completed validation, with the existing fixed scoring rule.
Their query budget is matched, but computational cost is not matched to LLM tokens.

Malformed, truncated, or out-of-grammar answers count as failures and are saved
verbatim. Report executable-answer rate separately from scientific recovery.
On null worlds, report false discoveries and invalid outputs separately: an
invalid answer is a failure, not evidence of a low false-discovery rate.

Twelve validation worlds, only two null worlds, and three training seeds do not
support a strong statistical conclusion. Results concern fixed-evidence parameter
inference only. There is no free-exploration result, held-out-structure result,
or claim about learning an internal research method. No hyperparameter search
is part of this run; failures guide later, separately identified development.

## Development record

An initial formatting precheck used a pseudocode JSON example containing literal
ellipses. The untrained model copied them into its 12 invalid validation answers.
The run was stopped after three B/101 updates, before any trained-arm evaluation.
Those raw answers and partial training logs remain in
`results/student_training_v1_prompt_precheck/`. The prompt was corrected to
describe the schema without malformed example JSON; a training-world probe still
showed poor base-model format compliance. The actual comparison restarts every
arm from the base model. This development is disclosed; validation is not claimed
to be an untouched final test. No scoring threshold or hidden structural test
was used to make the correction.

The first C/101 attempt later ran out of CUDA memory during its second update,
despite the longest-episode preflight passing. Its first completed step is kept
under `results/student_training_v1/interrupted_attempts/`. The implementation now
collects checkpoint reference cycles and clears unused CUDA blocks after each
complete episode. The completed B/101 result is retained; the incomplete C arm
restarts from the original base weights and same recorded schedule. The original
executed source and an implementation amendment are saved with the results.
This storage change does not shorten context, reduce the curriculum, or alter
loss normalization, optimizer settings, inference, or scientific scoring.

The cache-clearing retry completed ten updates but then exceeded the original
72% process allocation cap while physical VRAM was still available. That partial
attempt is also preserved. The cap was raised to 80% (4.8 GiB), and each completed
optimizer update now atomically saves adapter weights, optimizer state, and
Python/NumPy/PyTorch RNG states. A unit test confirms that a resumed dropout
training step matches uninterrupted training exactly on a small CPU model.
Attempts without a saved checkpoint restart from base; subsequent interruptions
can continue from the last complete update. Partial arms are never scored as
completed experiments. Memory-management amendments and original sources are
kept beside the results; the scientific training settings remain unchanged.

## User-requested pause and frozen-model diagnostic

After B/101 and C/101 completed, the user requested a stop at the next checkpoint,
seen-training-world inference using existing models, and an audit of C's training
format, final-answer supervision and validation prompts. C/102 is paused after
update 5/12. Its checkpoint, optimizer and RNG states are preserved; no later
arms have begun. This is a paused, incomplete pilot rather than a completed
three-seed comparison.

`diagnose_student_pilot.py` uses only completed seed-101 adapters, in evaluation
mode with gradients disabled. Six seen training worlds are selected by fixed
family order; prompt contrasts use the first field, three-body and null world
from training and parameter validation. The diagnostic adds no laboratory queries
and opens no reserved structural tests. The original validation scores and frozen
thresholds are retained. Native C history contains teacher reasoning and is
explicitly excluded from fair fixed-evidence comparisons. A constant JSON prefix
uses 8 tokens out of the same 256-token completion budget and provides no mechanism.

See `results/student_diagnostics_v1/plan.json` for the 48 inference calls and
protected-file hashes, and `REPORT_ZH.md` there for the results, supervision audit
and proposed changes. No revised training has been executed. Any changed teaching
format or loss allocation belongs in a separately identified follow-up, rather
than changing settings midway through this archived comparison.

## Reproduction

Requires Python 3.12 and a CUDA GPU supporting bfloat16. The executed machine
has an RTX 3060 Laptop GPU with 6 GiB VRAM. Install PyTorch for the machine first;
the executed version is `torch==2.9.1+cu126` from the official PyTorch wheel index.

```powershell
python -m venv .venv-discovery
& .\.venv-discovery\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
& .\.venv-discovery\Scripts\python.exe -m pip install -r requirements-discovery-training.txt
& .\.venv-discovery\Scripts\python.exe setup_discovery_model.py
& .\.venv-discovery\Scripts\python.exe prepare_student_training.py
& .\.venv-discovery\Scripts\python.exe -m unittest -v test_student_training
& .\.venv-discovery\Scripts\python.exe train_student_pilot.py preflight
& .\.venv-discovery\Scripts\python.exe train_student_pilot.py run
```

Downloads use public model files with a pinned revision and published weight
checksum. Training/inference are local and offline. To repeat an experiment use
a new `--output` directory; scripts refuse to overwrite an existing run.
