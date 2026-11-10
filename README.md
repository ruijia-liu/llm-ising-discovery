# LLM Ising Structure Discovery

A synthetic physics laboratory for studying whether language models can learn to propose and test interaction mechanisms from experimental teaching records.

The project combines a generalized Ising simulator, a sparse-search teacher, paired teaching curricula, Qwen student training, and evaluation of mechanism recovery under a measurement budget. **This is a research code release. The formal A/B/C student comparison is incomplete; the repository does not establish a benefit from process teaching.**

[中文说明](README_ZH.md) · [Code and experiment guide](PROJECT_GUIDE.md) · [Release scope](RELEASE_SCOPE.md)

## Research question

Can a student learn more from a teacher's hypotheses, predictions and revisions than from the same experiments and final answers alone?

| Group | Teaching information |
| --- | --- |
| A | Initial measurements and final supervised answer |
| B | A plus the actual experiments and observations |
| C | B plus hypotheses, predictions made before observation, selection rationale and updates |

B and C share experiments, observations and final labels. The teacher is an instrumented numerical search algorithm. Labels are attached by the evaluator after the teacher finishes; incorrect teacher estimates remain in the record. LLMs are the students in the training and inference pipelines.

## What the laboratory measures

- A periodic 16 × 16 spin lattice with a known nearest-neighbour baseline and up to two additional interaction terms.
- Hidden terms drawn from a bounded grammar, with interactions of up to four spins.
- Eight initial measurements and a maximum of 32 single-spin-flip energy measurements per episode.
- Independent Gaussian measurement noise, paired random/active search, structural recovery, prediction error and budget accounting.
- A student interface that fits coefficients only for structures proposed by the student, using already revealed observations.

The environment uses prepared configurations rather than equilibrium samples. Discovering a term in this finite grammar is a narrower task than open-ended discovery of physical laws.

## Quick start: CPU only

Run commands from the repository root with Python 3.12:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-discovery-numeric-lock.txt
python public_checks.py
python demo_teaching.py --output output/teaching_demo
```

The demo creates one synthetic episode, audits all 24 decisions and numerical predictions, and exports paired A/B/C views and chat messages. `inference_request.json` contains a label-free prompt for a student. No model download, GPU, API key or paid service is required. Output directories must be new so previous examples are preserved.

Run a small numerical search comparison:

```sh
python run_structure_discovery.py --worlds-per-family 1 --eval-count 32 --output results/search_demo
```

This compares numerical random and active search across six world families. It produces metrics, scores and trajectories; it is not an LLM result.

## Student models and training

The source includes local Qwen model setup, tokenizer/loss-mask checks, LoRA training, inference workers, resumable queues and paired evaluation. The original student pipelines use Qwen2.5-0.5B-Instruct; separate exploratory code covers Qwen3-0.6B and a larger robustness model.

GPU training requires a compatible CUDA-enabled PyTorch installation and the packages in `requirements-discovery-training.txt`. The inference worker currently uses CUDA and bfloat16. The recorded Windows/Python 3.12 environment is documented by `requirements-discovery-model-repro-lock.txt`; it is not a universal installation lock.

Model weights, generated corpora, local checkpoints and historical experiment outputs are excluded. Some historical scripts verify exact source hashes and require those original artifacts. Follow [the experiment guide](PROJECT_GUIDE.md) before running training or reproduction scripts; the CPU quick start is the supported fresh-checkout entry point.

## Evidence and limitations

The numerical environment and causal teaching audits can be exercised from source. Exploratory student studies also produced negative results: input preprocessing, interface practice and a parameter-diversity screen did not demonstrate improved mechanism recovery. These development observations do not establish a general conclusion about teaching or augmentation.

Formal heldout student results and the complete three-seed comparison remain pending. The code includes planned evaluation and statistical analyses; their presence does not mean those experiments have been completed. The Python information boundary prevents accidental leakage through public replies, but is not a security sandbox against code that deliberately reads local files.

## Repository map

| Location | Role |
| --- | --- |
| `discovery/` | Physics, search, teaching views, student interfaces and scoring |
| `demo_teaching.py` | Standalone audited teaching-data example |
| `run_structure_discovery*.py` | Numerical baseline and noise study |
| `build_*curriculum.py` | Curriculum generation and audit |
| `prepare_*.py`, `train_*.py` | Tokenization, training inputs and student training |
| `evaluate_*.py`, `analyze_*.py`, `report_*.py` | Evaluation, statistics and diagnostics |
| `test_*.py`, `public_checks.py` | Research checks and the self-contained release suite |
| `*_PROTOCOL.md` | Historical experimental designs retained at their original paths |

Experiment entry points stay at the root because source manifests and scripts reference their relative filenames. See [release scope](RELEASE_SCOPE.md) for dependencies, exclusions and model licensing boundaries.
