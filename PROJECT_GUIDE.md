# Code and experiment guide

Run all entry points from the repository root. The release preserves original relative paths so imports and recorded source manifests remain understandable.

## Fresh-checkout entry points

| Command | Requirements and output |
| --- | --- |
| `python public_checks.py` | NumPy only; synthetic fixtures for physics, causality, paired data and statistics |
| `python demo_teaching.py` | NumPy only; one A/B/C teaching episode in `output/teaching_demo/` |
| `python run_structure_discovery.py --worlds-per-family 1 --eval-count 32 --output results/search_demo` | NumPy only; numerical random/active comparison |
| `python run_structure_discovery_noise.py calibrate` | NumPy only; creates calibration for the noise study |
| `python build_research_curriculum.py` | Requires the generated default calibration; builds the initial teaching corpus |

The full default curriculum and numerical studies take longer than the small demos. Generated data include private evaluator labels; inference should receive only the public evidence exported by the student interface. The demo uses new synthetic data and does not inspect a formal heldout set.

## Student pipeline

`setup_discovery_model.py` downloads a model snapshot and records its revision and checksums. On a fresh download it resolves the current model revision; an existing manifest retains its revision. This is not a guarantee of reproducing an earlier study snapshot.

`prepare_student_training.py` converts the initial corpus into model-specific inputs, verifies lossless message compaction and masks non-assistant tokens. `train_student_pilot.py` contains the early LoRA pilot. These require the training dependencies, a local model, generated teaching data and compatible CUDA hardware.

The later `prepare_mainline_training.py`, `train_mainline_student.py`, `run_mainline_queue.py` and `evaluate_mainline.py` encode the formal A/B/C study. Consult [the study protocol](MAINLINE_STUDY_PROTOCOL.md) and [the public student interface](ACTIVE_STUDENT_PROTOCOL.md). Their historical manifests, fixed hashes and checkpoints are not bundled, and these scripts should not be treated as turnkey commands on a fresh checkout.

`student_inference_worker.py` reads one JSON request per line from standard input. Each request has `messages` and `max_new_tokens`; each reply contains generated text and token/time accounting. It loads a local model and optional adapter, with no dataset or oracle imports. It currently requires CUDA/bfloat16. The demo's `inference_request.json` illustrates the request object; serialize it as a single line before sending it to the worker.

## Historical development and analysis

Input-format, interface-alignment, parameter-diversity and model-size scripts preserve exploratory work. Several intentionally require the original control run, queue state or checkpoint before executing. Replacing those guards or hashes solely to make a release checkout run would change the experiment.

`analyze_mainline_primary.py`, `analyze_prediction_ablation.py` and `analyze_uncertainty_supplement.py` implement distinct analyses. Synthetic checks demonstrate code behavior, not observed teaching effects. Protocols describe planned or historical designs; current release status is stated in the README.

## Tests and reproduction

Use `public_checks.py` for the suite supported without external artifacts. The broader `test_*.py` collection also contains integration checks tied to historical corpora, model tokenizers or saved runs. A bare `python -m unittest discover` therefore requires more than the minimal release environment.

The `reproduce_*.py` scripts compare regenerated material against original research outputs. Those comparison targets are intentionally excluded. This release is sufficient to run fresh synthetic examples and inspect the experimental implementation, but does not ship a complete archival reproduction bundle.
