# LLM4PPM

Code artifact for **LLM4PPM: Learning Transferable Semantics for Cross-Domain Predictive Process Monitoring**.

LLM4PPM converts heterogeneous event logs into semantic stories, performs cross-domain continuous pre-training, and then applies multi-task instruction tuning for next activity prediction, remaining time prediction, and final outcome prediction.

## Repository Layout

```text
.
├── prepare_data.py              # data-preparation command dispatcher
├── pretrain.py                  # continuous pre-training entrypoint
├── finetune.py                  # multi-task instruction-tuning entrypoint
├── test.py                      # evaluation entrypoint
├── src/llm4ppm/
│   ├── story.py                 # semantic-story construction and pretraining corpus generation
│   ├── instruction_data.py      # multi-task instruction data generation
│   ├── pretrain.py              # continuous pre-training with LoRA
│   ├── sft.py                   # response-only instruction tuning
│   └── evaluate.py              # constrained decoding and metrics
├── tools/
│   ├── canonicalize_activities.py
│   ├── label_outcomes.py
│   ├── add_remaining_time.py
│   └── merge_lora.py
├── configs/
│   └── evaluate.yaml             # optional evaluation config
└── data/
    └── samples/                 # small public samples, not the full corpora
```

## Setup

```bash
conda create -n llm4ppm python=3.10 -y
conda activate llm4ppm
pip install -r requirements.txt
pip install -e .
```

The training scripts use Unsloth, PEFT, bitsandbytes, and Hugging Face Transformers. A CUDA GPU is strongly recommended.

## Data Preparation

The repository does not redistribute full event logs or full training corpora. Put raw logs under `data/raw/`, then prepare processed CSV files:

```bash
python prepare_data.py canonicalize \
  --raw-root data/raw \
  --output-root data/interim/mapped

python prepare_data.py label-outcomes \
  --input-dir data/interim/mapped \
  --output-dir data/interim/labeled \
  --positive-label success \
  --negative-label failure

python prepare_data.py remaining-time \
  --input-dir data/interim/labeled \
  --output-dir data/processed \
  --unit days \
  --round-ndigits 0
```

Processed CSV files should include case id, activity, timestamp, `label`, and `remaining_time` columns. See `data/README.md` for the expected schema.

Small public samples are included for artifact inspection:

- `data/samples/pretrain_story_sample.csv`
- `data/samples/finetune_story_sample.csv`

The local full corpora, if present, are expected at `data/pretrain_story.csv` and `data/finetune_story.csv`; these files are ignored by Git.

## Source Event Logs

The paper-style split uses the following public event-log sources. Download the raw logs from the original providers, then use `prepare_data.py` and the scripts in `tools/` to standardize names, labels, and remaining time.

| Dataset alias in code | Source |
| --- | --- |
| `BPIC2012A`, `BPIC2012O`, `BPIC2012W` | [BPI Challenge 2012, 4TU](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204) |
| `BPIC2013I`, `BPIC2013O`, `BPIC2013C` | [BPI Challenge 2013 collection, 4TU](https://data.4tu.nl/collections/bpi_challenge_2013/5065448/1) |
| `BPIC2017O` | [BPI Challenge 2017, 4TU](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884) |
| `BPIC2019` | [BPI Challenge 2019, 4TU](https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853) |
| `BPIC2020D`, `BPIC2020I`, `BPIC2020Pe`, `BPIC2020Pr`, `BPIC2020R` | [BPI Challenge 2020 collection, 4TU](https://data.4tu.nl/collections/BPI_Challenge_2020/5065541) |
| `Receipt` | [Receipt phase event log, 4TU](https://data.4tu.nl/articles/dataset/Receipt_phase_of_an_environmental_permit_application_process_WABO_CoSeLoG_project/12709127) |
| `Sepsis` | [Sepsis Cases - Event Log, 4TU](https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639) |
| `Road_Traffic_Fine` | [Road Traffic Fine Management Process, 4TU](https://data.4tu.nl/articles/dataset/Road_Traffic_Fine_Management_Process/12683249) |
| `Hospital-billing` | [Hospital Billing event log, 4TU](https://data.4tu.nl/articles/dataset/Hospital_Billing_-_Event_Log/12705113) |
| `Helpdesk` | [Help desk log of an Italian Company, 4TU](https://data.4tu.nl/datasets/94ee26c8-78f6-4387-b32b-f028f2103a2c) |
| `Service_process` | [Customer Service - Device Repair Process, Zenodo](https://zenodo.org/records/3928487) |

## Build Training Data

Continuous pre-training corpus:

```bash
python prepare_data.py pretrain-data \
  --input_dir data/processed \
  --output_path artifacts/pretrain/pretrain.csv
```

Multi-task instruction-tuning data:

```bash
python prepare_data.py sft-data \
  --input_dir data/processed \
  --output_dir artifacts/sft \
  --output_format csv \
  --validation_ratio 0.02
```

## Train

Continuous pre-training:

```bash
python pretrain.py \
  --model_path /path/to/Llama-2-7b \
  --dataset_path data/pretrain_story.csv
```

The main hyperparameters are defined directly in `pretrain.py` under `DEFAULTS`. You can edit them there for a released artifact, or override any value from the command line:

```bash
python pretrain.py \
  --model_path /path/to/Llama-2-7b \
  --dataset_path data/samples/pretrain_story_sample.csv \
  --output_path outputs/pretrain_adapter_sample \
  --max_train_samples 16 \
  --max_steps 1
```

Instruction tuning:

```bash
python finetune.py \
  --model_path outputs/pretrain_adapter/merged \
  --dataset_path data/finetune_story.csv
```

The main hyperparameters are defined directly in `finetune.py` under `DEFAULTS`. A small smoke run can use the public sample:

```bash
python finetune.py \
  --model_path outputs/pretrain_adapter/merged \
  --dataset_path data/samples/finetune_story_sample.csv \
  --output_path outputs/sft_adapter_sample \
  --max_train_samples 16 \
  --max_steps 1
```

## Evaluate

```bash
python test.py \
  --input_dir data/processed \
  --output_dir outputs/eval \
  --model_path outputs/sft_adapter_merged \
  --test_ratio 0.2 \
  --batch_size 8
```

or equivalently:

```bash
python test.py --config configs/evaluate.yaml
```

The evaluator writes `case_split_summary.csv`, `metrics_by_dataset_task.csv`, `metrics_by_dataset_task_stage.csv`, `metrics_dataset_macro.csv`, and optional row-level predictions.

## Notes

- Default source datasets follow the paper setting in `llm4ppm.story.TRAIN_DATASETS`.
- Default target datasets follow `llm4ppm.story.TEST_DATASETS`.
- Outcome labels are normalized to `success` and `failure`.
- Activity canonicalization is deterministic and one-to-one.
- Baseline model implementations are not included in this artifact; the released code focuses on reproducing the LLM4PPM pipeline.
