# Data

This repository does not redistribute event logs or model checkpoints.

Public sample corpora are stored in `data/samples/`. The full generated corpora
should stay local:

- `data/pretrain_story.csv`
- `data/finetune_story.csv`

Expected processed CSV files should contain:

- a case id column, for example `case_id` or `case:concept:name`
- an activity column, for example `activity` or `concept:name`
- a timestamp column, for example `timestamp` or `time:timestamp`
- `label` for final outcome prediction, with values `success` or `failure`
- `remaining_time` in days, optional because it can be recomputed from timestamps

The pipeline assumes one CSV file per dataset under `data/processed/`. File names
are matched by dataset aliases such as `BPIC2012A.csv`, `BPIC2020R.csv`, and
`Hospital-billing.csv`.

Recommended source datasets:

- pre-training and instruction tuning: `BPIC2013I`, `BPIC2013O`, `BPIC2019`,
  `BPIC2020D`, `BPIC2020I`, `BPIC2020Pe`, `BPIC2020Pr`, `BPIC2012A`,
  `BPIC2012O`, `Receipt`, `Sepsis`, `Road_Traffic_Fine`, `Service_process`
- downstream evaluation: `BPIC2013C`, `BPIC2017O`, `BPIC2020R`, `BPIC2012W`,
  `Hospital-billing`, `Helpdesk`
