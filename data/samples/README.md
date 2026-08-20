# Sample Data

This directory contains small public samples derived from the local training
corpora:

- `pretrain_story_sample.csv`: 100 semantic stories for continuous pre-training.
- `finetune_story_sample.csv`: 135 instruction-tuning examples, with 45 examples
  each for next activity prediction, remaining time prediction, and final outcome
  prediction.

The full training corpora are intentionally excluded from the public artifact.
Use the original event-log sources listed in the main README and regenerate the
full corpora with `prepare_data.py`.
