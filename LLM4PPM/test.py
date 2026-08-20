from __future__ import annotations

import sys

from artifact_utils import add_src_to_path, load_yaml_args


def main() -> None:
    add_src_to_path()
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        print("""usage: test.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --model_path MODEL_PATH [options]

Evaluate an LLM4PPM model. Common options:
  --adapter_path ADAPTER_PATH
  --dataset_names DATASET [DATASET ...]
  --test_ratio TEST_RATIO
  --batch_size BATCH_SIZE
""".strip())
        return

    sys.argv = [sys.argv[0], *load_yaml_args(sys.argv[1:])]

    from llm4ppm.evaluate import main as run

    run()


if __name__ == "__main__":
    main()
