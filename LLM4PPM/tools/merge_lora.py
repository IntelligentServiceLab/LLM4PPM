from __future__ import annotations

import os
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from peft import PeftModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base causal LM.")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--merged_out", type=str, required=True)

    parser.add_argument("--device_map", type=str, default="auto", help='"auto" or "cpu" etc.')
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]

    adapter_dir = Path(args.adapter_dir)
    merged_out = Path(args.merged_out)
    merged_out.mkdir(parents=True, exist_ok=True)

    tokenizer = LlamaTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    vocab_size = len(tokenizer)
    print(f"[Tokenizer] vocab_size={vocab_size}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.resize_token_embeddings(vocab_size)
    model.config.vocab_size = vocab_size

    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)

    model = model.merge_and_unload()

    emb_shape = tuple(model.get_input_embeddings().weight.shape)
    print(f"[Check] embed_tokens={emb_shape}")
    if emb_shape[0] != vocab_size:
        raise RuntimeError(
            f"Export failed: embed_tokens rows ({emb_shape[0]}) != tokenizer vocab_size ({vocab_size}). "
            "Your adapter may not include modules_to_save for embeddings, or resizing failed."
        )

    model.save_pretrained(str(merged_out), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_out))
    print(f"[Save] merged full model -> {merged_out}")


if __name__ == "__main__":
    main()
