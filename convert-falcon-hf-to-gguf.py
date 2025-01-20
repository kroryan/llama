#!/usr/bin/env python3
# HF falcon--> gguf conversion

from __future__ import annotations

import argparse
import contextlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from transformers import AutoTokenizer  # type: ignore[import]

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py' / 'gguf'))
import gguf


def count_model_parts(dir_model: Path, prefix: str) -> int:
    num_parts = 0
    for filename in os.listdir(dir_model):
        if filename.startswith(prefix):
            num_parts += 1

    if num_parts > 0:
        print("gguf: found " + str(num_parts) + " model parts")
    return num_parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Falcon model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input",
    )
    parser.add_argument(
        "model", type=Path,
        help="directory containing model file, or model file itself (*.bin)",
    )
    parser.add_argument(
        "ftype", type=int, choices=[0, 1], default=1, nargs='?',
        help="output format - use 0 for float32, 1 for float16",
    )
    return parser.parse_args()


def should_skip_tensor(name: str) -> bool:
    """Determine if a tensor should be skipped during processing."""
    skip_patterns = [
        'absmax',
        'quant_map',
        'scales',
        'zeros',
        'scaled_masked_score'
    ]
    return any(pattern in name for pattern in skip_patterns)


def process_tensor(data: torch.Tensor, name: str, n_head: int, n_head_kv: int, head_dim: int) -> torch.Tensor:
    """Process a tensor, handling special cases and conversions."""
    # Convert data type if needed
    if data.dtype not in [torch.float16, torch.float32]:
        data = data.to(torch.float32)

    # Handle QKV transformation
    if "query_key_value" in name:
        try:
            qkv = data.view(n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)
            q = qkv[:, :-2].reshape(n_head * head_dim, head_dim * n_head)
            k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
            v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)
            data = torch.cat((q, k, v)).reshape_as(data)
        except Exception as e:
            print(f"Warning: QKV transformation failed for {name}: {str(e)}")
            # Return original tensor if transformation fails
            pass

    return data.squeeze()


def main() -> None:
    args = parse_args()

    dir_model = args.model
    ftype = args.ftype
    if not dir_model.is_dir():
        print(f'Error: {args.model} is not a directory', file=sys.stderr)
        sys.exit(1)

    # possible tensor data types
    #   ftype == 0 -> float32
    #   ftype == 1 -> float16
    ftype_str = ["f32", "f16"]

    if args.outfile is not None:
        fname_out = args.outfile
    else:
        # output in the same directory as the model by default
        fname_out = dir_model / f'ggml-model-{ftype_str[ftype]}.gguf'

    print("gguf: loading model " + dir_model.name)

    try:
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            hparams = json.load(f)
    except Exception as e:
        print(f"Error loading config.json: {str(e)}")
        sys.exit(1)

    if hparams["architectures"][0] not in ("RWForCausalLM", "FalconForCausalLM"):
        print("Model architecture not supported: " + hparams["architectures"][0])
        sys.exit(1)

    # get number of model parts
    num_parts = count_model_parts(dir_model, "model-00")
    if num_parts:
        is_safetensors = True
        from safetensors import safe_open
    else:
        is_safetensors = False
        num_parts = count_model_parts(dir_model, "pytorch_model-")

    ARCH = gguf.MODEL_ARCH.FALCON
    gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH])

    print("gguf: get model metadata")

    # Extract model parameters with fallbacks
    block_count = hparams.get("num_hidden_layers")
    if block_count is None:
        block_count = hparams["n_layer"]  # old name

    n_head = hparams.get("num_attention_heads")
    if n_head is None:
        n_head = hparams["n_head"]  # old name

    n_head_kv = hparams.get("num_kv_heads")
    if n_head_kv is None:
        n_head_kv = hparams.get("n_head_kv", 1)  # old name

    # Add model metadata
    gguf_writer.add_name("Falcon")
    gguf_writer.add_context_length(2048)  # not in config.json
    gguf_writer.add_tensor_data_layout("jploski")  # qkv tensor transform
    gguf_writer.add_embedding_length(hparams["hidden_size"])
    gguf_writer.add_feed_forward_length(4 * hparams["hidden_size"])
    gguf_writer.add_block_count(block_count)
    gguf_writer.add_head_count(n_head)
    gguf_writer.add_head_count_kv(n_head_kv)
    gguf_writer.add_layer_norm_eps(hparams["layer_norm_epsilon"])
    gguf_writer.add_file_type(ftype)

    # TOKENIZATION
    print("gguf: get tokenizer metadata")

    tokens: list[bytearray] = []
    scores: list[float] = []
    toktypes: list[int] = []

    # gpt2 tokenizer
    gguf_writer.add_tokenizer_model("gpt2")

    print("gguf: get gpt2 tokenizer vocab")

    try:
        tokenizer = AutoTokenizer.from_pretrained(dir_model)
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        sys.exit(1)

    vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))
    try:
        assert max(tokenizer.vocab.values()) < vocab_size
    except AssertionError:
        print("Warning: Vocabulary size mismatch")
        vocab_size = max(tokenizer.vocab.values()) + 1

    reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}

    for i in range(vocab_size):
        if i in reverse_vocab:
            tokens.append(reverse_vocab[i])
            scores.append(0.0)  # dummy
            toktypes.append(gguf.TokenType.NORMAL)
        else:
            print(f"Warning: Missing token for id {i}")
            tokens.append(bytearray())
            scores.append(0.0)
            toktypes.append(gguf.TokenType.NORMAL)

    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(dir_model, load_merges=True, n_vocab=len(tokens))
    special_vocab.add_to_gguf(gguf_writer)

    # TENSORS
    tensor_map = gguf.get_tensor_name_map(ARCH, block_count)
    head_dim = hparams["hidden_size"] // n_head

    print("gguf: get tensor metadata")

    if num_parts == 0:
        part_names = iter(("pytorch_model.bin",))
    elif is_safetensors:
        part_names = (
            f"model-{n:05}-of-{num_parts:05}.safetensors" for n in range(1, num_parts + 1)
        )
    else:
        part_names = (
            f"pytorch_model-{n:05}-of-{num_parts:05}.bin" for n in range(1, num_parts + 1)
        )

    for part_name in part_names:
        if args.vocab_only:
            break
        print("gguf: loading model part '" + part_name + "'")
        try:
            if is_safetensors:
                ctx = safe_open(dir_model / part_name, framework="pt", device="cpu")
            else:
                ctx = contextlib.nullcontext(torch.load(dir_model / part_name, map_location="cpu", weights_only=True))

            with ctx as model_part:
                for name in model_part.keys():
                    if should_skip_tensor(name):
                        print(f"Skipping tensor: {name}")
                        continue

                    try:
                        data = model_part.get_tensor(name) if is_safetensors else model_part[name]
                        data = process_tensor(data, name, n_head, n_head_kv, head_dim)
                        data = data.numpy()

                        new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
                        if new_name is None:
                            print(f"Warning: Cannot map tensor: {name}, attempting to use original name")
                            new_name = name

                        gguf_writer.add_tensor(new_name, data)
                    except Exception as e:
                        print(f"Warning: Error processing tensor {name}: {str(e)}")
                        continue

        except Exception as e:
            print(f"Error processing part {part_name}: {str(e)}")
            sys.exit(1)

    try:
        print("gguf: write header")
        gguf_writer.write_header_to_file()
        print("gguf: write metadata")
        gguf_writer.write_kv_data_to_file()
        if not args.vocab_only:
            print("gguf: write tensors")
            gguf_writer.write_tensors_to_file()

        gguf_writer.close()
        print(f"gguf: model successfully exported to '{fname_out}'")
        print("")
    except Exception as e:
        print(f"Error writing GGUF file: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
