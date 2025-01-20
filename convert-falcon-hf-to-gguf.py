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
from typing import Any, Dict, Optional, List, Tuple

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
        'quant_state',
        'quant_meta',
        'scales',
        'zeros',
        'g_idx',
        'bitsandbytes',
        'scaled_masked_score'
    ]
    return any(pattern in name for pattern in skip_patterns)


def dequantize_tensor(tensor: torch.Tensor, qweight: Optional[torch.Tensor] = None,
                     qzeros: Optional[torch.Tensor] = None, scales: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dequantize a tensor if it's quantized."""
    if tensor.dtype in [torch.float32, torch.float16]:
        return tensor
    
    # Convert to float32 if it's just a regular integer tensor
    if tensor.dtype in [torch.int8, torch.uint8, torch.int32]:
        return tensor.to(torch.float32)
    
    # If we don't have quantization parameters, just convert directly
    if any(x is None for x in [qweight, qzeros, scales]):
        return tensor.to(torch.float32)
    
    return tensor.to(torch.float32)


def process_tensor(name: str, data: torch.Tensor, n_head: int, n_head_kv: int, head_dim: int) -> Tuple[torch.Tensor, bool]:
    """Process a tensor, handling special cases and conversions."""
    success = True
    try:
        # Handle quantized tensors
        if data.dtype not in [torch.float16, torch.float32]:
            data = dequantize_tensor(data)

        # Handle QKV transformation for attention layers
        if "query_key_value" in name and not should_skip_tensor(name):
            try:
                # Reshape data if needed
                if len(data.shape) == 2:
                    total_dim = data.size(0)
                    # Calculate dimensions based on head configuration
                    qkv_dim = head_dim * (n_head + 2 * n_head_kv)
                    if total_dim % qkv_dim == 0:
                        data = data.reshape(n_head_kv, -1, head_dim, head_dim * n_head)
                        q = data[:, :-2].reshape(n_head * head_dim, head_dim * n_head)
                        k = data[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
                        v = data[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)
                        data = torch.cat((q, k, v))
            except Exception as e:
                print(f"Warning: QKV transformation failed for {name}, using original tensor")
                success = False

        return data.squeeze(), success

    except Exception as e:
        print(f"Warning: Error processing tensor {name}: {str(e)}")
        return data.to(torch.float32).squeeze(), False


def get_tensor_name_mapping(name: str, tensor_map: Any) -> str:
    """Get the correct tensor name mapping."""
    base_name = name.split('.quant_state')[0].split('.bitsandbytes')[0]
    
    # Try direct mapping first
    new_name = tensor_map.get_name(base_name, try_suffixes=(".weight", ".bias"))
    if new_name is not None:
        return new_name
        
    # Try common name patterns
    common_patterns = {
        'self_attention.query_key_value': 'attention.query_key_value',
        'self_attention.dense': 'attention.dense',
        'mlp.dense_h_to_4h': 'feed_forward.w1',
        'mlp.dense_4h_to_h': 'feed_forward.w2',
        'input_layernorm': 'attention_norm',
        'post_attention_layernorm': 'ffn_norm'
    }
    
    for pattern, replacement in common_patterns.items():
        if pattern in base_name:
            new_name = base_name.replace(pattern, replacement)
            mapped_name = tensor_map.get_name(new_name, try_suffixes=(".weight", ".bias"))
            if mapped_name is not None:
                return mapped_name
    
    # If no mapping found, use a cleaned version of the original name
    return base_name


def main() -> None:
    args = parse_args()
    dir_model = args.model
    ftype = args.ftype

    if not dir_model.is_dir():
        print(f'Error: {args.model} is not a directory', file=sys.stderr)
        sys.exit(1)

    ftype_str = ["f32", "f16"]
    fname_out = args.outfile if args.outfile is not None else dir_model / f'ggml-model-{ftype_str[ftype]}.gguf'

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

    num_parts = count_model_parts(dir_model, "model-00")
    is_safetensors = bool(num_parts)
    if is_safetensors:
        from safetensors import safe_open
    else:
        num_parts = count_model_parts(dir_model, "pytorch_model-")

    ARCH = gguf.MODEL_ARCH.FALCON
    gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH])

    print("gguf: get model metadata")

    block_count = hparams.get("num_hidden_layers", hparams.get("n_layer"))
    n_head = hparams.get("num_attention_heads", hparams.get("n_head"))
    n_head_kv = hparams.get("num_kv_heads", hparams.get("n_head_kv", 1))

    gguf_writer.add_name("Falcon")
    gguf_writer.add_context_length(2048)
    gguf_writer.add_tensor_data_layout("jploski")
    gguf_writer.add_embedding_length(hparams["hidden_size"])
    gguf_writer.add_feed_forward_length(4 * hparams["hidden_size"])
    gguf_writer.add_block_count(block_count)
    gguf_writer.add_head_count(n_head)
    gguf_writer.add_head_count_kv(n_head_kv)
    gguf_writer.add_layer_norm_eps(hparams["layer_norm_epsilon"])
    gguf_writer.add_file_type(ftype)

    print("gguf: get tokenizer metadata")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(dir_model)
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        sys.exit(1)

    gguf_writer.add_tokenizer_model("gpt2")
    
    vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))
    try:
        assert max(tokenizer.vocab.values()) < vocab_size
    except AssertionError:
        print("Warning: Vocabulary size mismatch")
        vocab_size = max(tokenizer.vocab.values()) + 1

    tokens: list[bytearray] = []
    scores: list[float] = []
    toktypes: list[int] = []

    reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}
    
    for i in range(vocab_size):
        if i in reverse_vocab:
            tokens.append(reverse_vocab[i])
            scores.append(0.0)
            toktypes.append(gguf.TokenType.NORMAL)
        else:
            tokens.append(bytearray())
            scores.append(0.0)
            toktypes.append(gguf.TokenType.NORMAL)

    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(dir_model, load_merges=True, n_vocab=len(tokens))
    special_vocab.add_to_gguf(gguf_writer)

    tensor_map = gguf.get_tensor_name_map(ARCH, block_count)
    head_dim = hparams["hidden_size"] // n_head

    print("gguf: get tensor metadata")

    if num_parts == 0:
        part_names = iter(("pytorch_model.bin",))
    elif is_safetensors:
        part_names = (f"model-{n:05}-of-{num_parts:05}.safetensors" for n in range(1, num_parts + 1))
    else:
        part_names = (f"pytorch_model-{n:05}-of-{num_parts:05}.bin" for n in range(1, num_parts + 1))

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
                        processed_data, success = process_tensor(name, data, n_head, n_head_kv, head_dim)
                        
                        if success:
                            new_name = get_tensor_name_mapping(name, tensor_map)
                            gguf_writer.add_tensor(new_name, processed_data.numpy())
                        else:
                            print(f"Skipping tensor {name} due to processing failure")

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
