#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export a RUAR FSDP actor checkpoint to Hugging Face weights."""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.distributed._tensor import DTensor, Placement
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq


def merge_by_placement(tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
    if placement.is_replicate():
        return tensors[0]
    if placement.is_partial():
        raise NotImplementedError("Partial placement is not supported")
    if placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise ValueError(f"Unsupported placement: {placement}")


def model_class_from_config(config):
    architectures = getattr(config, "architectures", None) or []
    arch = architectures[0] if architectures else ""
    if "ForTokenClassification" in arch:
        return AutoModelForTokenClassification
    if "ForCausalLM" in arch:
        return AutoModelForCausalLM
    if "ForConditionalGeneration" in arch:
        return AutoModelForVision2Seq
    raise NotImplementedError(f"Unknown architecture {architectures}")


def find_world_size(local_dir: Path) -> int:
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            return int(match.group(1))
    raise FileNotFoundError(f"No model_world_size_*_rank_0.pt found in {local_dir}")


def save_hf_model(local_dir: Path, state_dict: dict[str, torch.Tensor]) -> None:
    hf_path = local_dir / "huggingface"
    config = AutoConfig.from_pretrained(hf_path)
    auto_model = model_class_from_config(config)
    with torch.device("meta"):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    model.save_pretrained(hf_path, state_dict=state_dict)
    print(f"saved Hugging Face weights to {hf_path}")


def export_checkpoint(local_dir: Path) -> None:
    if local_dir.name == "huggingface":
        raise ValueError("--local-dir should point to the actor checkpoint directory, not actor/huggingface")
    if not (local_dir / "huggingface").is_dir():
        raise FileNotFoundError(f"Missing Hugging Face metadata directory: {local_dir / 'huggingface'}")

    world_size = find_world_size(local_dir)
    rank0 = torch.load(local_dir / f"model_world_size_{world_size}_rank_0.pt", map_location="cpu", weights_only=False)
    pivot_key = sorted(rank0)[0]
    pivot = rank0[pivot_key]

    if not isinstance(pivot, DTensor):
        state_dict = {}
        for key, tensor in rank0.items():
            state_dict[key] = tensor.bfloat16() if torch.is_tensor(tensor) and tensor.is_floating_point() else tensor
        save_hf_model(local_dir, state_dict)
        return

    device_mesh = pivot.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names
    if mesh_dim_names != ("fsdp",):
        raise NotImplementedError(f"Unsupported device mesh: {mesh_dim_names}")

    total_shards = int(mesh.shape[-1])
    shard_state_dicts: list[dict | None] = [rank0] + [None] * (total_shards - 1)

    def load_shard(rank: int) -> None:
        path = local_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        shard_state_dicts[rank] = torch.load(path, map_location="cpu", weights_only=False)

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
        list(executor.map(load_shard, range(1, total_shards)))

    merged: dict[str, torch.Tensor] = {}
    placements_by_key = {}
    for key in set(shard_state_dicts[0].keys()):
        pieces = []
        for shard_state in shard_state_dicts:
            tensor = shard_state.pop(key)
            if isinstance(tensor, DTensor):
                pieces.append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                placements_by_key.setdefault(key, placements)
                if placements_by_key[key] != placements:
                    raise RuntimeError(f"Inconsistent DTensor placement for {key}")
            else:
                merged[key] = tensor.bfloat16() if torch.is_tensor(tensor) and tensor.is_floating_point() else tensor
                pieces = []
                break
        if pieces:
            placements = placements_by_key[key]
            if len(placements) != 1:
                raise NotImplementedError(f"Unsupported placement rank for {key}: {placements}")
            merged[key] = merge_by_placement(pieces, placements[0])

    save_hf_model(local_dir, merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True, help="Path to global_step_*/actor")
    args = parser.parse_args()
    export_checkpoint(args.local_dir)


if __name__ == "__main__":
    main()
