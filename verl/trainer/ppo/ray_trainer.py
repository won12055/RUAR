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

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Type

import numpy as np
import ray
from codetiming import Timer

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class Role(Enum):
    ActorRollout = 0
    RefPolicy = 1


@dataclass
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for name, process_on_nodes in self.resource_pool_spec.items():
            self.resource_pool_dict[name] = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=name,
            )
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def _check_resource_available(self):
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0)
            for node, node_info in node_available_resources.items()
        }
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            num_gpus
            for process_on_nodes in self.resource_pool_spec.values()
            for num_gpus in process_on_nodes
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than required GPUs {total_required_gpus}"
            )

        for pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {pool_name} cannot be satisfied by the Ray cluster")


def reduce_metrics(metrics: dict):
    return {key: np.mean(val) for key, val in metrics.items()}


def _compute_response_info(batch: DataProto):
    response_width = batch.batch["responses"].shape[-1]
    prompt_mask = batch.batch["attention_mask"][:, :-response_width]
    response_mask = batch.batch["attention_mask"][:, -response_width:]
    return {
        "response_mask": response_mask,
        "prompt_length": prompt_mask.sum(-1).float(),
        "response_length": response_mask.sum(-1).float(),
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None, text="{name}: {seconds:.1f} seconds") as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer:
    """Minimal Ray trainer support used by the RUAR training driver."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        if real_train_batch_size % n_gpus != 0:
            raise ValueError(
                f"data.train_batch_size * rollout.n ({real_train_batch_size}) must be divisible by GPUs ({n_gpus})"
            )

        if config.actor_rollout_ref.actor.strategy != "fsdp":
            raise NotImplementedError("RUAR public training supports FSDP actor training only.")

        if (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ) and not config.actor_rollout_ref.model.use_remove_padding:
            raise ValueError("Sequence parallelism requires actor_rollout_ref.model.use_remove_padding=True.")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        generations_to_log = int(self.config.trainer.val_generations_to_log_to_wandb)
        if generations_to_log <= 0:
            return
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda item: item[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        self.validation_generations_logger.log(
            self.config.trainer.logger,
            samples[:generations_to_log],
            self.global_steps,
        )

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        resource_pool_to_cls = {
            pool: {}
            for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        actor_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        resource_pool_to_cls[actor_pool]["actor_rollout"] = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )

        if self.use_reference_policy:
            ref_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            resource_pool_to_cls[ref_pool]["ref"] = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )

        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            worker_group = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            all_wg.update(worker_group.spawn(prefix_set=class_dict.keys()))
            self.wg_dicts.append(worker_group)

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
