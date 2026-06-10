from __future__ import annotations

import os
from pathlib import Path
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from .ray_trainer import RayRUARTrainer


os.environ.setdefault("RUAR_ROOT", str(Path(__file__).resolve().parents[1]))


@hydra.main(config_path="config", config_name="ruar_trainer", version_base=None)
def main(config):
    run_ruar(config)


def _ray_init_kwargs() -> dict:
    kwargs = {}
    temp_dir = os.environ.get("RUAR_RAY_TEMP_DIR")
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        kwargs["_temp_dir"] = temp_dir

    metrics_port = os.environ.get("RUAR_RAY_METRICS_PORT")
    if metrics_port:
        kwargs["_metrics_export_port"] = int(metrics_port)
    return kwargs


def run_ruar(config, compute_score=None):
    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                }
            },
            **_ray_init_kwargs(),
        )
    ray.get(main_task.remote(config, compute_score))


def _needs_reference_policy(config) -> bool:
    return bool(config.actor_rollout_ref.actor.get("use_kl_loss", False))


@ray.remote(num_cpus=1)
def main_task(config, compute_score=None):
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.workers.reward_manager import RUARRewardManager

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_model_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_model_path)

    if config.actor_rollout_ref.actor.strategy != "fsdp":
        raise NotImplementedError("RUAR public training supports actor_rollout_ref.actor.strategy=fsdp.")

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    }
    mapping = {
        Role.ActorRollout: "global_pool",
    }
    if _needs_reference_policy(config):
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = "global_pool"

    reward_fn = RUARRewardManager(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
    val_reward_fn = RUARRewardManager(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    trainer = RayRUARTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=ResourcePoolManager(
            resource_pool_spec={"global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
            mapping=mapping,
        ),
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
