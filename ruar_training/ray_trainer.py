# Copyright 2024 PRIME team and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import statistics
import threading
import time
import uuid
import asyncio
from collections import defaultdict
from contextlib import contextmanager
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import Role, WorkerType, ResourcePoolManager, reduce_metrics, _compute_response_info, \
    _timer
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from . import core_algorithms
from .core_algorithms import compute_return_abs_accuracy, compute_return_smoothness
from ruar.answer_forcing_utility_estimation import estimate_answer_forcing_utilities
from ruar.reflective_step_extraction import (
    char_span_to_token_span,
    find_final_answer_start,
)
from verl.workers.reward_manager.ruar import parallel_compute_score_async


def compute_advantage(data: DataProto, adv_estimator, config):
    if adv_estimator != 'rloo':
        raise NotImplementedError("Public RUAR training supports algorithm.adv_estimator=rloo.")

    responses = data.batch['responses']
    response_length = responses.size(-1)
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    advantages, returns = core_algorithms.compute_rloo_advantage_return(
        data, response_mask, config.actor_rollout_ref.rollout.n, config)
    data.batch['advantages'] = advantages
    data.batch['returns'] = returns
    return data


def compute_data_metrics(batch):

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    metrics = {
        'advantage/mean':
            torch.mean(valid_adv).detach().item(),
        'advantage/max':
            torch.max(valid_adv).detach().item(),
        'advantage/min':
            torch.min(valid_adv).detach().item(),
        'return/mean':
            torch.mean(valid_returns).detach().item(),
        'return/max':
            torch.max(valid_returns).detach().item(),
        'return/min':
            torch.min(valid_returns).detach().item(),
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'adv', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


class RayRUARTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        super().__init__(config,
                         tokenizer,
                         role_worker_mapping,
                         resource_pool_manager,
                         ray_worker_group_cls,
                         reward_fn=reward_fn,
                         val_reward_fn=val_reward_fn)

        self._cot_dump_initialized_paths = set()

    def _validate_config(self):
        super()._validate_config()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.truncation)
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation=self.config.data.truncation)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load dataloader,
        # highlight: Due to some bugs I can't fix, dataloader state will no longer be loaded

        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        self.train_dataloader = torch.load(dataloader_local_path, weights_only=False)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def _decode_prompt_for_probe(self, prompt_ids: torch.Tensor) -> str:
        prompt_ids = prompt_ids[prompt_ids != self.tokenizer.pad_token_id]
        return self.tokenizer.decode(prompt_ids, skip_special_tokens=False)

    def _decode_response_for_probe(self, response_ids: torch.Tensor, response_mask: torch.Tensor) -> str:
        response_ids = response_ids[response_mask.bool()]
        return self.tokenizer.decode(response_ids, skip_special_tokens=False)

    def _build_probe_gen_batch(self, prompts: list[str], max_prompt_length: int) -> DataProto:
        input_ids_lst = []
        attention_mask_lst = []
        raw_prompt_ids = []
        for prompt in prompts:
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt,
                tokenizer=self.tokenizer,
                max_length=max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='left')
            input_ids_lst.append(input_ids[0])
            attention_mask_lst.append(attention_mask[0])
            raw_prompt_ids.append(
                input_ids[0][attention_mask[0].bool()].cpu().numpy().tolist()
            )

        input_ids = torch.stack(input_ids_lst, dim=0)
        attention_mask = torch.stack(attention_mask_lst, dim=0)
        position_ids = compute_position_id_with_mask(attention_mask)
        return DataProto.from_dict(
            tensors={
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            },
            non_tensors={'raw_prompt_ids': np.array(raw_prompt_ids, dtype=object)},
            meta_info={
                'do_sample': True,
                'reflection_probe': True,
                'sampling_kwargs': {
                    'max_tokens': self.config.reflection.get('max_answer_tokens', 64),
                    'n': self.config.reflection.get(
                        'answer_rollouts', self.config.actor_rollout_ref.rollout.n),
                    'temperature': self.config.reflection.get(
                        'temperature', self.config.actor_rollout_ref.rollout.temperature),
                    'top_p': self.config.reflection.get(
                        'top_p', self.config.actor_rollout_ref.rollout.top_p),
                    'top_k': self.config.reflection.get(
                        'top_k', self.config.actor_rollout_ref.rollout.top_k),
                }
            })

    def _generate_sequences_padded(
        self,
        gen_batch: DataProto,
        output_multiplier: int = 1,
        phase_label: str = "generate_sequences",
    ) -> DataProto:
        """Pad prompt batches to the actor worker-group divisor before generation.

        `generate_sequences()` is dispatched with DP_COMPUTE_PROTO, which requires
        equal chunk sizes across worker-group ranks. This becomes a real issue once
        we run with world_size=2 and the prompt batch size is odd, especially for
        answer-forcing probes where the number of unique probe points is data-
        dependent. When sampling with n>1, each padded prompt expands to `n`
        responses, so we remove `pad_size * output_multiplier` rows afterward.
        """
        world_size = self.actor_rollout_wg.world_size
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, world_size)
        prompt_count = int(gen_batch.batch['input_ids'].shape[0])
        padded_prompt_count = int(gen_batch_padded.batch['input_ids'].shape[0])
        phase_label = (
            f"{phase_label} prompts={prompt_count} padded_prompts={padded_prompt_count} "
            f"output_multiplier={output_multiplier}"
        )
        with self._phase_heartbeat(phase_label):
            output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
        return unpad_dataproto(output_padded, pad_size=pad_size * output_multiplier)

    def _answer_forcing_generate_fn(self, prompts: list[str], k: int) -> list[list[str]]:
        if len(prompts) == 0:
            return []
        reflection_cfg = self.config.reflection
        max_prompt_length = reflection_cfg.get('max_probe_prompt_length', None)
        if max_prompt_length is None:
            max_prompt_length = self.config.data.max_prompt_length + self.config.data.max_response_length
        gen_batch = self._build_probe_gen_batch(prompts, max_prompt_length=max_prompt_length)
        output = self._generate_sequences_padded(
            gen_batch,
            output_multiplier=k,
            phase_label=f"answer_forcing_generate k={k}",
        )
        responses = self.tokenizer.batch_decode(output.batch['responses'], skip_special_tokens=True)
        if len(responses) % len(prompts) != 0:
            raise RuntimeError(f'Answer-forcing generated {len(responses)} responses for {len(prompts)} prompts')
        actual_k = len(responses) // len(prompts)
        if actual_k != k:
            raise RuntimeError(f'Expected {k} answer-forcing rollouts per prompt, got {actual_k}')
        return [responses[i * k:(i + 1) * k] for i in range(len(prompts))]

    def _answer_forcing_verify_fn(self, completions, ground_truths, data_sources, extra_infos):
        compute_score = getattr(self.reward_fn, 'compute_score', None)
        if compute_score is None:
            raise RuntimeError('Reflection answer forcing requires RUARRewardManager.compute_score')
        phase_label = (
            f"answer_forcing_verify completions={len(completions)} "
            f"num_processes={self.config.reflection.get('verify_num_processes', 64)}"
        )
        with self._phase_heartbeat(phase_label):
            return asyncio.run(
                parallel_compute_score_async(
                    compute_score,
                    completions,
                    ground_truths,
                    data_sources,
                    extra_info=extra_infos,
                    num_processes=self.config.reflection.get('verify_num_processes', 64)))

    @contextmanager
    def _phase_heartbeat(self, label: str, interval_s: int = 60):
        start = time.monotonic()
        stop_event = threading.Event()

        def _heartbeat():
            while not stop_event.wait(interval_s):
                elapsed = time.monotonic() - start
                print(f"[ruar heartbeat] {label} still running elapsed={elapsed:.1f}s", flush=True)

        print(f"[ruar heartbeat] {label} start", flush=True)
        worker = threading.Thread(target=_heartbeat, daemon=True)
        worker.start()
        try:
            yield
        finally:
            stop_event.set()
            worker.join(timeout=1.0)
            elapsed = time.monotonic() - start
            print(f"[ruar heartbeat] {label} done elapsed={elapsed:.1f}s", flush=True)

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return self._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    def _get_non_tensor_value(self, batch: DataProto, key: str, idx: int, default=None):
        if key not in batch.non_tensor_batch:
            return default
        values = batch.non_tensor_batch[key]
        try:
            return values[idx]
        except Exception:
            return default

    def _cot_dump_path(self, split: str):
        cfg = self.config.trainer.get('cot_dump', {})
        path = cfg.get('path', None)
        if path is None:
            path = os.path.join(self.config.trainer.default_local_dir, 'cot_dumps')
        path = str(path)
        if path.endswith('.jsonl'):
            return path
        return os.path.join(path, f'{split}.jsonl')

    def _maybe_dump_cot_samples(self, batch: DataProto, scores, split: str):
        cfg = self.config.trainer.get('cot_dump', {})
        if not cfg.get('enable', False):
            return
        every_n_steps = int(cfg.get('every_n_steps', 1))
        if every_n_steps > 1 and self.global_steps % every_n_steps != 0:
            return
        max_global_step = cfg.get('max_global_step', None)
        if max_global_step is not None and self.global_steps > int(max_global_step):
            return

        response_len = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_len:]
        prompt_mask = batch.batch['attention_mask'][:, :-response_len]
        n_samples = 1 if split == 'val' else int(self.config.actor_rollout_ref.rollout.n)
        include_prompt = cfg.get('include_prompt', True)
        max_samples_key = 'max_val_samples_per_step' if split == 'val' else 'max_samples_per_step'
        max_samples = int(cfg.get(max_samples_key, cfg.get('max_samples_per_step', 0)))
        limit = len(batch) if max_samples <= 0 else min(len(batch), max_samples)

        scores_list = scores
        if torch.is_tensor(scores_list):
            scores_list = scores_list.detach().cpu().tolist()
        if scores_list is None:
            scores_list = [None] * len(batch)

        path = self._cot_dump_path(split)
        path_dir = os.path.dirname(path)
        if path_dir:
            os.makedirs(path_dir, exist_ok=True)
        if path not in self._cot_dump_initialized_paths:
            mode = 'w' if cfg.get('overwrite', True) else 'a'
            with open(path, mode, encoding='utf-8'):
                pass
            self._cot_dump_initialized_paths.add(path)

        with open(path, 'a', encoding='utf-8') as f:
            for i in range(limit):
                response = self._decode_response_for_probe(batch.batch['responses'][i], response_mask[i])
                prompt = None
                if include_prompt:
                    prompt = self._decode_prompt_for_probe(batch.batch['prompts'][i])
                reward_info = self._json_safe(self._get_non_tensor_value(batch, 'reward_info', i, {}))
                extra_info = self._json_safe(self._get_non_tensor_value(batch, 'extra_info', i, None))
                raw_prompt = self._json_safe(self._get_non_tensor_value(batch, 'raw_prompt', i, None))
                acc = None
                if 'acc' in batch.batch:
                    acc = float(batch.batch['acc'][i].detach().cpu().item())
                record = {
                    'split': split,
                    'global_step': int(self.global_steps),
                    'condition': cfg.get('condition', None) or self.config.trainer.experiment_name,
                    'sample_idx': int(i),
                    'group_idx': int(i // n_samples) if n_samples > 0 else int(i),
                    'rollout_idx': int(i % n_samples) if n_samples > 0 else 0,
                    'uid': self._json_safe(self._get_non_tensor_value(batch, 'uid', i, None)),
                    'data_source': self._json_safe(self._get_non_tensor_value(batch, 'data_source', i, None)),
                    'ability': self._json_safe(self._get_non_tensor_value(batch, 'ability', i, None)),
                    'ground_truth': reward_info.get('ground_truth') if isinstance(reward_info, dict) else None,
                    'score_raw': scores_list[i] if i < len(scores_list) else None,
                    'acc_after_penalty': acc,
                    'prompt_length': int(prompt_mask[i].sum().detach().cpu().item()),
                    'response_length': int(response_mask[i].sum().detach().cpu().item()),
                    'wait_count': int(response.count('Wait')),
                    'alternatively_count': int(response.count('Alternatively')),
                    'boxed_count': int(response.count('\\boxed')),
                    'has_final_answer': bool('Final Answer' in response or 'final answer' in response.lower()),
                    'extra_info': extra_info,
                    'raw_prompt': raw_prompt,
                    'prompt': prompt,
                    'response': response,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _attach_wait_counts(self, batch: DataProto) -> DataProto:
        response_len = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_len:]
        wait_counts = []
        alternatively_counts = []
        for response_ids, mask in zip(batch.batch['responses'], response_mask):
            response = self._decode_response_for_probe(response_ids, mask)
            wait_counts.append(response.count('Wait'))
            alternatively_counts.append(response.count('Alternatively'))
        count_tensors = DataProto.from_dict(tensors={
            'wait_count': torch.tensor(wait_counts, dtype=torch.float32, device=batch.batch['responses'].device),
            'alternatively_count': torch.tensor(
                alternatively_counts,
                dtype=torch.float32,
                device=batch.batch['responses'].device,
            ),
        })
        return batch.union(count_tensors)

    def _round_list(self, values, digits: int = 6):
        return [round(float(v), digits) for v in values]

    def _histogram_metrics(self, prefix: str, values, bins):
        if torch.is_tensor(values):
            values = values.detach().float().cpu().tolist()
        values = [float(v) for v in values]
        metrics = {}
        total = len(values)
        if total == 0:
            return metrics
        for lo, hi in zip(bins[:-1], bins[1:]):
            if hi == bins[-1]:
                count = sum(1 for v in values if lo <= v <= hi)
            else:
                count = sum(1 for v in values if lo <= v < hi)
            key = f"{prefix}/{lo:.2f}_{hi:.2f}"
            metrics[key] = count / total
        return metrics

    def _truncate_debug_text(self, text: str, max_chars: int):
        if max_chars <= 0 or text is None:
            return text
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...[truncated]...\n" + text[-(max_chars - half):]

    def _maybe_attach_reflection_labels(self, batch: DataProto, metrics: dict) -> DataProto:
        reflection_cfg = self.config.get('reflection', {})
        if not reflection_cfg.get('enable', False):
            return batch

        k = reflection_cfg.get('answer_rollouts', self.config.actor_rollout_ref.rollout.n)
        response_len = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_len:]
        response_token_lengths = response_mask.sum(dim=-1).long()
        response_texts = [
            self._decode_response_for_probe(response_ids, mask)
            for response_ids, mask in zip(batch.batch['responses'], response_mask)
        ]
        samples = []
        for i, response in enumerate(response_texts):
            reward_info = batch.non_tensor_batch['reward_info'][i]
            samples.append({
                'prompt': self._decode_prompt_for_probe(batch.batch['prompts'][i]),
                'response': response,
                'ground_truth': reward_info['ground_truth'],
                'data_source': batch.non_tensor_batch['data_source'][i],
                'extra_info': batch.non_tensor_batch.get('extra_info', [None] * len(batch))[i],
            })

        answer_forcing_suffix = reflection_cfg.get('answer_forcing_suffix', None)
        if answer_forcing_suffix is None:
            answer_forcing_suffix = "\n**Final Answer**\n\\boxed"

        include_probe_rollouts = bool(reflection_cfg.get('dump_probe_rollouts', False))
        configured_max_spans = reflection_cfg.get('max_spans_per_sample', 2)
        if configured_max_spans is not None:
            configured_max_spans = int(configured_max_spans)
        estimator_max_spans = (
            None
            if configured_max_spans is not None and configured_max_spans <= 0
            else configured_max_spans
        )
        scaling_cfg = self.config.algorithm.get('advantage_scaling', {})
        estimate_result = estimate_answer_forcing_utilities(
            samples=samples,
            generate_fn=self._answer_forcing_generate_fn,
            verify_fn=self._answer_forcing_verify_fn,
            K=k,
            max_spans_per_sample=estimator_max_spans,
            max_span_chars=reflection_cfg.get('max_span_chars', None),
            cue_types=reflection_cfg.get('cue_types', ['all']),
            answer_forcing_suffix=answer_forcing_suffix,
            ready_threshold=scaling_cfg.get('ready_threshold', 0.75),
            consecutive_required=scaling_cfg.get('consecutive_required', 3),
            stop_after_ready=bool(reflection_cfg.get('stop_after_ready', False)),
            return_probe_evaluations=include_probe_rollouts,
        )
        if include_probe_rollouts:
            labels, _, _, probe_evaluations, ready_boundary_chars = estimate_result
        else:
            labels, _, _, ready_boundary_chars = estimate_result
            probe_evaluations = []

        if estimator_max_spans is None:
            max_spans = max(1, max((label.span_idx + 1 for label in labels), default=0))
        else:
            max_spans = estimator_max_spans
        bsz = len(batch)
        device = batch.batch['responses'].device
        refl_start = torch.zeros((bsz, max_spans), dtype=torch.long, device=device)
        refl_end = torch.zeros((bsz, max_spans), dtype=torch.long, device=device)
        refl_type = torch.zeros((bsz, max_spans), dtype=torch.long, device=device)
        refl_utility = torch.zeros((bsz, max_spans), dtype=torch.float32, device=device)
        refl_raw_utility = torch.zeros((bsz, max_spans), dtype=torch.float32, device=device)
        refl_p_before = torch.zeros((bsz, max_spans), dtype=torch.float32, device=device)
        refl_p_after = torch.zeros((bsz, max_spans), dtype=torch.float32, device=device)
        refl_cue_start_char = torch.zeros((bsz, max_spans), dtype=torch.long, device=device)
        refl_cue_end_char = torch.zeros((bsz, max_spans), dtype=torch.long, device=device)
        refl_span_end_char = torch.zeros((bsz, max_spans), dtype=torch.long, device=device)
        refl_valid = torch.zeros((bsz, max_spans), dtype=torch.bool, device=device)
        refl_ready_boundary_char = torch.full((bsz,), -1, dtype=torch.long, device=device)
        probe_records_by_sample = [[] for _ in range(bsz)]

        filled = [0 for _ in range(bsz)]
        for label in labels:
            slot = filled[label.sample_idx]
            if slot >= max_spans:
                continue
            start, end = char_span_to_token_span(
                response_texts[label.sample_idx],
                label.cue_start_char,
                label.span_end_char,
                self.tokenizer)
            start = min(start, response_len)
            end = min(end, response_len)
            if end <= start:
                continue
            refl_start[label.sample_idx, slot] = start
            refl_end[label.sample_idx, slot] = end
            refl_type[label.sample_idx, slot] = label.cue_type
            refl_utility[label.sample_idx, slot] = label.utility
            refl_raw_utility[label.sample_idx, slot] = label.raw_utility
            refl_p_before[label.sample_idx, slot] = label.p_before
            refl_p_after[label.sample_idx, slot] = label.p_after
            refl_cue_start_char[label.sample_idx, slot] = label.cue_start_char
            refl_cue_end_char[label.sample_idx, slot] = label.cue_end_char
            refl_span_end_char[label.sample_idx, slot] = label.span_end_char
            refl_valid[label.sample_idx, slot] = True
            filled[label.sample_idx] += 1

        if include_probe_rollouts:
            max_probe_records = int(reflection_cfg.get('max_probe_rollouts_per_sample', 16))
            max_probe_text_chars = int(reflection_cfg.get('max_probe_text_chars', 1200))
            include_probe_prefix = bool(reflection_cfg.get('include_probe_prefix', True))
            include_full_completion = bool(reflection_cfg.get('include_probe_full_completion', False))
            for probe in probe_evaluations:
                if probe.sample_idx >= bsz:
                    continue
                records = probe_records_by_sample[probe.sample_idx]
                if max_probe_records > 0 and len(records) >= max_probe_records:
                    continue
                rec = {
                    'char_pos': int(probe.char_pos),
                    'probability': round(float(probe.probability), 6),
                    'scores': self._round_list(probe.scores),
                    'continuations': [
                        self._truncate_debug_text(text, max_probe_text_chars)
                        for text in probe.continuations
                    ],
                }
                if include_probe_prefix:
                    rec['response_prefix_tail'] = self._truncate_debug_text(
                        probe.response_prefix,
                        max_probe_text_chars,
                    )
                if include_full_completion:
                    rec['full_completions'] = [
                        self._truncate_debug_text(probe.response_prefix + text, max_probe_text_chars)
                        for text in probe.continuations
                    ]
                records.append(rec)

        for sample_idx, boundary_char in enumerate(ready_boundary_chars):
            if boundary_char is not None and int(boundary_char) >= 0:
                refl_ready_boundary_char[sample_idx] = int(boundary_char)

        adv_gamma_pos = torch.ones((bsz, response_len), dtype=torch.float32, device=device)
        adv_gamma_neg = torch.ones((bsz, response_len), dtype=torch.float32, device=device)
        scaling_boundaries = [-1 for _ in range(bsz)]
        scaling_ready_threshold = float(scaling_cfg.get('ready_threshold', 0.75))
        scaling_enabled = bool(scaling_cfg.get('enable', False))
        if scaling_enabled:
            def utility_to_gammas(utility_value: float) -> tuple[float, float]:
                utility_value = float(utility_value)
                return max(0.0, 1.0 + utility_value), max(0.0, 1.0 - utility_value)

            post_ready_default_gamma_pos = float(scaling_cfg.get('post_ready_default_gamma_pos', 0.25))
            post_ready_default_gamma_neg = float(scaling_cfg.get('post_ready_default_gamma_neg', 1.25))
            consecutive_required = max(1, int(scaling_cfg.get('consecutive_required', 3)))
            exclude_final_answer = bool(scaling_cfg.get('exclude_final_answer', True))
            boundary_char_positions = []
            boundary_token_positions = []
            boundary_token_fracs = []
            final_answer_protected_samples = 0
            final_answer_protected_tokens = 0
            pre_ready_scaled_spans = 0
            pre_ready_scaled_tokens = 0

            # Group labels by sample_idx for per-sample boundary computation.
            labels_by_sample: dict = {}
            for label in labels:
                labels_by_sample.setdefault(label.sample_idx, []).append(label)

            # Compute ready boundary using consecutive ready spans.
            for sample_idx in range(bsz):
                if int(refl_ready_boundary_char[sample_idx].detach().cpu().item()) >= 0:
                    scaling_boundaries[sample_idx] = int(refl_ready_boundary_char[sample_idx].detach().cpu().item())
                    continue
                sample_labels = labels_by_sample.get(sample_idx, [])
                if not sample_labels:
                    continue
                # sort by cue position so "consecutive" follows response order
                sample_labels_sorted = sorted(sample_labels, key=lambda l: l.cue_start_char)
                streak = 0
                streak_start: int = -1
                for label in sample_labels_sorted:
                    if label.p_before >= scaling_ready_threshold:
                        if streak == 0:
                            streak_start = label.cue_start_char
                        streak += 1
                        if streak >= consecutive_required:
                            scaling_boundaries[sample_idx] = streak_start
                            break
                    else:
                        streak = 0
                        streak_start = -1

            for sample_idx, sample_labels in labels_by_sample.items():
                boundary = scaling_boundaries[sample_idx]
                for label in sample_labels:
                    if boundary >= 0 and label.cue_start_char >= boundary:
                        continue
                    start, end = char_span_to_token_span(
                        response_texts[sample_idx],
                        label.cue_start_char,
                        label.span_end_char,
                        self.tokenizer,
                    )
                    start = min(start, response_len)
                    end = min(end, response_len)
                    if end <= start:
                        continue
                    gamma_pos, gamma_neg = utility_to_gammas(label.utility)
                    adv_gamma_pos[sample_idx, start:end] = gamma_pos
                    adv_gamma_neg[sample_idx, start:end] = gamma_neg
                    pre_ready_scaled_spans += 1
                    pre_ready_scaled_tokens += end - start

            for sample_idx, boundary in enumerate(scaling_boundaries):
                if boundary < 0:
                    continue
                response_token_length = int(response_token_lengths[sample_idx].detach().cpu().item())

                boundary_token, _ = char_span_to_token_span(
                    response_texts[sample_idx],
                    boundary,
                    boundary,
                    self.tokenizer,
                )
                boundary_token = min(boundary_token, response_token_length)
                boundary_char_positions.append(boundary)
                boundary_token_positions.append(boundary_token)
                if response_token_length > 0:
                    boundary_token_fracs.append(boundary_token / response_token_length)

                scaling_end_token = response_token_length
                if exclude_final_answer:
                    final_answer_start = find_final_answer_start(response_texts[sample_idx])
                    if final_answer_start is not None:
                        final_answer_token, _ = char_span_to_token_span(
                            response_texts[sample_idx],
                            final_answer_start,
                            final_answer_start,
                            self.tokenizer,
                        )
                        final_answer_token = min(final_answer_token, response_token_length)
                        if final_answer_token < response_token_length:
                            final_answer_protected_samples += 1
                            final_answer_protected_tokens += response_token_length - final_answer_token
                            scaling_end_token = max(boundary_token, final_answer_token)

                if boundary_token < scaling_end_token:
                    adv_gamma_pos[sample_idx, boundary_token:scaling_end_token] = post_ready_default_gamma_pos
                    adv_gamma_neg[sample_idx, boundary_token:scaling_end_token] = post_ready_default_gamma_neg

        reflection_tensors = DataProto.from_dict(tensors={
            'refl_start': refl_start,
            'refl_end': refl_end,
            'refl_type': refl_type,
            'refl_utility': refl_utility,
            'refl_raw_utility': refl_raw_utility,
            'refl_p_before': refl_p_before,
            'refl_p_after': refl_p_after,
            'refl_cue_start_char': refl_cue_start_char,
            'refl_cue_end_char': refl_cue_end_char,
            'refl_span_end_char': refl_span_end_char,
            'refl_ready_boundary_char': refl_ready_boundary_char,
            'refl_valid': refl_valid,
            'adv_gamma_pos': adv_gamma_pos,
            'adv_gamma_neg': adv_gamma_neg,
        })
        batch = batch.union(reflection_tensors)
        if include_probe_rollouts:
            batch.non_tensor_batch['reflection_probe_rollouts'] = np.array(
                probe_records_by_sample,
                dtype=object,
            )
        valid_count = refl_valid.sum().item()
        valid_p_before = refl_p_before[refl_valid]
        valid_p_after = refl_p_after[refl_valid]
        valid_utility = refl_utility[refl_valid]
        valid_raw_utility = refl_raw_utility[refl_valid]
        positive_count = ((refl_utility > 0.0) & refl_valid).sum().item()
        negative_count = ((refl_utility < 0.0) & refl_valid).sum().item()
        zero_count = (refl_valid & (refl_utility == 0.0)).sum().item()
        metrics.update({
            'reflection/valid_spans': valid_count,
            'reflection/samples_with_span': (refl_valid.any(dim=1).float().mean().item()),
            'reflection/positive_spans': positive_count,
            'reflection/zero_spans': zero_count,
            'reflection/negative_spans': negative_count,
            'reflection/mean_utility': valid_utility.mean().item() if valid_count > 0 else 0.0,
            'reflection/mean_raw_utility': valid_raw_utility.mean().item() if valid_count > 0 else 0.0,
            'reflection/mean_p_before': valid_p_before.mean().item() if valid_count > 0 else 0.0,
            'reflection/mean_p_after': valid_p_after.mean().item() if valid_count > 0 else 0.0,
        })
        if scaling_enabled:
            active = response_mask > 0
            scaled_pos = active & (adv_gamma_pos < 0.999)
            scaled_neg = active & (adv_gamma_neg < 0.999)
            changed_pos = active & ((adv_gamma_pos - 1.0).abs() > 1e-6)
            changed_neg = active & ((adv_gamma_neg - 1.0).abs() > 1e-6)
            active_token_total = active.sum().item()
            scaling_metrics = {
                'reflection_scaling/samples_with_boundary': sum(boundary >= 0 for boundary in scaling_boundaries) / max(1, bsz),
                'reflection_scaling/boundary_char_mean': sum(boundary_char_positions) / max(1, len(boundary_char_positions)),
                'reflection_scaling/boundary_token_mean': sum(boundary_token_positions) / max(1, len(boundary_token_positions)),
                'reflection_scaling/boundary_token_frac_mean': sum(boundary_token_fracs) / max(1, len(boundary_token_fracs)),
                'reflection_scaling/pre_ready_scaled_spans': pre_ready_scaled_spans,
                'reflection_scaling/pre_ready_scaled_tokens': pre_ready_scaled_tokens,
                'reflection_scaling/pre_ready_scaled_token_rate': pre_ready_scaled_tokens / max(1, active_token_total),
                'reflection_scaling/final_answer_excluded': float(exclude_final_answer),
                'reflection_scaling/final_answer_protected_samples': final_answer_protected_samples,
                'reflection_scaling/final_answer_protected_sample_rate': final_answer_protected_samples / max(1, bsz),
                'reflection_scaling/final_answer_protected_token_rate': final_answer_protected_tokens / max(1, active_token_total),
                'reflection_scaling/scaled_pos_token_rate': scaled_pos[active].float().mean().item() if active.any() else 0.0,
                'reflection_scaling/scaled_neg_token_rate': scaled_neg[active].float().mean().item() if active.any() else 0.0,
                'reflection_scaling/changed_pos_token_rate': changed_pos[active].float().mean().item() if active.any() else 0.0,
                'reflection_scaling/changed_neg_token_rate': changed_neg[active].float().mean().item() if active.any() else 0.0,
                'reflection_scaling/pos_gamma_mean': adv_gamma_pos[active].mean().item() if active.any() else 1.0,
                'reflection_scaling/neg_gamma_mean': adv_gamma_neg[active].mean().item() if active.any() else 1.0,
            }
            metrics.update(scaling_metrics)
        if valid_count > 0:
            metrics.update(self._histogram_metrics('reflection/p_before_hist', valid_p_before, [0.0, 0.25, 0.5, 0.75, 1.0]))
            metrics.update(self._histogram_metrics('reflection/p_after_hist', valid_p_after, [0.0, 0.25, 0.5, 0.75, 1.0]))
            metrics.update(self._histogram_metrics('reflection/utility_hist', valid_utility, [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]))
        return batch

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs', 'raw_prompt'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'raw_prompt'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.trainer.validate_sample,
                'validate': True,
            }

            test_output_gen_batch = self._generate_sequences_padded(
                test_gen_batch,
                output_multiplier=1,
                phase_label="validation_generate",
            )
            print('validation generation end')

            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor = self.val_reward_fn(test_batch)
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            test_batch.batch['acc'] = reward_tensor.sum(-1)
            self._maybe_dump_cot_samples(test_batch, scores, split='val')

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)

        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                metrics = {}
                timing_raw = {}

                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs', 'raw_prompt'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'raw_prompt'],
                    )

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self._generate_sequences_padded(
                            gen_batch,
                            output_multiplier=self.config.actor_rollout_ref.rollout.n,
                            phase_label=f"train_rollout_generate step={self.global_steps}",
                        )

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    metrics_ratio = self.count_prefix_ratio(batch)
                    print(metrics_ratio)
                    metrics.update(metrics_ratio)

                    # verify
                    with _timer('verify', timing_raw):
                        n_samples = self.config.actor_rollout_ref.rollout.n
                        phase_label = (
                            f"verify step={self.global_steps} "
                            f"samples={int(batch.batch['responses'].shape[0])} n={n_samples}"
                        )
                        with self._phase_heartbeat(phase_label):
                            scores = self.reward_fn.verify(batch, n_samples=n_samples)
                        metrics['acc'] = statistics.mean(scores)
                        metrics.update(self.metric_sources(batch))

                    batch = self._attach_wait_counts(batch)
                    self._maybe_dump_cot_samples(batch, scores, split='train')

                    batch.meta_info['n'] = self.config.actor_rollout_ref.rollout.n
                    n_samples = self.config.actor_rollout_ref.rollout.n

                    with _timer('reflection_label', timing_raw):
                        batch = self._maybe_attach_reflection_labels(batch, metrics)

                    batch.meta_info['avg_response_length'] = batch.batch[
                        'attention_mask'][:, -batch.batch['responses'].shape[-1]:].sum(dim=-1).float().mean().item()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        phase_label = (
                            f"old_log_prob step={self.global_steps} "
                            f"samples={int(batch.batch['responses'].shape[0])}"
                        )
                        with self._phase_heartbeat(phase_label):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            phase_label = (
                                f"ref_log_prob step={self.global_steps} "
                                f"samples={int(batch.batch['responses'].shape[0])}"
                            )
                            with self._phase_heartbeat(phase_label):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with _timer('adv', timing_raw):
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  config=self.config)
                        if 'adv_metrics'in batch.meta_info:
                            metrics.update(batch.meta_info['adv_metrics'])

                        metrics.update({
                            'return/sign_accuracy':
                                compute_return_abs_accuracy(batch.batch['returns'], batch.batch['acc']).item()
                        })

                        metrics.update({'return/smoothness': compute_return_smoothness(batch.batch['returns']).item()})

                    with _timer('update_actor', timing_raw):
                        ppo_epochs = max(1, int(self.config.actor_rollout_ref.actor.ppo_epochs))
                        for ppo_epoch in range(1, ppo_epochs + 1):
                            phase_label = (
                                f"update_actor step={self.global_steps} ppo_epoch={ppo_epoch} "
                                f"samples={int(batch.batch['responses'].shape[0])}"
                            )
                            with self._phase_heartbeat(phase_label):
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                    metrics.update(actor_output_metrics)
                    metrics['ppo_epoch'] = ppo_epoch

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None and self.config.trainer.get('run_final_validation', True):
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return

    def count_prefix_ratio(self, batch):
        # Check what proportion of prefix is shared in the same prompt
        n_samples =self.config.actor_rollout_ref.rollout.n
        responses = batch.batch['responses']
        response_len = responses.size(1)
        attention_mask = batch.batch['attention_mask'][:, -response_len: ]
        prefix_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for start_pos in range(0, len(batch), n_samples):
            for i in range(n_samples):
                for j in range(i+1,n_samples):
                    prefix_len = (responses[start_pos+i] == responses[start_pos+j]).cumprod(dim=0).sum()
                    prefix_mask[[start_pos+i,start_pos+j],:prefix_len] = True
        prefix_mask[attention_mask==0]=False
        return {
            'prefix_ratio': prefix_mask.sum().item() / attention_mask.sum().item()
        }

    def metric_sources(self, batch):
        # Separately count acc for data from different sources
        sources = batch.non_tensor_batch['data_source']
        acc = batch.batch['acc'].cpu().tolist()
        metrics = defaultdict(list)
        for a,s in zip(acc, sources):
            key_name = 'train_acc/'+s
            metrics[key_name].append(a)

        for k,v in metrics.items():
            metrics[k] = statistics.mean(v)
        return metrics
