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
"""vLLM rollout for the public RUAR FSDP training path."""

import os

import numpy as np
from typing import List
from contextlib import contextmanager

from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _as_token_id_list(token_ids) -> List[int]:
    if isinstance(token_ids, np.ndarray):
        return token_ids.tolist()
    if isinstance(token_ids, torch.Tensor):
        return token_ids.detach().cpu().tolist()
    return list(token_ids)


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the Hugging Face config used to initialize the vLLM model
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        enable_sleep_mode = _as_bool(config.get("enable_sleep_mode", True), default=True)
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=enable_sleep_mode,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            seed=int(os.getenv("RANK", "0")) // tensor_parallel_size,
        )

        # Offload vLLM model to reduce peak memory usage when sleep mode is enabled.
        initial_sleep = _as_bool(config.get("initial_sleep", True), default=True)
        if enable_sleep_mode and initial_sleep:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=1,
            max_tokens=config.response_length,
        )

        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': _as_token_id_list(raw_prompt_ids), 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': _as_token_id_list(raw_prompt_ids)
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        is_validating = prompts.meta_info.get('validate', False)
        do_sample = prompts.meta_info.get('do_sample', True)
        sampling_n = kwargs.get('n', self.config.n)
        sampling_response_length = kwargs.get('max_tokens', self.config.response_length)
        if is_validating and do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': self.config.get('top_p', 0.95),
                'top_k': self.config.get('top_k', -1),
                'min_p': self.config.get('min_p', 0.0),
                'temperature': self.config.get('temperature', 0.6),
                'n': 1
            }
            sampling_n = 1
        elif not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
            sampling_n = 1

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)

        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

        response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response.append(output.outputs[sample_id].token_ids)

        if self.config.get('dedup', True):
            for i in range(len(response)):
                response[i] = dedupe_tensor(torch.tensor(response[i])).numpy().tolist()

        if sampling_n > 1 and do_sample and not is_validating:
            if 'data_source' in non_tensor_batch.keys():
                non_tensor_batch['data_source'] = _repeat_interleave(non_tensor_batch['data_source'],
                                                                            sampling_n)
            if 'reward_info' in non_tensor_batch.keys():
                non_tensor_batch['reward_info'] = _repeat_interleave(non_tensor_batch['reward_info'], sampling_n)

        response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=sampling_response_length).to(idx.device)

        if sampling_n > 1 and do_sample and not is_validating:
            idx = _repeat_interleave(idx, sampling_n)
            attention_mask = _repeat_interleave(attention_mask, sampling_n)
            position_ids = _repeat_interleave(position_ids, sampling_n)
            batch_size = batch_size * sampling_n
            if 'multi_modal_inputs' in non_tensor_batch.keys():
                non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                            sampling_n)
            if 'raw_prompt' in non_tensor_batch.keys():
                non_tensor_batch['raw_prompt'] = _repeat_interleave(non_tensor_batch['raw_prompt'],
                                                                            sampling_n)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def dedupe_tensor(x: torch.Tensor, threshold: int = 5) -> torch.Tensor:
    """Remove repeated adjacent token blocks from a 1D token tensor."""
    L = x.size(0)
    for n in range(1, 1000):
        m = L // n
        if m <= threshold:
            continue

        main = x[: m * n].view(m, n)
        tail = x[m * n :]

        eq = (main[1:] == main[:-1]).all(dim=1)  # length = m-1
        if not eq.any():
            continue

        eq_int = eq.int()
        padded = torch.cat([
            eq_int.new_zeros(1),
            eq_int,
            eq_int.new_zeros(1),
        ])  # length = m+1
        dif = torch.diff(padded)  # length = (m+1)-1 = m

        starts = (dif == 1).nonzero(as_tuple=True)[0]
        ends = (dif == -1).nonzero(as_tuple=True)[0] - 1

        mask = torch.ones(m, dtype=torch.bool, device=x.device)
        for s, e in zip(starts.tolist(), ends.tolist()):
            run_len = e - s + 1
            if run_len >= threshold:
                drop_idx = torch.arange(s+1, e+2, device=x.device)
                mask[drop_idx] = False

        if mask.all():
            continue

        kept = main[mask].reshape(-1)
        x = torch.cat([kept, tail], dim=0)
        L = x.size(0)

    return x
