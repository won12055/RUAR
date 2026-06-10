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
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['gsm8k', 'math500', 'aime2024', 'aime2025'] or str(data_source).startswith('boxed_math/'):
        from . import boxed_math
        data_name = str(data_source).split('/', 1)[-1]
        res = boxed_math.compute_score(solution_str, ground_truth, data_name=data_name)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina', 'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc',
            'numina_cn_k12', 'numina_olympiads', 'numina_3k', 'aime-aops', 'SYNTHETIC-1', 'ttrl'
    ]:
        from . import math_answer
        res = math_answer.compute_score(solution_str, ground_truth)
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ['deepscaler']:
        from . import hf_math_verify
        res = hf_math_verify.compute_score(solution_str, ground_truth)
    elif data_source.startswith('mcq/'):
        from . import mcq
        res = mcq.compute_score(solution_str, ground_truth, extra_info=extra_info)
    else:
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
