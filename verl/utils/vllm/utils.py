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


from msgspec import field
from packaging import version as vs
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

from verl.third_party.vllm import get_version


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class TensorLoRAWorkerManager(LRUCacheWorkerLoRAManager):
    """An owned loader sharing one worker's native adapter cache.

    Native requests continue through vLLM's own manager. Tensor requests use this
    instance and the inherited validation-before-eviction/activation lifecycle.
    Only cache binding uses a private vLLM field; the supported fork pins 0.19.1.
    """

    @classmethod
    def for_worker(cls, model_runner):
        native = model_runner.lora_manager
        manager = cls(model_runner.vllm_config, native.device, native.embedding_modules, native._lora_model_cls)
        manager._adapter_manager = native._adapter_manager
        return manager

    def _load_adapter(self, request):
        if not isinstance(request, TensorLoRARequest):
            return super()._load_adapter(request)
        helper = PEFTHelper.from_dict(request.peft_config)
        helper.validate_legal(self.lora_config)
        model = self._adapter_manager.model
        return self._lora_model_cls.from_lora_tensors(
            tensors=request.lora_tensors,
            peft_helper=helper,
            lora_model_id=request.lora_int_id,
            device="cpu",
            dtype=self.lora_config.lora_dtype,
            model_vocab_size=self.vocab_size,
            weights_mapper=getattr(model, "hf_to_vllm_mapper", None),
            skip_prefixes=getattr(model, "lora_skip_prefixes", None),
        )


def is_version_ge(pkg: str = "vllm", minver: str = "0.7.3"):
    """check if the package version is greater than or equal to the minimum version"""
    return vs.parse(get_version(pkg)) >= vs.parse(minver)
