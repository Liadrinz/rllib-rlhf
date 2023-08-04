import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.typing import ModelConfigDict, TensorType, Union
from ray.rllib.policy.sample_batch import SampleBatch
from torch import nn
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer


def risky_value(value: Any, warn: str):
    warnings.warn(warn.format(value))
    return value


def shift(t: torch.Tensor, value=0, shift=1, dim=-1):
    zeros = (0, 0) * (-dim - 1)
    return F.pad(t, (*zeros, shift, -shift), value=value)


class ActorCriticForLM(TorchModelV2, nn.Module):
    
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        *,
        attention_dim: int,
        max_prompt_length: int,
        max_answer_length: int,
        tokenizer: PreTrainedTokenizer,
        generative: PreTrainedModel,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, tokenizer.vocab_size, model_config, name)
        nn.Module.__init__(self)
        self.attention_dim = attention_dim
        self.max_answer_length = max_answer_length
        self.max_prompt_length = max_prompt_length
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.generative = generative
        self.critic = nn.Linear(self.attention_dim, 1)
        self._features: Optional[TensorType] = None
    
    @staticmethod
    def preprocess_input_ids(input_ids, tokenizer):
        input_ids = input_ids.squeeze(-1).long()
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}
    
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        batch = self.preprocess_input_ids(input_dict[SampleBatch.OBS].values, self.tokenizer)
        prompt_ids = batch["input_ids"]
        prompt_attention_mask = batch["attention_mask"]
        valid_length = max(1, prompt_attention_mask.sum(dim=-1).max())
        prompt_ids = prompt_ids[:, :valid_length]
        prompt_attention_mask = prompt_attention_mask[:, :valid_length]
        prompt_length = prompt_ids.size(1)
        if SampleBatch.ACTIONS in input_dict:
            action_ids = input_dict[SampleBatch.ACTIONS].long()
            output_ids = torch.cat((prompt_ids, action_ids), dim=-1)
        else:
            output_ids = self.generative.generate(
                inputs=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=self.max_answer_length,
            )
        
        # pad actions to max_answer_length for batching
        action_pad_len = prompt_length+self.max_answer_length-output_ids.size(1)
        output_ids = F.pad(output_ids, (0, action_pad_len), value=self.tokenizer.pad_token_id)
        
        output_attention_mask = output_ids[:, prompt_length:] != self.tokenizer.eos_token_id
        output_attention_mask = torch.cat((prompt_attention_mask, output_attention_mask), dim=-1)
        outputs = self.generative(input_ids=output_ids, attention_mask=output_attention_mask, output_hidden_states=True)
        self._features = outputs.hidden_states[-1][:, -1]
        action_logits = outputs.logits[:, prompt_length-1:-1, :]
        action_logits[:, -action_pad_len:, :] = torch.finfo(action_logits.dtype).min
        action_logits[:, -action_pad_len, self.tokenizer.pad_token_id] = 0
        return action_logits, state

    def value_function(self) -> TensorType:
        return self.critic(self._features).squeeze(-1)

    class SequenceDistribution(TorchCategorical):

        @override(ActionDistribution)
        def deterministic_sample(self) -> TensorType:
            self.last_sample = self.dist.probs.argmax(dim=2)
            return self.last_sample
        
        @override(ActionDistribution)
        def logp(self, actions: Any) -> TensorType:
            pad_token_id = self.model.tokenizer.pad_token_id
            mask = actions != pad_token_id
            return super().logp(actions) * mask

        @staticmethod
        def required_model_output_shape(action_space: spaces.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
            return action_space.child_space.n
