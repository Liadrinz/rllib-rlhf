import ray

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gymnasium import Env
from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import Discrete, Box
from ray.rllib.utils.spaces.repeated import Repeated


class InputIdsSpace(Repeated):
    
    def __init__(self, child_space, sample_len, max_len):
        super().__init__(child_space, max_len)
        self.sample_length = sample_len
    
    def sample(self):
        return [
            self.child_space.sample()
            for _ in range(self.np_random.integers(1, self.sample_length + 1))
        ]
    
    @property
    def shape(self):
        r"""
        cope with `get_dummy_batch_for_space`
        """
        return (self.sample_length,)


class HuggingFaceLMEnv(Env):
    
    def __init__(self, env_config: Dict) -> None:
        super().__init__()
        
        self.tokenizer = env_config["tokenizer"]
        self.reward_model = env_config["reward_model"]
        self.max_prompt_length = env_config["max_prompt_length"]
        self.max_answer_length = env_config["max_answer_length"]
        self.prompt_dataset = env_config.get("prompt_dataset", None)
        self.vocab_size = getattr(self.tokenizer, "vocab_size", env_config.get("vocab_size", None))
        if self.vocab_size is None:
            raise ValueError(f"Please specify the vocab_size in env_config['vocab_size']")
        assert self.max_prompt_length > 0 and self.max_answer_length > 0
        
        self.spec = EnvSpec("huggingface-lm", max_episode_steps=1)
        self.observation_space = InputIdsSpace(
            Box(-1, self.vocab_size, shape=(), dtype=int),
            sample_len=self.max_prompt_length,
            max_len=self.max_prompt_length+self.max_answer_length
        )
        self.action_space = InputIdsSpace(
            Discrete(self.vocab_size),
            sample_len=self.max_answer_length,
            max_len=self.max_answer_length,
        )
        self.prompt_ids = []
        self.input_ids = []
    
    @property
    def state(self):
        return self.input_ids
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        if self.prompt_dataset is None:
            self.prompt_ids = self.input_ids = [Discrete(self.vocab_size).sample()]
        else:
            self.prompt_ids = self.input_ids = self.prompt_dataset[Discrete(len(self.prompt_dataset)).sample()][:self.max_prompt_length]
        return self.state, {}
    
    def step(self, action: Union[List[int], np.ndarray]) -> Tuple[Any, float, bool, bool, dict]:
        if isinstance(action, np.ndarray):
            action_ids = action.tolist()
        else:
            action_ids = list(action)
        self.input_ids = self.input_ids + action_ids
        reward_ref = self.reward_model.reward.remote(
            self.tokenizer.decode(self.prompt_ids),
            self.tokenizer.decode(action_ids),
        )
        reward = ray.get(reward_ref)
        return self.state, reward, True, True, {}
