import random

import ray
from ray import air, tune
from ray.air.config import CheckpointConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer

from algo import PPOForLM
from data_utils import PromptDataset
from lm_env import HuggingFaceLMEnv
from model import ActorCriticForLM
from reward import RankBasedRewardModel


ray.init(local_mode=True)

ModelCatalog.register_custom_model("ac4lm", ActorCriticForLM)
ModelCatalog.register_custom_action_dist("sequence_dist", ActorCriticForLM.SequenceDistribution)

max_prompt_length = 16
max_answer_length = 32
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
reward_model = ray.remote(RankBasedRewardModel).remote("OpenAssistant/reward-model-deberta-v3-base", "cpu")
config = (
    PPOConfig(algo_class=PPOForLM)
    .environment(
        HuggingFaceLMEnv,
        env_config={
            "tokenizer": tokenizer,
            "reward_model": reward_model,
            "max_prompt_length": max_prompt_length,
            "max_answer_length": max_answer_length,
            "prompt_dataset": PromptDataset(tokenizer, "demo_prompt.txt"),
        }
    )
    # .resources(num_gpus=2)
    .rollouts(rollout_fragment_length=1)
    .framework("torch")
    .training(
        model={
            "custom_model": "ac4lm",
            "custom_action_dist": "sequence_dist",
            "custom_model_config": {
                "tokenizer": tokenizer,
                "generative": model,
                "attention_dim": model.config.n_embd,
                "max_prompt_length": max_prompt_length,
                "max_answer_length": max_answer_length,
            },
        },
        train_batch_size=2,
        sgd_minibatch_size=2,
    )
    .evaluation(evaluation_interval=1)
    .experimental(
        _disable_preprocessor_api=False,
    )
)
tuner = tune.Tuner(
    PPOForLM,
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop={
            "training_iteration": 10
        },
        verbose=1,
    )
)
tuner.fit()
