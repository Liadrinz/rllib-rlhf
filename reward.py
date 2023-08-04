import ray
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification



class RankBasedRewardModel:
    
    def __init__(self, model_name_or_path, device="cpu") -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(device)
    
    def reward(self, prompt: str, answer: str):
        inputs = self.tokenizer(prompt, answer, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            score = self.model(**inputs).logits[0].item()
        return score


if __name__ == "__main__":
    reward_model = RankBasedRewardModel("OpenAssistant/reward-model-deberta-v3-large-v2", "cuda:0")
    reward = reward_model.reward(
        "How quickly can I expect to see results from a new workout routine.",
        "That depends on various factors such as the type of routine, how often you exercise, and how much effort you put into it. Generally, you should start to see results after about 3-4 weeks."
    )
    print(reward)
