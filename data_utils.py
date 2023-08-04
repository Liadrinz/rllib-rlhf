from torch.utils.data.dataset import Dataset


class PromptDataset(Dataset):
    
    def __init__(self, tokenizer, prompt_file) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_file = prompt_file
        with open(prompt_file, "r") as fin:
            self.lines = [line.strip() for line in fin]
    
    def __getitem__(self, index):
        return self.tokenizer(self.lines[index])["input_ids"]
    
    def __len__(self):
        return len(self.lines)
