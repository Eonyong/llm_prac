import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# code-2.5
class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx:int):
        return self.input_ids[idx], self.target_ids[idx]
    
# section-2.6
class Section_2_6:
    def __init__(self) -> None:
        pass
        
    def create_dataloader_v1(self, txt:str, batch_size:int=4, max_length:int=256, stride:int=128, shuffle:bool=True, drop_table:bool=True, num_workers:int=0):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_table,
            num_workers=num_workers
        )
        return dataloader
    
    # def section2_8(self, raw_text:str, max_length:int):
    #     vocab_size = 50257
    #     output_dim = 256
    #     token_embedding_layer=torch.nn.Embedding(vocab_size, output_dim)
    #     self.create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    #     return token_embedding_layer

# section-2.7
class Picture_2_15:
    def __init__(self) -> None:
        self.input_ids = torch.tensor([2, 3, 5, 1])
        self.vocab_size = 50257
        self.output_dim = 256
        torch.manual_seed(123)
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.output_dim)
        
    def embedding(self):
        return self.embedding_layer.weight
    
    def embedding_vector(self):
        return self.embedding_layer(torch.tensor([3]))