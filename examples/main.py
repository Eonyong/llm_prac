import re

import torch
from chapter3 import SelfAttention_v1, SelfAttention_v2
from tokenizer import SimpleTokenizerV1, SimpleTokenizerV2
from pytorch_example import Picture_2_15, Section_2_6

import tiktoken
# from importlib.metadata import version
# print(f"tiktoken ver.: {version("tiktoken")}")

with open('the-verdict.txt', "r", encoding='utf-8') as verdict:
    raw_text = verdict.read()

# tokenizer
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed))
# print(preprocessed[:30])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
# print(vocab_size)

# token sampling
vocab = {token:integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i > 100:
#         break

tokenizer1 = SimpleTokenizerV1(vocab)
tokenizer2 = SimpleTokenizerV2(vocab)
# text = """"It's the last he painted, you know,"
#         Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}
# print(len(vocab))

# text = "Hello, do you like tea?"
# try:
#     print(tokenizer1.encode(text))
# except Exception as e:
#     print(e)

# try:
#     print(tokenizer2.encode(text))
# except Exception as e:
#     print(e)

tokenizer_tiktoken = tiktoken.get_encoding("gpt2")

# text = (
#     "Hello, do you like tea? <|endoftext|> In the Sunlit terraces"
#     " of someunknownPlace."
# )
# # integers = tokenizer_tiktoken.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
# strings = tokenizer_tiktoken.decode(integers)
# print(strings)

enc_text = tokenizer_tiktoken.encode(raw_text)
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
# print(f"x : {x}")
# print(f"y :     {y}")

# code-2.6
# dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)
# second_batch = next(data_iter)
# print(second_batch)

# Picture-2.14
# dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
# data_iter = iter(dataloader)
# inputs, targets = next(data_iter)
# print("입력:\n", inputs)
# print("\n타깃:\n", targets)

# section-2.7
# picture_2_15 = Picture_2_15()
# print(picture_2_15.embedding())
# print(picture_2_15.embedding_vector())

# section-2.8
max_length=4
section_2_6 = Section_2_6()
dataloader = section_2_6.create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
# print("토큰 ID:\n", inputs)
# print("\n입력 크기:\n", inputs.shape)

# Create embedding layer to convert token IDs to embeddings
vocab_size = 50257  # GPT-2 vocabulary size
d_in = 3  # embedding dimension
embedding_layer = torch.nn.Embedding(vocab_size, d_in)
token_embeddings = embedding_layer(inputs)
print("Token embeddings shape:", token_embeddings.shape)

d_out = 2
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
# print(sa_v2(token_embeddings))