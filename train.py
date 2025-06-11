# read from input.txt, text has global scope
with open('data/input.txt', 'r') as f:
   text = f.read()

# Print the length and first 200 characters of the dataset
#print("Length of dataset in characters: \n", len(text))
#print("First 200 characters of dataset: \n", text[:200])

# unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#print(stoi)
#print(itos)
#print(encode("hello world"))
#print(decode(encode("hello world")))

import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:3])

# Split the data into train and validation sets
n = int(0.9 * len(data))  # 90% for training, 10% for validation
train_data = data[:n]
val_data = data[n:]