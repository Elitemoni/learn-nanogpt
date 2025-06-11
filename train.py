# read from input.txt, text has global scope
with open('data/input.txt', 'r') as f:
   text = f.read()

# Print the length and first 200 characters of the dataset
#print("Length of dataset in characters: \n", len(text))
#print("First 200 characters of dataset: \n", text[:200])

# unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#print(''.join(chars))
#print(vocab_size)

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
#print(data.shape, data.dtype)
#print(data[:3])

# Split the data into train and validation sets
n = int(0.9 * len(data))  # 90% for training, 10% for validation
train_data = data[:n]
val_data = data[n:]

# we don't train on all the data at once, we train on small blocks
# sometimes called batches, context-length
block_size = 8  # how many characters to predict
train_data[:block_size + 1]  # first block of data

x = train_data[:block_size]  # input block
y = train_data[1:block_size + 1]  # target block (next character)
for t in range(block_size):
   context = x[:t+1]  # current character in the input block
   target = y[t]  # next character in the target block
   print(f"input: {context}, target: {target}")

torch.manual_seed(1337) # for reproducibility
batch_size = 4 # how many independent sequences to process in parallel
block_size = 8 # how many characters to read at once

def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
   # generate a small batch of data of inputs x and targets y
   data = train_data if split == 'train' else val_data
   ix = torch.randint(len(data) - block_size, (batch_size,))  # random starting indices
   x = torch.stack([data[i:i + block_size] for i in ix])  # input block
   y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # target block
   print(type(x, y))
   return x, y

xb, yb = get_batch('train')
print('inputs: ', xb.shape)
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
   for t in range(block_size): # time dimension
      context = xb[b, :t+1]  # current character in the input block
      target = yb[b, t]  # next character in the target block
      print(f"input: {context}, target: {target}")