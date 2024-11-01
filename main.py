import time

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from model import GPT2
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train import train_Step, test_step
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset



epochs = 50
batch_size = 64

# load datasets

# class SentenceDataset(Dataset):
#     def __init__(self, file_path, mode='train'):
#         with open(file_path, 'r') as f:
#             self.sentences = f.readlines()
#
#         # 移除每个句子末尾的换行符
#         self.sentences = [sentence.strip() for sentence in self.sentences]
#
#     def __len__(self):
#         return len(self.sentences)
#
#     def __getitem__(self, idx):
#         return self.sentences[idx]
#
# file_path = 'sentences_dataset.txt'
# dataset = SentenceDataset(file_path)
#
# train_dataset = torch.utils.data.dataset.Subset(dataset, range(90))
# test_dataset = torch.utils.data.dataset.Subset(dataset, range(90, 100))

dataset = load_dataset('roneneldan/TinyStories')
train_dataset = dataset['train'].select(range(300))

test_dataset = dataset['validation'].select(range(100))
# # 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2().to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model.parameters(), lr=1e-4)



loss_last = []
bleu_list = []
avg_loss_list =[]
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"train: Epoch {epoch+1}"):
        # print(batch['text'])
        loss = train_Step(batch['text'], model, tokenizer, optimizer)
        total_loss += loss


    avg_loss = total_loss / len(train_dataset) * batch_size
    loss_last.append(avg_loss)
    print(f"avg_loss = {avg_loss}\n")


    with torch.no_grad():
        t = 0
        sum_bleu = 0

        for batch in tqdm(test_loader, desc=f"test: Epoch{epoch+1}"):
            bleu = test_step(batch['text'], model, tokenizer)
            sum_bleu = sum_bleu + bleu

        print(sum_bleu)

        bleu_list.append(sum_bleu/len(test_loader))


torch.save(model.state_dict(), f'pretrained/{time.time()}.pt')
loss_last = np.array(loss_last)
bleu_list = np.array(bleu_list)


np.savetxt("metric/Bleu.txt", bleu_list)

plt.figure(0)
plt.plot(loss_last, 'b')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss-epoch.jpg', dpi=300)
plt.show()





