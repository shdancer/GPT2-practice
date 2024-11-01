from model import GPT2
from transformers import GPT2Tokenizer
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def test(model, tokenizer):
    input = tokenizer(
        # ["He likes", "The little", "He is", "She likes", "We had", "They took", "The ice ", "The house", "They are", The birds"],
        ["he", "I", "She"],
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)


    avg_loss = 0

    for i in tqdm(range(30)):
        output = model(input.input_ids, input.attention_mask)
        # print(output.shape)
        logits = output[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)

        probabilities_ids = torch.argmax(probabilities, dim=1)

        input.input_ids = torch.cat((input.input_ids, probabilities_ids.unsqueeze(-1)), dim=1)

        # print(input.attention_mask.shape)
        input.attention_mask = torch.cat((input.attention_mask, torch.ones(input.input_ids.size()[0], 1).to("cuda")), dim=1)


    # print(half.shape)
    output = tokenizer.batch_decode(input.input_ids, skip_special_tokens=True)


    return output

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


model = GPT2().to(device)
model.load_state_dict(torch.load('pretrained/1730453676.006962.pt'))


output = test(model, tokenizer)

print(output)



