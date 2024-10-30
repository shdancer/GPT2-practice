from model import GPT2
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
t = tokenizer(
    ["Life is like a box of chocolate", "When you watch it close"],
    padding=True,
    return_tensors="pt",
)
print(t)
# print(tokenizer.vocab_size)
model = GPT2().to("cuda")
t = t.to("cuda")
print(model(t.input_ids, t.attention_mask).shape)
