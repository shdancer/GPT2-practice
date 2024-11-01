import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载预训练的 GPT-2 模型和 tokenizer


# 设置模型为评估模式
def train_Step(inputs, model, tokenizer, optimizer):
    model.train()
    optimizer.zero_grad()
    # print(inputs)
    input = tokenizer(
        inputs,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=100

    ).to(device)

    # 向前转播
    output = model(input.input_ids, input.attention_mask)

    # 获取每一个词的预测概率分布
    logits = output[:, :-1, :]

    input_modified = input.input_ids[:, 1:]

    # 反向传播
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), input_modified.reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

def test_step(inputs, model, tokenizer):
    # print(inputs)
    # model.eval()
    input = tokenizer(
        inputs,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=100
    ).to(device)

    # print(input)
    half = input.input_ids[:, :input.input_ids.size()[1]//2]
    half_len = half.size()[1]
    # print(half.shape, input.input_ids.shape)
    gen_times = input.input_ids.size()[1] - half.size()[1]

    avg_loss = 0
    for i in range(gen_times):
        output = model(half, torch.ones((input.input_ids.size()[0], half.size()[1])).cuda())
        # print(output.shape)
        logits = output[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)

        predicted_token_id = torch.argmax(probabilities, dim=-1)
        # print(predicted_token_id.shape)
        half = torch.cat((half, predicted_token_id.unsqueeze(-1)), dim=1)

    # print(half.shape)
    half_Str = [ x.split() for x in tokenizer.batch_decode(half.squeeze(0), skip_special_tokens=True)]
    inputs = [ x.split() for x in inputs]
    core = [sentence_bleu(x, y) for x, y in zip(half_Str, inputs)]
    core = sum(core)

    return core











