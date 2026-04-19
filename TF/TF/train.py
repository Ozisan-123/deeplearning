import torch
from torch.utils.data import DataLoader
import sentencepiece as spm

from config import Config
from dataset import TranslationDataset, collate_fn
from model import Transformer

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 tokenizer
sp = spm.SentencePieceProcessor()
sp.load("spm.model")

# ===== 训练数据 =====
with open("train.zh", encoding="utf-8") as f:
    src_texts = f.read().splitlines()
with open("train.en", encoding="utf-8") as f:
    tgt_texts = f.read().splitlines()

# ===== 验证数据 =====
with open("val.zh", encoding="utf-8") as f:
    val_src_texts = f.read().splitlines()
with open("val.en", encoding="utf-8") as f:
    val_tgt_texts = f.read().splitlines()

train_dataset = TranslationDataset(src_texts, tgt_texts, sp)
val_dataset = TranslationDataset(val_src_texts, val_tgt_texts, sp)

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                          shuffle=True, collate_fn=collate_fn)

val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                        shuffle=False, collate_fn=collate_fn)
model = Transformer(
    cfg.src_vocab_size,
    cfg.tgt_vocab_size,
    cfg.d_model,
    cfg.num_heads,
    cfg.num_layers,
    cfg.max_len
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

best_val = float("inf")

for epoch in range(cfg.epochs):

    # ===== train =====
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()

        output = model(src, tgt[:, :-1])

        loss = criterion(
            output.reshape(-1, cfg.tgt_vocab_size),
            tgt[:, 1:].reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # ===== validation =====
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)

            output = model(src, tgt[:, :-1])

            loss = criterion(
                output.reshape(-1, cfg.tgt_vocab_size),
                tgt[:, 1:].reshape(-1)
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f}")

    # ===== 保存最优模型（关键）=====
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best.pt")
        print("保存最优模型")

# ===== 推理 =====
def translate(sentence):
    model.eval()
    src = torch.tensor([sp.encode(sentence)]).to(device)

    tgt = torch.tensor([[1]]).to(device)

    for _ in range(50):
        out = model(src, tgt)
        next_token = out[:, -1].argmax(-1).item()

        tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(device)], dim=1)

        if next_token == 2:
            break

    return sp.decode(tgt.squeeze().tolist())


print(translate("我喜欢你"))