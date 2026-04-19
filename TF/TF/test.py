import torch
import sentencepiece as spm
from config import Config
from model import Transformer

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== model =====
model = Transformer(
    cfg.src_vocab_size,
    cfg.tgt_vocab_size,
    cfg.d_model,
    cfg.num_heads,
    cfg.num_layers,
    cfg.max_len
).to(device)

model.load_state_dict(torch.load("best.pt", map_location=device))
model.eval()

# ===== tokenizer =====
sp = spm.SentencePieceProcessor()
sp.load("spm.model")


def translate(sentence):
    src_ids = [1] + sp.encode(sentence) + [2]
    src = torch.tensor([src_ids]).to(device)

    tgt = torch.tensor([[1]]).to(device)  # BOS

    for _ in range(50):
        out = model(src, tgt)
        next_token = out[:, -1, :].argmax(-1).item()

        tgt = torch.cat(
            [tgt, torch.tensor([[next_token]]).to(device)],
            dim=1
        )

        if next_token == 2:  # EOS
            break

    tokens = tgt.squeeze().tolist()[1:]
    if 2 in tokens:
        tokens = tokens[:tokens.index(2)]

    return sp.decode(tokens)

while(True):
    a = input()
    print(translate(a))