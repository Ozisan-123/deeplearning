import random

def split_data(src_file, tgt_file, ratio=0.9):
    with open(src_file, encoding="utf-8") as f:
        src = f.readlines()
    with open(tgt_file, encoding="utf-8") as f:
        tgt = f.readlines()

    assert len(src) == len(tgt)

    data = list(zip(src, tgt))
    random.shuffle(data)

    split = int(len(data) * ratio)
    train_data = data[:split]
    val_data = data[split:]

    with open("train.zh", "w", encoding="utf-8") as f1, \
         open("train.en", "w", encoding="utf-8") as f2:
        for s, t in train_data:
            f1.write(s)
            f2.write(t)

    with open("val.zh", "w", encoding="utf-8") as f1, \
         open("val.en", "w", encoding="utf-8") as f2:
        for s, t in val_data:
            f1.write(s)
            f2.write(t)

split_data("train.zh", "train.en")