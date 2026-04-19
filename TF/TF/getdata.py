from datasets import load_dataset

dataset = load_dataset("ted_hrlr_translate", "zh_to_en")

train = dataset["train"]

with open("train.zh", "w", encoding="utf-8") as fzh, \
     open("train.en", "w", encoding="utf-8") as fen:

    for item in train:
        fzh.write(item["translation"]["zh"] + "\n")
        fen.write(item["translation"]["en"] + "\n")