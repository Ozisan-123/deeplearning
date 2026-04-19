import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input="train.zh,train.en",
    model_prefix="spm",
    vocab_size=8000,
    character_coverage=0.9995
)