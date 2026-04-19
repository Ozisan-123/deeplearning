class Config:
    src_vocab_size = 8000
    tgt_vocab_size = 8000

    d_model = 512
    num_heads = 4
    num_layers = 4

    max_len = 512
    batch_size = 32
    lr = 1e-4
    epochs = 10

    pad_id = 0
    bos_id = 1
    eos_id = 2