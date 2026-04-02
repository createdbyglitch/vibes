# Vibes

**A transformer language model from scratch in pure C++17. Zero dependencies.**

Vibes is a complete, self-contained language model implementation that you can actually train and run. No PyTorch, no TensorFlow, no CUDA — just standard C++17 and your CPU.

```

./vibes train --data shakespeare.txt
./vibes gen --ckpt vibes.ckpt --tok vibes.tok --prompt "To be or not"

```

## Why?

Because understanding transformers means building them from scratch. This code implements everything:

- Full backpropagation with analytical gradients (no finite difference)
- Byte-Pair Encoding (BPE) tokenizer built from scratch
- Multi-head causal self-attention with KV caching
- AdamW optimizer + cosine LR schedule + warmup + gradient clipping
- LoRA adapters for efficient fine-tuning
- Dropout, LayerNorm, GeLU — all with correct backward passes
- Sampling: temperature, top-k, top-p, repetition penalty, beam search
- Checkpoint save/load

## Quick Start

### Build

```

c++ -std=c++17 -O3 -march=native -flto -o vibes vibes.cpp

```

### Train

```

./vibes train --data /path/to/text.txt --ckpt mymodel.ckpt

```

Training options:
--n_embd 256        # model width
--n_layer 6         # depth  
--batch 8           # batch size
--lr 3e-4           # learning rate
--dropout 0.1       # dropout probability
--bpe_vocab 2048    # BPE vocabulary size
--lora_rank 8       # LoRA rank (0 = no LoRA)
--resume            # resume from checkpoint

### Generate

```

./vibes gen --ckpt mymodel.ckpt --tok vibes.tok --prompt "Once upon a time"

```

Generation options:
--max_new 200       # tokens to generate
--temp 0.8          # temperature
--top_k 40          # top-k sampling
--top_p 0.9         # nucleus sampling
--rep_pen 1.1       # repetition penalty
--beam 4            # beam search width
--personality aight_bet

## Personalities

| Personality | Vibe |
|-------------|------|
| aight_bet | Memes, lowercase energy, "fr fr", "no cap" |
| professional | Clean, neutral, grammatically correct |
| chaotic | Maximum entropy, unpredictable |

## License

GPL v3
