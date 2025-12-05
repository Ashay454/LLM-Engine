# 🧠 Llama2.c - The Neural Dashboard

**Stop treating AI like a black box. Watch it think.**

*(Replace this line with your actual screenshot or GIF of the blue bars moving)*

## 🚀 Overview

This is a **pure C implementation** of the Llama 2 Large Language Model inference engine. It runs with **zero external dependencies** (no PyTorch, no Python, no Accelerate)—just raw memory management and matrix multiplication.

**The Twist:**
Unlike standard inference engines that just output text, this engine features a real-time **Neural Dashboard**. It taps directly into the model's logit layer to visualize the probability distribution *before* a token is selected.

You can watch the model's confidence fluctuate in real-time—seeing exactly when it is "hallucinating" (low confidence) versus when it is reciting facts (high confidence).

## ✨ Features

  * **100% Pure C:** Built from scratch using standard C libraries (`stdlib`, `math`, `time`).
  * **Zero Dependencies:** No heavy ML frameworks. If you have a C compiler (GCC/Clang), it runs.
  * **Real-Time Visualization:** A custom terminal UI that displays top-5 token probabilities dynamically.
  * **Llama 2 Architecture:** Implements RoPE (Rotary Positional Embeddings), KV Caching, RMSNorm, and SwiGLU activations.
  * **Cross-Platform:** Runs natively on Windows (MinGW), Linux, and macOS.

## 🛠️ Quick Start

### 1\. Clone & Download Weights

You need the model weights (`stories15M.bin`) and the tokenizer (`tokenizer.bin`).

```bash
# Download Model (15M Parameters - ~60MB)
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

# Download Tokenizer
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/tokenizer.bin
```

### 2\. Compile

Use `gcc` with optimization flags for best performance.

**Linux/Mac:**

```bash
gcc -O3 -o engine engine.c -lm
```

**Windows (MinGW/PowerShell):**

```bash
gcc -O3 -o engine engine.c
```

### 3\. Run the "Neural Dashboard"

```bash
./engine
```

## 🧠 How It Works

### The Architecture

The engine manually allocates memory for the Transformer weights and runs the forward pass mathematically:

1.  **Tokenization:** Converts input text into integer IDs.
2.  **Embedding:** Looks up vector representations.
3.  **Transformer Layers:** Passes vectors through Attention Heads and Feed-Forward Networks.
4.  **Logit Generation:** Calculates the probability of the next word across the entire vocabulary (32,000 tokens).

### The Visualization (The "Why")

Standard engines use `argmax` or hidden sampling to produce text. This engine exposes the uncertainty behind the curtain.

  * **Green Bars:** High confidence (\>90%). The model is traversing a well-trodden path (e.g., reciting a common phrase like "Once upon a time").
  * **Yellow/Blue Bars:** High entropy. The model is splitting probability mass between multiple valid continuations. This is where "creativity" (and hallucination) happens.

## 📊 Performance

  * **Hardware:** Tested on a Standard Dual-Core Laptop.
  * **Speed:** \~35-40 tokens/second (with Visualizer enabled).
  * **Optimization:** Uses `-O3` compiler optimizations for loop unrolling and vectorization.

## 🤝 Credits & Inspiration

  * **Andrej Karpathy:** For the original `llama2.c` project which inspired the base architecture.
  * **Meta AI:** For the Llama 2 model architecture.

## 📝 License

MIT License. Feel free to fork, modify, and learn\!
