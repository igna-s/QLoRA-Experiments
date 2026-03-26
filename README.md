# 🤖 Fine-tuning & Distillation Experiments

Personal ML project focused on fine-tuning and knowledge distillation of language models for medical and legal domains, using QLoRA, DDP multi-GPU, and SFT.

---

## 📦 Models on Hugging Face

| Model | Params | Technique | Notebook |
|-------|--------|-----------|----------|
| [Cukinator/qwen25-14b-medlegal](https://huggingface.co/Cukinator/qwen25-14b-medlegal) | 14B | QLoRA + DDP | `qwen-medlegal-finetunning-14b.ipynb` |
| [Cukinator/qwen25-7b-medlegal](https://huggingface.co/Cukinator/qwen25-7b-medlegal) | 7B | QLoRA + DDP | `qwen-medlegal-finetune.ipynb` |
| [Cukinator/qwen25-7b-destilled-gpt4](https://huggingface.co/Cukinator/qwen25-7b-destilled-gpt4) | 7B | Distillation + QLoRA + DDP | `qwen-distill-igna-s-v1.ipynb` |
| [Cukinator/Mistral-7B-4bit-Ignacio-Final](https://huggingface.co/Cukinator/Mistral-7B-4bit-Ignacio-Final) | 7B | 4-bit Quantization | `destileria-destilosa.ipynb` |

---

## 🧠 Model Descriptions

### 1. `qwen25-14b-medlegal` — Qwen 2.5 14B Fine-tuned Medical-Legal

Fine-tuning of Qwen 2.5 14B on a combined dataset of medical and legal queries. This is the highest-capacity model in the project.

**Training configuration:**
- Technique: QLoRA (nf4, double quant) + SFTTrainer + DDP on 2x T4 (fp16)
- LoRA: `r=32`, `lora_alpha=64`, targets: q/k/v/o_proj + gate/up/down_proj
- Dataset: ChatDoctor-HealthCareMagic-100k + Law Stack Exchange
- Format: Native ChatML with medical/legal role system prompt
- Epochs: 2, effective batch = 32, `max_seq_length=768`
- Estimated time: ~7h on 2x Tesla T4

**Post-processing:** LoRA adapters are merged into the fp16 base model with `merge_and_unload()` before uploading to HF.

---

### 2. `qwen25-7b-medlegal` — Qwen 2.5 7B Fine-tuned Medical-Legal

Lightweight version of the medical-legal model. Same pipeline as the 14B but on Qwen 2.5 7B.

**Training configuration:**
- Technique: QLoRA (nf4) + SFTTrainer + DDP on 2x T4 (fp16)
- LoRA: `r=32`, `lora_alpha=64`
- Dataset: 18,000 medical examples (ChatDoctor) + 638 legal (Law Stack Exchange) -> 17,100 train / 900 val
- System prompts: expert clinical doctor / expert lawyer, selected by domain
- Epochs: 2, effective batch = 32, `max_seq_length=768`
- Final train loss: ~1.65, ~7.5h on 2x Tesla T4
- Trainable params: 80.7M / 7.696B (~1.05%)

**Observation:** The severe imbalance of 18k medical vs. only 638 legal examples makes the model heavily biased toward medical-style responses. A larger legal dataset must be sourced for future training.

---

### 3. `qwen25-7b-destilled-gpt4` — Qwen 2.5 7B Distilled from GPT-4

Knowledge distillation using GPT-4-generated conversations (via OpenHermes-2.5) as the training signal.

**Training configuration:**
- Teacher: GPT-4 (indirect, via `teknium/OpenHermes-2.5`)
- Student: Qwen 2.5 7B Instruct
- Technique: QLoRA + SFTTrainer + DDP on 2x T4, fp16
- Dataset: 15,000 examples from OpenHermes-2.5 -> 14,250 train / 750 val
- Format: ChatML with system/user/assistant roles
- LoRA: `r=32`, `lora_alpha=64`, dropout=0.05
- Epochs: 1, effective batch = 32, `packing=True`, `max_seq_length=512`

**Results (ARC-Challenge, zero-shot, 100 examples):**

| Metric | Value |
|--------|-------|
| Baseline Qwen2.5-7B acc_norm (no fine-tune) | 46.0% |
| Distilled — acc | 43.0% |
| Distilled — acc_norm | 40.0% |
| Delta vs baseline (acc_norm) | **-6.0%** |

**Observation:** Distillation on a general-purpose dataset did not improve ARC benchmark scores. Likely reasons: only 1 training epoch, and `max_seq_length=512` which truncates many responses from OpenHermes. More epochs, a higher `max_seq_length`, or a dataset closer to ARC-style reasoning should help.

---

### 4. `Mistral-7B-4bit-Ignacio-Final` — Mistral 7B 4-bit Quantized

4-bit post-training quantization of Mistral 7B for efficient inference on low-VRAM GPUs.

**Configuration:**
- Technique: Post-training quantization (BitsAndBytes NF4 4-bit, double quant)
- Base: Mistral 7B v0.1
- Hardware: 2x Tesla T4, `device_map="balanced"` (layers 0-12 on GPU 0, layers 13-31 on GPU 1)
- Goal: Reduce memory footprint (~14 GB -> ~4 GB VRAM) for local inference without retraining

**Evaluation results:**

| Metric | Value |
|--------|-------|
| Perplexity (WikiText-2, 4-bit model) | **4.79** |
| ARC-Challenge acc (zero-shot, 100 examples) | 49.0% |
| ARC-Challenge acc_norm | 46.0% |

---

## 📐 LoRA Architecture

The file `qwen_lora_diagram.jsx` contains an interactive diagram illustrating how LoRA adapters are applied over the attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP layers (`gate_proj`, `up_proj`, `down_proj`) of the frozen base model.

---

## 💡 Training Process Learnings

### 🐛 Bugs Found and How They Were Solved

**1. Silent loss=0 — pad_token conflicting with eos_token**

In Qwen2.5, the `eos_token` (`<|im_end|>`) appears at the end of every ChatML turn.
If used as `pad_token`, the data collator masks all positions where this token appears as `-100` (ignored labels), so the entire sequence has no valid training signal — **loss = 0.0 with no error**.

Fix — use the model's neutral padding token instead:

```python
# GOOD: neutral pad token that does not appear in ChatML turns
tokenizer.pad_token = "<|endoftext|>"   # id=151643 in Qwen2.5
```

**2. DDP race condition — workers loading model simultaneously**

When launching with `accelerate launch`, multiple processes try to download the same model weights at the same time, causing corrupted downloads or timeouts.

Fix — pre-download the model to local cache before launching DDP:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen2.5-7B-Instruct", ignore_patterns=["*.msgpack", "*.h5"])
# Now launch: accelerate launch train_ddp.py
```

**3. OOM during LoRA merge — loading base model while adapters are still in VRAM**

Merging adapters into the full fp16 model fails with OOM if the quantized model is still loaded.

Fix — free GPU memory completely before starting the merge step:

```python
import gc, torch
del model        # remove the quantized training model
gc.collect()
torch.cuda.empty_cache()
# Now load base model in fp16 and merge
```

---

## 🔮 Future Improvements / Better Datasets

The following limitations were identified during training and should be addressed in future runs:

### Legal Dataset
The current legal subset (Law Stack Exchange, ~638 examples) is ~28x smaller than the medical subset.
This produces a heavily medically-biased model.

**Better datasets to explore:**
- [pile-of-law/pile-of-law](https://huggingface.co/datasets/pile-of-law/pile-of-law) — large, multi-jurisdiction legal corpus
- [joelniklaus/legal_case_document_summarization](https://huggingface.co/datasets/joelniklaus/legal_case_document_summarization)
- [nguha/legalbench](https://huggingface.co/datasets/nguha/legalbench) — 162 legal reasoning tasks, aligned with benchmark-style eval

### Distillation Dataset
OpenHermes-2.5 is general-purpose. Distilling on it improved conversational fluency but did not transfer structured reasoning to ARC.

**Better datasets to explore:**
- [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) — train directly on ARC-style questions
- [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) — structured reasoning
- [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) — GPT-4 completions with explicit chain-of-thought

### Training Hyperparameters
- Increase `max_seq_length` from 512 -> 1024+ to avoid truncating long OpenHermes responses
- Train for more epochs (1 is likely insufficient for knowledge transfer)
- Experiment with larger LoRA rank (`r=64`) for higher-capacity tasks