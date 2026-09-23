# 📌 Briefly AI — Smart Dialogue Summarizer

**Fine-tuned FLAN-T5 that turns messy conversations into clean summaries — and translates them across languages.**

Dialogue → Summarize / Translate → Deployed Live on Hugging Face 🚀

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FFD21E?logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?logo=gradio&logoColor=white)
![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Deployed-yellow)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 What is this?

**Briefly AI** is a multi-task NLP system built on Google's **FLAN-T5-base**, fine-tuned end-to-end with **LoRA (PEFT)** to do two jobs with a single set of weights: summarizing multi-turn conversations, and translating English text into multiple languages.

The project was built and shipped solo — from raw dataset curation, through parameter-efficient fine-tuning on a free GPU, to a live, publicly usable web app.

- 📝 Summarizes conversations even when speaker labels (`Person1:`/`Person2:`) are missing — trained on augmented data to generalize to messy, real-world chat text
- 🌍 Translates plain English sentences into other languages using the same model, controlled by a simple task-prefix (no separate models needed)
- 🚀 Fully deployed as a live, interactive Gradio app on Hugging Face Spaces

*Built as an independent, self-driven ML deployment project — end-to-end ownership of data, training, and production hosting.*

---

## ✨ Features

| Capability | Description |
|---|---|
| 📝 **Dialogue Summarization** | Condenses multi-turn conversations into short, accurate summaries |
| 🏷️ **Speaker-Tag Robustness** | Understands dialogues with *or without* `Person1:`/`Person2:` labels, via augmented training data (tags randomly stripped during training) |
| 🌍 **Multilingual Translation** | Translates English sentences into German 🇩🇪 and French 🇫🇷, using the same fine-tuned model |
| 🎯 **Single Multi-Task Model** | One fine-tuned checkpoint handles every task — task-prefix routing decides the behavior at inference time |
| ⚡ **Parameter-Efficient Fine-Tuning** | Trained with **LoRA (Low-Rank Adaptation)** — only a small fraction of parameters updated, making training feasible on a **free Colab T4 GPU** |
| 🖥️ **Live Interactive Demo** | Publicly deployed Gradio app on Hugging Face Spaces (ZeroGPU hardware) |

---

## 🏗️ Architecture & Pipeline

```
                    ┌─────────────────────────────┐
                    │   User Input (raw text)     │
                    │  + Task selected in the UI  │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   Task-Prefix Construction   │  → e.g. "summarize: ..."
                    │                              │     "translate English to German: ..."
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │      T5 Tokenizer            │  → converts text to token IDs
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   FLAN-T5-base + LoRA        │  → base weights frozen during training
                    │   (merged into base weights  │     only LoRA adapters (q, v projections)
                    │    after training)            │    were updated — then merged for inference
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   Beam Search Decoding       │  → num_beams=4, max_new_tokens=128
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   Output shown in Gradio UI  │
                    └─────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Base Model** | Google `flan-t5-base` (250M param encoder-decoder, instruction-tuned) |
| **Fine-Tuning Method** | LoRA (Low-Rank Adaptation) via 🤗 PEFT — rank 16, applied to attention `q`/`v` projections |
| **Optimizer** | Adafactor (T5-recommended — avoids the `fp16` numerical instability T5 models are known for) |
| **Frameworks** | PyTorch, 🤗 Transformers, 🤗 Datasets, 🤗 Accelerate |
| **Training Environment** | Google Colab (free T4 GPU) |
| **Checkpointing** | Google Drive-backed checkpoints (training resumable across dropped Colab sessions) |
| **Model Hosting** | 🤗 Hugging Face Hub |
| **Interface** | Gradio |
| **Deployment** | Hugging Face Spaces (ZeroGPU hardware) |

**Training data:**
- **Summarization:** [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) (12,460 dialogues), plus a speaker-tag-stripped augmented copy of the same set for robustness to messy inputs
- **Translation:** [OPUS-100](https://huggingface.co/datasets/opus100) (German, French) and other parallel corpora, sampled and combined with the summarization data into one multi-task training set

---

## 📁 Project Structure

```
briefly-ai/
├── seq2seq.ipynb       # Full Colab notebook — data prep, augmentation,
│                        # LoRA fine-tuning, checkpointing, model merging
├── app.py               # Gradio app — task-prefix routing + inference
├── requirements.txt      # transformers, accelerate, sentencepiece, spaces
└── README.md             # You are here
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### 1. Clone & install
```bash
git clone <repo-url>
cd briefly-ai
pip install -r requirements.txt
```

### 2. Run locally
```bash
python app.py
```

> 💡 The fine-tuned model downloads automatically from the Hub (`prabhat-30/flan-t5-dialogsum-summarizer`) on first run — no manual setup needed.

### 3. Or just use it live
👉 **[Try Briefly AI on Hugging Face Spaces](https://huggingface.co/spaces/prabhat-30/smart-dialogue-summarizer)**

---

## 🗺️ Roadmap Ideas

- [ ] Swap in a multilingual-tokenizer base model (e.g. mT5 / IndicBART) to properly support Hindi & Hinglish, which FLAN-T5-base's tokenizer can't represent
- [ ] Add a ROUGE/BLEU evaluation dashboard to quantify summary and translation quality
- [ ] Move to always-on (non-sleeping) hosting for lower latency
- [ ] Expand translation to more target languages
- [ ] Add batch/multi-input support to the Gradio app

---

## 🤝 Contributing

Issues and PRs are welcome — this is an active learning project and feedback is always appreciated!

---

*Built with 🧠, a free Colab GPU, and a lot of patience for dependency conflicts.*
