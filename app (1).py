
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftConfig, get_peft_model
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import gradio as gr

# ---------- Config ----------
BASE_MODEL = "google/flan-t5-base"
ADAPTER_REPO = "prabhat-30/flan-t5-dialogsum-summarizer"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load Model ----------
def load_model():
    print("🔌 Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)

    print("📐 Loading LoRA config...")
    peft_config = PeftConfig.from_pretrained(ADAPTER_REPO)
    model = get_peft_model(base_model, peft_config)

    print("📥 Loading adapter weights manually...")
    try:
        adapter_path = hf_hub_download(ADAPTER_REPO, "adapter_model.safetensors")
        adapter_weights = load_file(adapter_path, device=device)
    except Exception:
        adapter_path = hf_hub_download(ADAPTER_REPO, "adapter_model.bin")
        adapter_weights = torch.load(adapter_path, map_location=device)

    missing, unexpected = model.load_state_dict(adapter_weights, strict=False)
    print(f"✅ Loaded | missing: {len(missing)} | unexpected: {len(unexpected)}")

    print("🔗 Merging LoRA into base model...")
    model = model.merge_and_unload()
    return tokenizer, model.to(device).eval()

tokenizer, model = load_model()
print("✅ Model ready for inference!")

# ---------- Inference ----------
def summarize_dialogue(dialogue):
    if not dialogue.strip():
        return "Please enter a dialogue."

    input_text = f"summarize: {dialogue.strip()}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------- Gradio UI ----------
demo = gr.Interface(
    fn=summarize_dialogue,
    inputs=gr.Textbox(
        lines=10,
        label="💬 Enter Dialogue / Conversation",
        placeholder="Person1: Hi, how are you?\nPerson2: I'm good! Just got a new job.\nPerson1: Congratulations!"
    ),
    outputs=gr.Textbox(
        label="📝 Summary",
        lines=5
    ),
    title="📌 Smart Dialogue Summarizer",
    description=(
        "This AI tool generates **concise summaries** of conversational dialogues. "
        "Built by fine-tuning **Google FLAN-T5-base** on the **DialogSum** dataset, "
        "it captures key points from multi-turn conversations — ideal for chat logs, "
        "customer support transcripts, and meeting notes."
    ),
    submit_btn="✨ Summarize"
)

if __name__ == "__main__":
    demo.launch(share=False)  # Space pe share=False rakho, port automatically handle hota hai
