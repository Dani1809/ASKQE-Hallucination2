import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc, os

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


# =========================
# LOAD MODEL
# =========================
def load_model(model_name, device):
    print(f"[LOAD] {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


def free_model(model, tokenizer):
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


# ==================================
# TRANSLATION FUNCTION (Qwen)
# ==================================
def translate_qwen(text, tokenizer, model, source_lang, target_lang, device):

    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}.\n"
        f"Text: {text}\n"
        f"Translation:"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=True,          # 🔥 abilita variazione
            temperature=0.7,         # 🔥 rischio controllato di hallucination
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

    response = outputs[0][input_ids.shape[-1]:]
    translation = tokenizer.decode(response, skip_special_tokens=True).strip()
    return translation


# =========================
# LOAD DATA
# =========================
def load_data(path):
    print(f"[STEP] Loading dataset: {path}")
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            print(f"[OK] JSON list with {len(data)} samples")
            return data
        else:
            data = [json.loads(line) for line in f if line.strip()]
            print(f"[OK] JSONL with {len(data)} samples")
            return data


# =========================
# SAVE DATA
# =========================
def save_results(data, output_path):
    print(f"[INFO] Saving results to {output_path}")
    with open(output_path, "w", encoding="utf-8") as fout:
        for ex in data:
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("✅ TRANSLATION COMPLETED SUCCESSFULLY!")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Translate text using Qwen Instruct (non-dedicated LLM)")
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--source_language', type=str, default="Italian")
    parser.add_argument('--target_language', type=str, default="English")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model = load_model(MODEL_ID, device)
    data_list = load_data(args.input_path)

    for idx, ex in enumerate(data_list, start=1):
        src = ex.get("src")
        if not src:
            continue

        ex["mt"] = translate_qwen(
            src, tokenizer, model,
            args.source_language, args.target_language, device
        )

        print(f"[{idx}/{len(data_list)}] MT: {ex['mt'][:80]}...")

    save_results(data_list, args.output_path)
    free_model(model, tokenizer)


if __name__ == "__main__":
    main()
