import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gc, os


# =========================
# LOAD MODEL
# =========================

def load_model(model_name, device):
    print(f"[LOAD] {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model

def free_model(model, tokenizer):
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

# ==================================
#  TRANSLATION FUNCTION (using NLLB)
# ==================================

def translate_nllb(text, tokenizer, model, source_language, target_language,device):
    tokenizer.src_lang = f"{source_language}_Latn" 
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    target_lang_token = f"{target_language}_Latn" 
    if target_lang_token not in tokenizer.get_vocab():
        print(f"[ERROR] Language token {target_lang_token} not found in the tokenizer vocab.")
        return None  

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang_token)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            forced_bos_token_id=forced_bos_token_id  
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


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
    print(" TRANSLATION COMPLETED SUCCESSFULLY!")


def main():
    parser = argparse.ArgumentParser(description="Translate text using NLLB model")
    parser.add_argument('--input_path', type=str, required=True, help="Input path for the data")
    parser.add_argument('--output_path', type=str, required=True, help="Output path for the full dataset (original + translated)")
    parser.add_argument('--source_language', type=str, default="eng", help="Source language (default is English)")
    parser.add_argument('--target_language', type=str, default="ita", help="Target language (default is Italian)")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)



    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"


    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer, model = load_model(model_name,device)


    data_list = load_data(args.input_path)

   
    for idx, ex in enumerate(data_list, start=1):
        src = ex.get("src", None)
        if not src:
            continue

        ex["mt"] = translate_nllb(src, tokenizer, model, args.source_language, args.target_language,device)

        if ex["mt"] is None:
            print(f"  [{idx}/{len(data_list)}] Translation failed for text: {src[:70]}...")
        else:
            print(f"  [{idx}/{len(data_list)}] Translated text: {ex['mt'][:70]}...")


    save_results(data_list, args.output_path)
    free_model(model, tokenizer)

if __name__ == "__main__":
    main()
