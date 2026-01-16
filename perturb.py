import torch
import json
import os
import argparse
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# MODEL
# =========================
model_id = "Qwen/Qwen2.5-3B-Instruct"

# =========================
# PROMPTS
# =========================
PERTURBATION_PROMPTS = {
    "entity_injection": """
Sei un sistema che esegue perturbazioni testuali controllate.

Compito:
Aggiungi UNA nuova entità nominata (persona, organizzazione o luogo)
che NON è presente nel testo originale.
L'entità deve essere non supportata dal source.
NON rimuovere né sostituire alcun contenuto esistente.
Aggiungi l'entità in modo naturale.

Testo originale:
{{sentence}}

Produci SOLO il testo modificato, in italiano.
""",

    "numerical_injection": """
Sei un sistema che esegue perturbazioni testuali controllate.

Compito:
Aggiungi UN valore numerico (numero, percentuale, data o quantità)
che NON è presente nel testo originale.
NON modificare né sostituire numeri esistenti.
Aggiungi solo nuova informazione numerica plausibile.

Testo originale:
{{sentence}}

Produci SOLO il testo modificato, in italiano.
""",

    "over_specification": """
Sei un sistema che esegue perturbazioni testuali controllate.

Compito:
Aggiungi UN dettaglio fine-grained ma plausibile NON menzionato nel testo originale.
Il dettaglio può riguardare luogo, causa, modalità o contesto.
NON contraddire né rimuovere informazioni esistenti.

Testo originale:
{{sentence}}

Produci SOLO il testo modificato, in italiano.
"""
}

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--perturbation_type",
        type=str,
        required=True,
        choices=list(PERTURBATION_PROMPTS.keys())
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto"
    ).to(device)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    processed_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                processed_ids.add(json.loads(line)["id"])

    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)
            sent_id = data.get("id")

            already_done = any(
                k.startswith("pert_mt") and args.perturbation_type in k
                for k in data.keys()
            )
            if already_done:
                continue

            # 🔍 trova tutti i campi mt*
            mt_fields = [
                k for k in data.keys()
                if re.fullmatch(r"mt\d+", k)
            ]

            if not mt_fields:
                continue

            for mt_field in mt_fields:
                sentence = data.get(mt_field)
                if not sentence:
                    continue

                prompt = (
                    PERTURBATION_PROMPTS[args.perturbation_type]
                    .replace("{{sentence}}", sentence)
                )

                messages = [
                    {"role": "system", "content": "Sei un assistente utile."},
                    {"role": "user", "content": prompt}
                ]

                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=200,
                        eos_token_id=tokenizer.eos_token_id
                    )

                response = outputs[0][input_ids.shape[-1]:]
                generated_text = tokenizer.decode(
                    response,
                    skip_special_tokens=True
                ).strip()

                out_field = f"pert_{mt_field}"
                data[out_field] = generated_text

                print(f"[OK] {sent_id} | {out_field}")
                print(generated_text)
                print("-" * 80)

            # data["perturbation_type"] = args.perturbation_type
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
