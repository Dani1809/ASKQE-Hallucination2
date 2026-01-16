# =========================
# ENV CLEANUP (NO WARNINGS)
# =========================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# IMPORTS
# =========================
import torch
import json
import argparse
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# MODEL
# =========================
# model_id = "Qwen/Qwen2.5-3B-Instruct"
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# =========================
# PROMPTS
# =========================
PERTURBATION_PROMPTS = {
    "entity_injection": """
You are a controlled text editing system.

TASK:
Identify the language of the provided sentence. 
Insert exactly ONE new named entity (person, organization, or place)
that is NOT mentioned in the original sentence.

CONSTRAINTS:
- Do NOT remove, rewrite, or replace any existing words.
- Insert the entity only by adding new tokens.
- Do NOT add more than one entity.
- Do NOT add explanations, comments, or notes.
- Keep the result as a single sentence.
- Integrate the entity naturally and grammatically.
- The output must differ from the input only by the insertion of the new entity.

ABSTRACT EXAMPLE (illustrative only):
Input:
"X observed an increase."

Output:
"X observed an increase in Y."

INPUT SENTENCE:
{{sentence}}

OUTPUT:
Return ONLY the modified sentence in the identified language.
""",

    "numerical_injection": """
You are a controlled text editing system.

TASK:
Identify the language of the input sentence.
Insert exactly ONE new numerical value.

LANGUAGE CONSTRAINT:
- The output language MUST be identical to the input language.
- Do NOT translate the sentence.

CONSTRAINTS:
- Insert the numerical value only by adding new tokens.
- Use a short numerical phrase (2–5 tokens).
- Do NOT remove, rewrite, or replace any existing words.
- Do NOT add explanations, comments, or notes.
- Do NOT add more than one numerical value.
- Do NOT add new clauses or sentences.
- Keep the result as a single sentence.
- The output must differ from the input only by minimal insertion.


INPUT SENTENCE:
{{sentence}}

OUTPUT:
Return ONLY the modified sentence.


""",

    "over_specification": """
You are a controlled text editing system.

TASK:
Insert exactly ONE manner-based over-specification that answers "how"
by adding a short descriptive construction.

DEFINITION:
The added detail must describe the manner or process of an existing action
using a minimal multi-token construction (2–4 tokens), without adding
new factual information.

LANGUAGE CONSTRAINT:
- The output language MUST be identical to the input language.
- Do NOT translate the sentence.

CONSTRAINTS:
- Add exactly ONE descriptive construction (2–4 tokens).
- The construction may include an adjective, an adverb, or a fixed manner phrase.
- Do NOT add numbers, dates, or named entities.
- Do NOT introduce causes, consequences, or explanations.
- Do NOT add parenthetical text or punctuation-based comments.
- Do NOT remove, rewrite, or replace any existing words.
- Do NOT add new clauses or sentences.
- Keep the result as a single sentence.
- The output must differ from the input only by the insertion of the construction.
- Integrate the construction naturally and grammatically.

ABSTRACT EXAMPLE (illustrative only):
Input:
"X increased."
Output:
"X increased in a gradual manner."

INPUT SENTENCE:
{{sentence}}

OUTPUT:
Return ONLY the modified sentence.

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


    if not os.path.isfile(args.input_path):
        print("[FATAL] Input file does not exist.")
        return

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[DEBUG] Output directory ready: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[DEBUG] pad_token was None → set to eos_token")

   
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()

    # =========================
    # PROCESS FILES
    # =========================
    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line_idx, line in enumerate(f_in, start=1):

            try:
                data = json.loads(line)
            except Exception as e:
                print(f"[ERROR] Line {line_idx} is not valid JSON: {e}")
                continue

          
            sent_id = data.get("id", f"line_{line_idx}")

            already_done = any(
                k.startswith("pert_mt") and args.perturbation_type in k
                for k in data.keys()
            )
            if already_done:
                print(f"[DEBUG] {sent_id} already perturbed → SKIP")
                continue

            sentence = data.get("mt")
            if not sentence:
                print(f"[WARNING] No 'mt' field found for {sent_id} → SKIP")
                continue

            mt_field = "mt"
            sentence = data.get(mt_field)

            if not sentence:
                print(f"[WARNING] Empty sentence in {mt_field} → SKIP")
                continue
            prompt = (
                PERTURBATION_PROMPTS[args.perturbation_type]
                .replace("{{sentence}}", sentence)
            )

            messages = [
                {"role": "system", "content": "Sei un assistente utile."},
                {"role": "user", "content": prompt}
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True
            ).to(device)

            print(f"[DEBUG] Input tokens shape: {inputs.shape}")

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    attention_mask=(inputs != tokenizer.pad_token_id),
                    max_new_tokens=200,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.25,
                    top_p=0.85
                )

                response = outputs[0][inputs.shape[-1]:]
                generated_text = tokenizer.decode(
                    response,
                    skip_special_tokens=True
                ).strip()

                out_field = f"pert_mt"
                data[out_field] = generated_text
  
                print(f"[OK] {sent_id} | {out_field}")
                print(generated_text)
                print("-" * 80)

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    main()
