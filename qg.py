from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import argparse
import os

#MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"



def load_prompt(prompt_path: str, prompt_key: str) -> str:
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    if prompt_key not in prompts:
        raise KeyError(
            f"Prompt key '{prompt_key}' not found. "
            f"Available keys: {list(prompts.keys())}"
        )

    return prompts[prompt_key]


def main():
    # =========================
    # ARGS
    # =========================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--prompt_key", type=str, required=True)
    args = parser.parse_args()

    # =========================
    # LOAD PROMPT
    # =========================
    PROMPT_TEMPLATE = load_prompt(args.prompt_path, args.prompt_key)
   
    # =========================
    # LOAD MODEL & TOKENIZER
    # =========================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto"
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # =========================
    # PROCESS DATASET
    # =========================
    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "w", encoding="utf-8") as f_out:

        for line_idx, line in enumerate(f_in, start=1):
            data = json.loads(line)

            src_field = "bt"
            sentence = data.get(src_field)



            if not sentence:
                print("[DEBUG] No 'bt' field found → saving empty questions")
                data["questions_bt"] = []
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            print(f"[DEBUG] Input sentence:\n{sentence}")

            # =========================
            # BUILD PROMPT
            # =========================
            prompt = PROMPT_TEMPLATE.replace("{{sentence}}", sentence)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

           # =========================
            # TOKENIZE (CHAT TEMPLATE)
            # =========================
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            # =========================
            # GENERATE
            # =========================
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=256,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )

            # =========================
            # DECODE
            # =========================
            prompt_len = inputs["input_ids"].shape[-1]
            response = outputs[0][prompt_len:]

            raw_output = tokenizer.decode(
                response,
                skip_special_tokens=True
            ).strip()

            raw_output = tokenizer.decode(
                response,
                skip_special_tokens=True
            ).strip()


            # =========================
            # CLEAN OUTPUT
            # =========================
            cleaned = raw_output.strip().strip('`').strip()

            if cleaned.startswith('python') or cleaned.startswith('json'):
                cleaned = cleaned.split('\n', 1)[-1].strip()
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3].strip()



            # =========================
            # PARSE OUTPUT
            # =========================
            questions = []

            try:
                questions = json.loads(cleaned)
                print(f"[DEBUG] json.loads OK → type={type(questions)}")

                if not isinstance(questions, list):
                    print("[DEBUG] Parsed object is not a list → forcing empty list")
                    questions = []

            except json.JSONDecodeError as e:
                print(f"[DEBUG] JSONDecodeError: {e}")
                questions = [
                    q.strip() for q in cleaned.split("\n")
                    if q.strip() and len(q.strip()) > 3
                ]
                print(f"[DEBUG] Fallback split produced {len(questions)} questions")

            print(f"[DEBUG] Final questions count: {len(questions)}")

            # =========================
            # SAVE (ALWAYS)
            # =========================
            data["questions_bt"] = questions
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

            print("[DEBUG] Record written to file")

    print("\nCOMPLETED — Output file written successfully.")


if __name__ == "__main__":
    main()


