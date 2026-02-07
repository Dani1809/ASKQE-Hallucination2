from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import argparse
import os

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


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
    # LOAD MODEL
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # =========================
    # PROCESS
    # =========================
    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)

            questions = data.get("questions_src")
            answers = data.get("answers_bt")

            if not questions or not answers:
                continue

            assert len(questions) == len(answers), \
                f"Mismatch questions/answers in {data.get('id')}"

            all_contrastive = []

            for q, a in zip(questions, answers):
                if not a or a.strip() == "No Answer":
                    continue

                # =========================
                # BUILD PROMPT
                # =========================
                prompt = (
                    PROMPT_TEMPLATE
                    .replace("{{question}}", q)
                    .replace("{{answer}}", a)
                )

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]

                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(
                    prompt_text,
                    return_tensors="pt"
                ).to(model.device)

                # =========================
                # GENERATE
                # =========================
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=96,
                        eos_token_id=tokenizer.eos_token_id
                    )

                response = outputs[0][inputs["input_ids"].shape[-1]:]
                raw_output = tokenizer.decode(
                    response,
                    skip_special_tokens=True
                ).strip()

                # =========================
                # PARSE OUTPUT
                # =========================
                cleaned = raw_output.strip().strip('`').strip()

                if cleaned.startswith('json') or cleaned.startswith('python'):
                    cleaned = cleaned.split('\n', 1)[-1].strip()
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3].strip()

                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, list):
                        all_contrastive.extend(parsed)
                except:
                    # fallback: estrai linee con '?'
                    for line in cleaned.split('\n'):
                        if '?' in line:
                            all_contrastive.append(line.strip())

            # =========================
            # SAVE
            # =========================
            data["contrastive_questions"] = list(dict.fromkeys(all_contrastive))
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

            print(f"[OK] {data.get('id')} → {len(all_contrastive)} contrastive questions")

    print("✅ Contrastive YES/NO generation completed.")


if __name__ == "__main__":
    main()
