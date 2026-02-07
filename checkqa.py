import json
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

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



def yesno_qa(tokenizer, model, prompt_template, sentence, question):
    prompt = (
        prompt_template
        .replace("{{sentence}}", sentence)
        .replace("{{question}}", question)
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
      outputs = model.generate(
          input_ids=inputs["input_ids"],
          attention_mask=inputs.get("attention_mask"),
          max_new_tokens=8,
          eos_token_id=tokenizer.eos_token_id,
          pad_token_id=tokenizer.pad_token_id,
          do_sample=False
      )

    response = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True).strip()

    # normalizzazione hard
    answer = answer.lower()
    if answer.startswith("yes"):
        return "Yes"
    if answer.startswith("no answer"):
        return "No Answer"
    if answer.startswith("no"):
        return "No"

    return "No Answer"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--prompt_key", type=str, required=True)
    args = parser.parse_args()

    # =========================
    # LOAD PROMPT
    # =========================
    prompt_template = load_prompt(args.prompt_path, args.prompt_key)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)

            src = data.get("src")
            contrastive_qs = data.get("contrastive_questions")

            if not src or not contrastive_qs:
                continue

            answers = []
            print(f"[YES/NO QA] {data.get('id')}")

            for q in contrastive_qs:
                ans = yesno_qa(
                    tokenizer, model, prompt_template, src, q
                )
                answers.append(ans)
                print(f"  Q: {q}")
                print(f"  A: {ans}")

            data["contrastive_answers_src"] = answers
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("✅ Contrastive YES/NO QA on SRC completed.")


if __name__ == "__main__":
    main()
