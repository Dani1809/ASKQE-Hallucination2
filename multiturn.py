import json
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


# =========================================================
# LOAD PROMPTS
# =========================================================
def load_prompt(path, key):
    with open(path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if key not in prompts:
        raise KeyError(f"Prompt key '{key}' not found")
    return prompts[key]


# =========================================================
# ANSWER ONE QUESTION (SRC or BT)
# =========================================================
def answer_one(tokenizer, model, qa_prompt, sentence, question):
    prompt = (
        qa_prompt
        .replace("{{sentence}}", sentence)
        .replace("{{questions}}", json.dumps([question], ensure_ascii=False))
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
    except:
        pass

    return "No Answer"


# =========================================================
# GENERATE FOLLOW-UP QUESTION (SRC ONLY)
# =========================================================
def generate_followup_question(
    tokenizer,
    model,
    followup_prompt,
    text,
    prev_question,
    prev_answer
):
    prompt = (
        followup_prompt
        .replace("{{text}}", text)
        .replace("{{prev_question}}", prev_question)
        .replace("{{prev_answer}}", prev_answer)
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id
        )

    q = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return q


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # ---- load prompts
    qa_prompt = load_prompt(args.prompt_path, "qa_prompt")
    followup_prompt = load_prompt(args.prompt_path, "qg_prompt")

    # ---- model
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
         open(args.output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)

            src = data.get("src")
            bt = data.get("bt")
            questions = data.get("questions_src")

            if not src or not bt or not questions:
                continue

            q = questions[0]
            multiturn = []

            if args.debug:
                print(f"\n[MULTITURN] ID={data['id']}")

            for turn in range(1, args.max_turns + 1):

                a_src = answer_one(tokenizer, model, qa_prompt, src, q)
                a_bt  = answer_one(tokenizer, model, qa_prompt, bt,  q)

                multiturn.append({
                    "turn": turn,
                    "question": q,
                    "answer_src": a_src,
                    "answer_bt": a_bt
                })

                if args.debug:
                    print(f" Turn {turn}")
                    print(f"  Q: {q}")
                    print(f"  A_SRC: {a_src}")
                    print(f"  A_BT : {a_bt}")

                # STOP CONDITIONS
                if a_src == "No Answer":
                    break

                next_q = generate_followup_question(
                    tokenizer,
                    model,
                    followup_prompt,
                    text=src,
                    prev_question=q,
                    prev_answer=a_src
                )

                if not next_q.endswith("?"):
                    break

                q = next_q

            data["multiturn"] = multiturn
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("✅ MULTI-TURN ASKQE COMPLETED")


if __name__ == "__main__":
    main()
