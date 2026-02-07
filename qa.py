from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import argparse
import os
import re

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


# =========================
# LOAD PROMPT
# =========================
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

    prompt = prompts[prompt_key]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt '{prompt_key}' is empty or invalid")

    return prompt


# =========================
# ANSWER QUESTIONS
# =========================
def answer_questions(
    tokenizer,
    model,
    prompt_template,
    sentence,
    questions,
    debug=False,
    ex_id=None,
    which=None,
):
    """
    Answer a list of questions given a sentence.
    Always returns a list of answers with SAME length as questions.
    """

    # ---------- BUILD PROMPT ----------
    questions_json = json.dumps(questions, ensure_ascii=False)

    prompt = (
        prompt_template
        .replace("{{sentence}}", sentence)
        .replace("{{questions}}", questions_json)
    )

    if debug:
        print("\n" + "=" * 80)
        print(f"[DEBUG][{ex_id}][{which}] PROMPT:")
        print(prompt[:1000])
        print("---- END PROMPT ----")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    # ---------- TOKENIZE ----------
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    n_tokens = inputs["input_ids"].shape[-1]

    if debug:
        print(f"[DEBUG][{ex_id}][{which}] input_tokens={n_tokens}")
        if n_tokens > 4096:
            print(
                f"[DEBUG][{ex_id}][{which}] ⚠️ PROMPT TOO LONG "
                f"({n_tokens} tokens) — context may be truncated"
            )

    # ---------- GENERATE ----------
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        response = outputs[0][prompt_len:]

        text = tokenizer.decode(
            response,
            skip_special_tokens=True
        ).strip()

        if debug:
            print(f"\n[DEBUG][{ex_id}][{which}] RAW MODEL OUTPUT:")
            print(text)
            print("---- END RAW OUTPUT ----")

        # ---------- CLEAN OUTPUT ----------
        text_clean = text.strip()

        # remove code fences if present
        if text_clean.startswith("```"):
            text_clean = text_clean.split("\n", 1)[-1]
        if text_clean.endswith("```"):
            text_clean = text_clean.rsplit("```", 1)[0]

        text_clean = text_clean.strip()

        # normalize Python list → JSON list
        text_clean = text_clean.replace("'", '"')

        # ---------- PARSE ----------
        answers = None
        try:
            parsed = json.loads(text_clean)
            if isinstance(parsed, list):
                answers = parsed
            elif isinstance(parsed, dict):
                for key in ["answers", "output", "result"]:
                    if key in parsed and isinstance(parsed[key], list):
                        answers = parsed[key]
                        break
        except json.JSONDecodeError as e:
            if debug:
                print(
                    f"[DEBUG][{ex_id}][{which}] JSONDecodeError: {e}\n"
                    f"FAILED TEXT:\n{text_clean}\n"
                    "---- END FAILED JSON ----"
                )

        # ---------- FALLBACK ----------
        if answers is None:
            if debug:
                print(
                    f"[DEBUG][{ex_id}][{which}] FALLBACK → line-based parsing"
                )
            lines = [
                re.sub(r"^\d+[\)\.\-]\s*", "", ln).strip()
                for ln in text_clean.split("\n")
                if ln.strip()
            ]
            answers = lines

        # ---------- FORCE ALIGNMENT ----------
        if not isinstance(answers, list):
            answers = ["No Answer"] * len(questions)

        if len(answers) != len(questions):
            if debug:
                print(
                    f"[DEBUG][{ex_id}][{which}] "
                    f"MISMATCH: {len(questions)} questions vs {len(answers)} answers"
                )
            answers = answers[:len(questions)]
            answers += ["No Answer"] * (len(questions) - len(answers))

        answers = [
            a if isinstance(a, str) and a.strip() else "No Answer"
            for a in answers
        ]

    except Exception as e:
        if debug:
            print(f"[DEBUG][{ex_id}][{which}] ERROR during generation: {e}")
        answers = ["No Answer"] * len(questions)

    if debug:
        print(f"[DEBUG][{ex_id}][{which}] FINAL ANSWERS:")
        for i, a in enumerate(answers, start=1):
            print(f"  A{i}: {a}")

    return answers


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--prompt_key", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_examples", type=int, default=0)
    args = parser.parse_args()

    prompt_template = load_prompt(args.prompt_path, args.prompt_key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Prompt key: {args.prompt_key}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto",
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    processed_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except:
                    pass
        print(f"[INFO] Resume mode: {len(processed_ids)} IDs already processed")

    done = 0
    skipped = 0

    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line_idx, line in enumerate(f_in, start=1):
            data = json.loads(line)
            ex_id = data.get("id")

            if not ex_id or ex_id in processed_ids:
                skipped += 1
                continue

            questions = data.get("questions_src")
            if not questions:
                skipped += 1
                continue

            if isinstance(questions, str):
                try:
                    questions = json.loads(questions)
                except:
                    questions = [
                        q.strip()
                        for q in questions.split("\n")
                        if q.strip()
                    ]

            questions = [
                q.strip()
                for q in questions
                if isinstance(q, str) and q.strip()
            ]

            if not questions:
                skipped += 1
                continue

            src = data.get("src")
            bt = data.get("bt")
            if not src or not bt:
                skipped += 1
                continue

            print(f"\n[QA | SRC + BT] {ex_id} (#q={len(questions)})")

            data["answers_src"] = answer_questions(
                tokenizer, model, prompt_template,
                src, questions,
                debug=args.debug, ex_id=ex_id, which="SRC"
            )

            data["answers_bt"] = answer_questions(
                tokenizer, model, prompt_template,
                bt, questions,
                debug=args.debug, ex_id=ex_id, which="BT"
            )

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            done += 1

            if args.max_examples and done >= args.max_examples:
                print(f"[INFO] Reached max_examples={args.max_examples}")
                break

    print(f"\n✅ QA completed. done={done}, skipped={skipped}")


if __name__ == "__main__":
    main()
