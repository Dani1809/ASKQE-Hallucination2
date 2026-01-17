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
            f"Prompt key '{prompt_key}' not found. Available keys: {list(prompts.keys())}"
        )

    prompt = prompts[prompt_key]
    if not isinstance(prompt, str) or len(prompt.strip()) == 0:
        raise ValueError(f"Prompt for key '{prompt_key}' is empty or not a string")

    return prompt


def answer_questions(tokenizer, model, prompt_template, sentence, questions, debug=False, ex_id=None, which=None):
    """
    Answer a list of questions given a sentence.
    Always returns a list of strings with the SAME length as questions.
    """
    questions_str = json.dumps(questions, ensure_ascii=False)

    prompt = (
        prompt_template
        .replace("{{sentence}}", sentence)
        .replace("{{questions}}", questions_str)
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    if debug:
        n_in = int(input_ids.shape[-1])
        print(f"[DEBUG][{ex_id}][{which}] #questions={len(questions)} input_tokens={n_in}")
        if n_in > 3800:  # warning threshold (rough)
            print(f"[DEBUG][{ex_id}][{which}] WARNING: very long prompt, may truncate generation context")

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                do_sample=False,         # ✅ deterministic for QA
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id
            )

        response = outputs[0][input_ids.shape[-1]:]
        text = tokenizer.decode(response, skip_special_tokens=True).strip()

        if debug:
            print(f"[DEBUG][{ex_id}][{which}] raw_model_text_len={len(text)}")
            print(f"[DEBUG][{ex_id}][{which}] raw_model_text_preview:\n{text[:300]}\n---")

        # remove wrapping quotes if any
        text_clean = text.strip().strip('"\'').strip()

        # try JSON parse
        try:
            answers = json.loads(text_clean)
        except json.JSONDecodeError as e:
            if debug:
                print(f"[DEBUG][{ex_id}][{which}] JSONDecodeError: {e}")
                print(f"[DEBUG][{ex_id}][{which}] text_clean_preview:\n{text_clean[:300]}\n---")
            answers = ["No Answer"] * len(questions)

        if not isinstance(answers, list):
            if debug:
                print(f"[DEBUG][{ex_id}][{which}] Parsed JSON is not a list: {type(answers)}")
            answers = ["No Answer"] * len(questions)

        # force 1–1 alignment
        if len(answers) != len(questions):
            if debug:
                print(f"[DEBUG][{ex_id}][{which}] MISMATCH: {len(questions)} questions vs {len(answers)} answers")
            answers = answers[:len(questions)]
            answers += ["No Answer"] * (len(questions) - len(answers))

        # final sanity
        answers = [a if isinstance(a, str) and a.strip() else "No Answer" for a in answers]

    except Exception as e:
        if debug:
            print(f"[DEBUG][{ex_id}][{which}] ERROR during generate/parse: {e}")
        answers = ["No Answer"] * len(questions)

    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--prompt_key", type=str, required=True)
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug prints")
    parser.add_argument("--max_examples", type=int, default=0, help="If >0, process only first N examples")
    args = parser.parse_args()

    prompt_template = load_prompt(args.prompt_path, args.prompt_key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Prompt key: {args.prompt_key}")
    if args.debug:
        print(f"[DEBUG] Prompt preview:\n{prompt_template[:300]}\n---")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto"
    ).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # load processed ids (resume)
    processed = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["id"])
                except:
                    pass
        print(f"[INFO] Resume mode: found {len(processed)} already processed IDs")

    n_done = 0
    n_skipped = 0

    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line_idx, line in enumerate(f_in, start=1):
            data = json.loads(line)
            ex_id = data.get("id")

            if not ex_id or ex_id in processed:
                n_skipped += 1
                continue

            raw_questions = data.get("questions_bt")
            if not raw_questions:
                if args.debug:
                    print(f"[DEBUG][{ex_id}] SKIP: no questions_bt")
                n_skipped += 1
                continue

            if isinstance(raw_questions, str):
                try:
                    questions = json.loads(raw_questions)
                except:
                    questions = [q.strip() for q in raw_questions.split("\n") if q.strip()]
            else:
                questions = list(raw_questions)

            # normalize questions
            questions = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
            if not questions:
                if args.debug:
                    print(f"[DEBUG][{ex_id}] SKIP: questions parsed empty")
                n_skipped += 1
                continue

            src = data.get("src")
            bt = data.get("bt")
            if not src or not bt:
                if args.debug:
                    print(f"[DEBUG][{ex_id}] SKIP: missing src or bt")
                n_skipped += 1
                continue

            print(f"[QA | SRC + BT] {ex_id}  (#q={len(questions)})")

            data["answers_src"] = answer_questions(
                tokenizer, model, prompt_template, src, questions,
                debug=args.debug, ex_id=ex_id, which="SRC"
            )
            data["answers_bt"] = answer_questions(
                tokenizer, model, prompt_template, bt, questions,
                debug=args.debug, ex_id=ex_id, which="BT"
            )

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            n_done += 1

            if args.max_examples and n_done >= args.max_examples:
                print(f"[INFO] Reached max_examples={args.max_examples}, stopping.")
                break

    print(f"✅ QA completed. done={n_done}, skipped={n_skipped}")


if __name__ == "__main__":
    main()
