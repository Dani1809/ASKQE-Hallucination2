import json
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# =========================
# LOAD PROMPTS
# =========================
def load_prompt(path, key):
    with open(path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if key not in prompts:
        raise KeyError(f"Prompt key '{key}' not found")
    return prompts[key]

def generate_response(
    tokenizer,
    model,
    qa_prompt,
    text,
    question
):
    print(f"Generating response for question: {question}")
    
    prompt = (
        qa_prompt
        .replace("{{sentence}}", text)
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
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    print(f"Generated response: {response}")
    return response


# =========================
# GENERATE FOLLOW-UP QUESTION (SRC ONLY)
# =========================
def generate_followup_question(
    tokenizer,
    model,
    followup_prompt,
    text,
    prev_question,
    prev_answer
):
    print(f"Generating follow-up for: Q: {prev_question} A: {prev_answer}")
    
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

    print(f"Generated follow-up question: {q}")
    return q

# =========================
# MAIN
# =========================
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
    followup_prompt = load_prompt(args.prompt_path, "followup_prompt")

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
            answers_src = data.get("answers_src")
            entry_id = data.get("id")  # Get entry id

            if not src or not bt or not questions or not answers_src or not entry_id:
                continue

            multiturn = {}

            if args.debug:
                print(f"\n[MULTITURN] ID={data['id']}")

            # =========================
            # TURN 1: Generate follow-up questions for each question-answer pair
            # =========================
            current_questions = questions
            current_answers = answers_src
            turn_counter = 1
            question_counter = 1  # Initialize question counter for each entry

            while turn_counter <= args.max_turns:
                next_questions = []
                next_answers=[]
                multiturn[f"turn_{turn_counter}"] = []

                # For each question-answer pair, generate a follow-up question and its response
                for q, a in zip(current_questions, current_answers):
                    question_id = f"{entry_id}.{question_counter}"  # Unique ID for each question

    

                    # Generate the follow-up question for the answer
                    next_q = generate_followup_question(
                        tokenizer,
                        model,
                        followup_prompt,
                        text=src,
                        prev_question=q,
                        prev_answer=a
                    )
                    followup_response = generate_response(
                        tokenizer,
                        model,
                        qa_prompt,
                        text=src,
                        question=next_q
                    )

                    # Add current question-answer pair with the follow-up question and its response
                    if next_q.endswith("?"):
                        multiturn[f"turn_{turn_counter}"].append({
                            "question_id": question_id,  # Unique question ID
                            "question": q,
                            "answer_src": a,
                            "follow_up_question": next_q,
                            "follow_up_response": followup_response  # Same response from previous turn
                        })

                    if args.debug:
                        print(f" Turn {turn_counter}")
                        print(f"  Q: {q}")
                        print(f"  A_SRC: {a}")
                        print(f"  Follow-Up Q: {next_q}")
                        print(f"  Follow-Up Response: {followup_response}")

                    next_questions.append(next_q)  # Collect the generated questions for the next round
                    next_answers.append(followup_response)
                    question_counter += 1  # Increment question ID for the next question

                # Prepare for the next round (use the new follow-up questions)
                current_questions = next_questions
                current_answers = next_answers
                turn_counter += 1  # Increment the turn counter

            # Save the generated multiturn questions and answers
            data["multiturn"] = multiturn
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("✅ MULTI-TURN ASKQE COMPLETED")

if __name__ == "__main__":
    main()
