import json
import argparse


def is_no_answer(x, no_answer="No Answer"):
    if not isinstance(x, str):
        return True
    t = x.strip()
    if not t:
        return True
    return t.casefold() == no_answer.strip().casefold()  # case-insensitive [web:280]


def ucr_question_level_count(answers_src, answers_bt, no_answer="No Answer", debug=False):
    assert len(answers_src) == len(answers_bt), \
        f"Length mismatch: src={len(answers_src)}, bt={len(answers_bt)}"

    hallucinated = 0
    total = len(answers_src)

    for a_src, a_bt in zip(answers_src, answers_bt):
        if is_no_answer(a_src, no_answer) and (not is_no_answer(a_bt, no_answer)):
            hallucinated += 1
            if debug:
                print("[HALLUCINATION]")
                print("SRC:", a_src)
                print("BT :", a_bt)
                print("-" * 40)

    rate = hallucinated / max(1, total)
    return rate, hallucinated, total


def ucr_sentence_level_count(data, no_answer="No Answer"):
    n_sent = 0
    n_flag = 0

    for item in data:
        a_src = item.get("answers_src")
        a_bt = item.get("answers_bt")

        if not a_src or not a_bt or len(a_src) != len(a_bt):
            continue

        n_sent += 1
        flagged = any(
            is_no_answer(s, no_answer) and (not is_no_answer(b, no_answer))
            for s, b in zip(a_src, a_bt)
        )
        n_flag += int(flagged)

    rate = n_flag / max(1, n_sent)
    return rate, n_flag, n_sent


def load_data(path):
    print(f"[STEP] Loading dataset: {path}")
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    print(f"[OK] Loaded {len(data)} samples")
    return data


def main():
    parser = argparse.ArgumentParser(description="Compute UCR Hallucination score")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--no_answer", type=str, default="No Answer")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data = load_data(args.input_path)

    # Aggrega solo domande da item validi (stessa lunghezza)
    all_answers_src = []
    all_answers_bt = []

    n_items_total = len(data)
    n_items_valid = 0
    n_items_skipped = 0

    for item in data:
        a_src = item.get("answers_src")
        a_bt = item.get("answers_bt")

        if not a_src or not a_bt or len(a_src) != len(a_bt):
            n_items_skipped += 1
            continue

        n_items_valid += 1
        all_answers_src.extend(a_src)
        all_answers_bt.extend(a_bt)

    # ---- per-question (micro) ----
    ucr_q, q_flag, q_total = ucr_question_level_count(
        all_answers_src,
        all_answers_bt,
        no_answer=args.no_answer,
        debug=args.debug
    )

    # ---- per-sentence (OR) ----
    ucr_sent, s_flag, s_total = ucr_sentence_level_count(
        data,
        no_answer=args.no_answer
    )

    print("\n====== UCR RESULTS ======")
    print(f"Items total in file         : {n_items_total}")
    print(f"Items valid (used)          : {n_items_valid}")
    print(f"Items skipped               : {n_items_skipped}")
    print("--------------------------------")
    print(f"Questions flagged           : {q_flag}/{q_total}  ({ucr_q:.4f})")
    print(f"Sentences flagged           : {s_flag}/{s_total}  ({ucr_sent:.4f})")
    print("================================\n")


if __name__ == "__main__":
    main()




