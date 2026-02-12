import json
import argparse


def is_no_answer(x, no_answer="No Answer"):
    if not isinstance(x, str):
        return True
    t = x.strip()
    if not t:
        return True
    return t.casefold() == no_answer.strip().casefold()  # case-insensitive [web:277][web:280]


def ucr_hallucination(answers_src, answers_bt, no_answer="No Answer", debug=False):
    assert len(answers_src) == len(answers_bt), \
        f"Length mismatch: src={len(answers_src)}, bt={len(answers_bt)}"

    hallucinated = 0
    for a_src, a_bt in zip(answers_src, answers_bt):
        if is_no_answer(a_src, no_answer) and (not is_no_answer(a_bt, no_answer)):
            if debug:
                print("[HALLUCINATION]")
                print("SRC:", a_src)
                print("BT :", a_bt)
                print("-" * 40)
            hallucinated += 1

    return hallucinated / max(1, len(answers_src))


def ucr_sentence_level(data, no_answer="No Answer"):
    n_sent, n_flag = 0, 0

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

    return n_flag / max(1, n_sent)


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

    # ---- per-question (micro) ----
    all_answers_src = []
    all_answers_bt = []

    for item in data:
        answers_src = item.get("answers_src")
        answers_bt = item.get("answers_bt")

        if not answers_src or not answers_bt:
            continue

        assert len(answers_src) == len(answers_bt), \
            f"Mismatch in item {item.get('id')}"

        all_answers_src.extend(answers_src)
        all_answers_bt.extend(answers_bt)

    ucr_q = ucr_hallucination(
        all_answers_src,
        all_answers_bt,
        no_answer=args.no_answer,
        debug=args.debug
    )

    # ---- per-sentence (hit@k / OR) ----
    ucr_sent = ucr_sentence_level(data, no_answer=args.no_answer)

    print(f"\nUCR (per-question)  : {ucr_q:.4f}")
    print(f"UCR (per-sentence)  : {ucr_sent:.4f}")


if __name__ == "__main__":
    main()


