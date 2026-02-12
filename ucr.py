import json
import argparse


def is_no_answer(x, no_answer="No Answer"):
    if not isinstance(x, str):
        return True
    t = x.strip()
    if not t:
        return True
    return t.casefold() == no_answer.strip().casefold()  # case-insensitive [web:277][web:280]


def ucr_hallucination(answers_src, answers_bt, no_answer="No Answer"):
    assert len(answers_src) == len(answers_bt), \
        f"Length mismatch: src={len(answers_src)}, bt={len(answers_bt)}"

    hallucinated = 0
    for a_src, a_bt in zip(answers_src, answers_bt):
        # prima: a_src.strip() == no_answer ...
        if is_no_answer(a_src, no_answer) and (not is_no_answer(a_bt, no_answer)):
            print("[HALLUCINATION]")
            print("SRC:", a_src)
            print("BT :", a_bt)
            print("-" * 40)
            hallucinated += 1

    return hallucinated / max(1, len(answers_src))


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
    args = parser.parse_args()

    data = load_data(args.input_path)

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

    ucr = ucr_hallucination(all_answers_src, all_answers_bt, no_answer=args.no_answer)
    print(f"\nUCR Hallucination Score: {ucr:.4f}")


if __name__ == "__main__":
    main()


