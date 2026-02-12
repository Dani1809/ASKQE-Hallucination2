# sample_jsonl.py
import json, random, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]

    random.seed(args.seed)
    idxs = random.sample(range(len(lines)), k=min(args.n, len(lines)))  # senza replacement [web:146]

    with open(args.output_path, "w", encoding="utf-8") as out:
        for i in idxs:
            out.write(lines[i])

if __name__ == "__main__":
    main()
