import json, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as fin, \
         open(args.output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            out = {
                "id": ex["id"],
                "src": ex["en"],
                "mt": ex["es"],
                "perturbation": ex["perturbation"],
                "pert_mt": ex["pert_es"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
