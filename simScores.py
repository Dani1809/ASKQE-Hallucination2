# answer_similarity_mismatches.py
import json
import argparse
import os
import numpy as np
from sentence_transformers import SentenceTransformer


def is_no_answer(x, no_answer="No Answer"):
    if not isinstance(x, str):
        return True
    t = x.strip()
    if not t:
        return True
    return t.casefold() == no_answer.strip().casefold()


def safe_get_list(ex, key):
    v = ex.get(key, [])
    return v if isinstance(v, list) else []


def main():
    ap = argparse.ArgumentParser(description="Compare answers_src vs answers_bt with SBERT cosine similarity.")
    ap.add_argument("--input_path", required=True, help="JSONL post-QA with questions_src, answers_src, answers_bt.")
    ap.add_argument("--output_path", required=True, help="JSONL output with mismatches under threshold.")
    ap.add_argument(
        "--embedding_model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-Transformers model id."
    )
    ap.add_argument("--tau", type=float, default=0.75, help="If sim < tau => mismatch.")
    ap.add_argument("--no_answer", type=str, default="No Answer")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--only_mismatches", action="store_true", help="Write only examples that contain >=1 mismatch.")
    ap.add_argument("--include_context", action="store_true",
                    help="Also include src/mt/pert_mt/bt fields in output records for manual inspection.")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)

    n_examples_in = 0
    n_examples_written = 0
    n_examples_valid = 0

    n_pairs_total = 0          # includes No Answer pairs
    n_pairs_compared = 0       # pairs where both sides answered
    all_sims = []

    with open(args.input_path, "r", encoding="utf-8") as fin, \
         open(args.output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_examples_in += 1
            ex = json.loads(line)

            ex_id = ex.get("id")
            questions = safe_get_list(ex, "questions_bt")
            answers_src = safe_get_list(ex, "answers_src")
            answers_bt = safe_get_list(ex, "answers_bt")

            # serve allineamento 1-1 per indice domanda
            if not questions or not answers_src or not answers_bt:
                continue
            if not (len(questions) == len(answers_src) == len(answers_bt)):
                continue

            n_examples_valid += 1

            # Costruisci coppie confrontabili (stessa domanda i)
            pairs = []
            for i, (q, a_s, a_b) in enumerate(zip(questions, answers_src, answers_bt)):
                n_pairs_total += 1
                if is_no_answer(a_s, args.no_answer) or is_no_answer(a_b, args.no_answer):
                    continue
                if not isinstance(q, str):
                    q = str(q)
                pairs.append({
                    "i": i,
                    "q": q.strip(),
                    "src": a_s.strip(),
                    "bt": a_b.strip()
                })

            # Se non ci sono coppie "both-answered", scrivi comunque (opzionale)
            if not pairs:
                if not args.only_mismatches:
                    out = {
                        "id": ex_id,
                        "n_q": len(questions),
                        "n_pairs_compared": 0,
                        "tau": args.tau,
                        "mismatches": []
                    }
                    if args.include_context:
                        for k in ["src", "mt", "perturbation", "pert_mt", "bt"]:
                            if k in ex:
                                out[k] = ex[k]
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    n_examples_written += 1
                continue

            # Embed e cosine similarity (dot su embedding normalizzati) [web:227][web:231]
            src_texts = [p["src"] for p in pairs]
            bt_texts = [p["bt"] for p in pairs]

            emb_src = model.encode(src_texts, normalize_embeddings=True, batch_size=args.batch_size)
            emb_bt = model.encode(bt_texts, normalize_embeddings=True, batch_size=args.batch_size)

            sims = (emb_src * emb_bt).sum(axis=1).astype(float).tolist()

            n_pairs_compared += len(sims)
            all_sims.extend(sims)

            mismatches = []
            for p, sim in zip(pairs, sims):
                if sim < args.tau:
                    mismatches.append({
                        "i": p["i"],
                        "sim": round(float(sim), 4),
                        "q": p["q"],
                        "src": p["src"],
                        "bt": p["bt"],
                    })

            if args.only_mismatches and not mismatches:
                continue

            out = {
                "id": ex_id,
                "n_q": len(questions),
                "n_pairs_compared": len(sims),
                "mean_sim": round(float(np.mean(sims)), 4),
                "min_sim": round(float(np.min(sims)), 4),
                "tau": args.tau,
                "n_mismatches": len(mismatches),
                "mismatches": mismatches
            }
            if args.include_context:
                for k in ["src", "mt", "perturbation", "pert_mt", "bt"]:
                    if k in ex:
                        out[k] = ex[k]

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_examples_written += 1

    mean_sim_global = float(np.mean(all_sims)) if all_sims else 0.0
    disagree_rate = float(np.mean([s < args.tau for s in all_sims])) if all_sims else 0.0

    print(f"[INFO] Read examples          : {n_examples_in}")
    print(f"[INFO] Valid examples         : {n_examples_valid}")
    print(f"[INFO] Written examples       : {n_examples_written}")
    print(f"[INFO] Pairs total (incl NA)  : {n_pairs_total}")
    print(f"[INFO] Pairs compared         : {n_pairs_compared}")
    print(f"[SIM]  Mean sim (global)      : {mean_sim_global:.4f}")
    print(f"[SIM]  Disagree rate (sim<tau): {disagree_rate:.4f}")
    print(f"[OK] Output -> {args.output_path}")


if __name__ == "__main__":
    main()
