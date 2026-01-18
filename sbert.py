import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

def clean_answers(ans_list):
    """
    Pulisce la lista di risposte rimuovendo "No Answer" e stringhe vuote
    """
    if not ans_list:
        return []
    return [
        a.strip()
        for a in ans_list
        if a and isinstance(a, str) and a.strip() and a.strip().lower() != "no answer"
    ]


def compute_all_cosines(query_emb, answers, model):
    """
    Calcola cosine similarity tra query_emb e ogni risposta in answers.
    Ritorna una lista di similarità (una per ogni risposta).
    """
    if not answers:
        return []

    ans_embs = model.encode(answers, normalize_embeddings=True)
    sims = cosine_similarity(
        query_emb.reshape(1, -1),
        ans_embs
    )
    return sims[0].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading embedding model...")
    model = SentenceTransformer(args.embedding_model)

    results = []

    with open(args.input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            ex = json.loads(line)

            new = ex.get("new", "").strip()
            if not new:
                print(f"[WARN] Line {line_num}: campo 'new' vuoto, skipping")
                continue

            # Prendi le risposte dai campi corretti
            ans_bt_raw = ex.get("answers_bt", [])
            ans_src_raw = ex.get("answers_src", [])

            # Pulisci le risposte (rimuovi "No Answer")
            ans_bt = clean_answers(ans_bt_raw)
            ans_src = clean_answers(ans_src_raw)


            # Encoding del campo "new"
            new_emb = model.encode(new, normalize_embeddings=True)

            # Calcola cosine similarity per ogni risposta
            cos_bt_list = compute_all_cosines(new_emb, ans_bt, model)
            cos_src_list = compute_all_cosines(new_emb, ans_src, model)

            # Calcola gli score per ogni risposta
            # Dato che potrebbero essere di lunghezza diversa dopo il cleaning,
            # calcoliamo lo score per il numero massimo di risposte disponibili
            max_len = max(len(cos_bt_list), len(cos_src_list))
            
            hallucination_scores = []
            for i in range(max_len):
                bt_score = cos_bt_list[i] if i < len(cos_bt_list) else 0.0
                src_score = cos_src_list[i] if i < len(cos_src_list) else 0.0
                hallucination_scores.append(round(bt_score - src_score, 4))

            results.append({
                "id": ex.get("id"),
                "new": new,
                "answers_bt_cleaned": ans_bt,
                "answers_src_cleaned": ans_src,
                "cos_bt": [round(c, 4) for c in cos_bt_list],
                "cos_src": [round(c, 4) for c in cos_src_list],
                "hallucination_scores": hallucination_scores,
            })

    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[OK] Evaluation completed → {args.output_path}")
    print(f"[INFO] Processed {len(results)} examples")


if __name__ == "__main__":
    main()