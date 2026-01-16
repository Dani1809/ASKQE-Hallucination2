import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# =========================
# PATH
# =========================
DATA_FILE = "/content/ASKQE-Hallucination/dataNoPerturb/qa_bt.jsonl"

# =========================
# SBERT
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# =========================
# MEAN POOLING
# =========================
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

# =========================
# PARAMS
# =========================
THRESHOLD = 0.75
total = 0
hallucinations = 0
similarities = []

# =========================
# MAIN LOOP
# =========================
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)

        src = ex.get("src", "").strip()
        if not src:
            continue

        # tutte le back-translations
        bt_keys = [k for k in ex if k.startswith("bt")]

        for bt_key in bt_keys:
            bt_text = ex[bt_key].strip()
            if not bt_text:
                continue

            enc_src = tokenizer(src, return_tensors="pt", truncation=True).to(device)
            enc_bt  = tokenizer(bt_text, return_tensors="pt", truncation=True).to(device)

            with torch.no_grad():
                out_src = model(**enc_src)
                out_bt  = model(**enc_bt)

            emb_src = F.normalize(
                mean_pooling(out_src, enc_src["attention_mask"]), p=2, dim=1
            )
            emb_bt = F.normalize(
                mean_pooling(out_bt, enc_bt["attention_mask"]), p=2, dim=1
            )

            sim = F.cosine_similarity(emb_src, emb_bt).item()
            similarities.append(sim)

            total += 1
            if sim < THRESHOLD:
                hallucinations += 1

# =========================
# RESULTS
# =========================
print("=" * 60)
print("SBERT SRC vs BACK-TRANSLATION")
print("=" * 60)
print(f"Total comparisons: {total}")
print(f"Average similarity: {sum(similarities)/len(similarities):.4f}")
print(f"Hallucinations (< {THRESHOLD}): {hallucinations}")
print(f"Hallucination rate: {hallucinations/total:.2%}")
