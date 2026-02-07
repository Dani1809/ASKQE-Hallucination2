#!/usr/bin/env python3

from datasets import load_dataset
import spacy
import json
import random
from tqdm import tqdm
import re
import argparse


# =========================
# CLEANING UTILITIES
# =========================

HEADER_PATTERN = re.compile(
    r"(?:(?<=^)|(?<=[\.\n]))\s*[A-Za-z][A-Za-z\s\-]{0,20}\s*:\s*",
    flags=re.IGNORECASE
)

ATTACHED_HEADER_PATTERN = re.compile(
    r"\.(results|methods|conclusions?|objectives?)\b",
    flags=re.IGNORECASE
)

def cleaning(text: str, nlp):
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # fix attached headers
    text = ATTACHED_HEADER_PATTERN.sub(r". \1", text)

    # remove inline headers
    text = HEADER_PATTERN.sub("", text)

    # sentence splitting
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    # remove trailing standalone headers
    cleaned = []
    for s in sentences:
        t = s.lower().strip(" .:")
        if len(t.split()) <= 2 and t.isalpha():
            continue
        cleaned.append(s)

    return cleaned


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="pubmed_mt_askqe_2000.jsonl",
        help="Path to output JSONL file"
    )
    args = parser.parse_args()

    # =========================
    # FIXED CONFIG
    # =========================
    DATASET_NAME = "ccdv/pubmed-summarization"
    SPLIT = "train"
    MAX_ABSTRACTS = 5000
    MAX_CONTEXTS = 2000
    K = 3
    SEED = 42

    random.seed(SEED)

    # =========================
    # LOAD NLP
    # =========================
    nlp = spacy.load(
        "en_core_sci_sm",
        disable=["ner", "parser", "lemmatizer"]
    )
    nlp.add_pipe("sentencizer")

    # =========================
    # LOAD DATASET
    # =========================
    dataset = load_dataset(DATASET_NAME, split=SPLIT)

    # =========================
    # PROCESS ABSTRACTS
    # =========================
    records = []

    for doc_id in tqdm(range(MAX_ABSTRACTS), desc="Processing abstracts"):
        abstract = dataset[doc_id].get("abstract", "")
        if not abstract:
            continue

        sentences = cleaning(abstract, nlp)

        if len(sentences) < K:
            continue

        for i in range(0, len(sentences), K):
            chunk = sentences[i:i + K]
            if len(chunk) < K:
                continue

            records.append({
                "context": " ".join(chunk)
            })

    if len(records) < MAX_CONTEXTS:
        raise ValueError(
            f"Only {len(records)} contexts available, less than {MAX_CONTEXTS}"
        )

    # =========================
    # SAMPLE + ASSIGN IDS
    # =========================
    records = random.sample(records, MAX_CONTEXTS)

    final_records = []
    for i, r in enumerate(records, start=1):
        final_records.append({
            "id": i,
            "src": r["context"]
        })

    # =========================
    # SAVE
    # =========================
    with open(args.output, "w", encoding="utf-8") as fout:
        for r in final_records:
            fout.write(json.dumps(r) + "\n")

    print(f"Saved {len(final_records)} contexts to {args.output}")


if __name__ == "__main__":
    main()
