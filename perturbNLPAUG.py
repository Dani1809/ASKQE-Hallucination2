import json
import os
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import argparse

# =========================
# ENV CLEANUP (NO WARNINGS)
# =========================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--perturbation_type",
        type=str,
        required=True,
        choices=["entity_injection", "numerical_injection", "over_specification"]
    )
    args = parser.parse_args()

    # =========================
    # NLPaug - Augmenters
    # =========================
    if args.perturbation_type == "entity_injection":
        aug = naw.EntitiesAugmenter()  # Perturbation for entity injection
    elif args.perturbation_type == "numerical_injection":
        aug = naw.SynonymAug(aug_p=0.1)  # Perturbation for numerical injection
    elif args.perturbation_type == "over_specification":
        aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="insert")  # Perturbation for over-specification

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # =========================
    # PROCESS FILES
    # =========================
    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)
            sent_id = data.get("id")

            # Skip already perturbed
            already_done = any(
                k.startswith("pert_mt") and args.perturbation_type in k
                for k in data.keys()
            )
            if already_done:
                continue

            # Look for 'mt' field (only one 'mt' field in the data)
            mt_field = "mt"  # Looking for a single 'mt' field
            sentence = data.get(mt_field)
            if not sentence:
                continue

            # Apply the perturbation
            augmented_sentence = aug.augment(sentence)

            # Save the augmented sentence in the dictionary
            out_field = f"pert_{mt_field}"
            data[out_field] = augmented_sentence

            print(f"[OK] {sent_id} | {out_field}")
            print(augmented_sentence)
            print("-" * 80)

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
