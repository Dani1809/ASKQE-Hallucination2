import json
import argparse


def norm_str(x):
    return x.strip().casefold() if isinstance(x, str) else ""  # case-insensitive [web:277][web:280]


def compute_metrics(input_path, yes_token="yes", no_token="no"):
    # per-question (micro)
    total_q = 0
    yes_count = 0
    no_count = 0

    # per-sentence (macro / OR)
    total_sent = 0
    sent_with_no = 0

    yes_token = yes_token.casefold()
    no_token = no_token.casefold()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            questions = data.get("contrastive_questions", [])
            answers = data.get("contrastive_answers_src", [])

            if not isinstance(questions, list) or not isinstance(answers, list):
                continue
            if not questions or not answers:
                continue

            n = min(len(questions), len(answers))
            answers = answers[:n]

            total_sent += 1
            has_no = False

            for a in answers:
                total_q += 1
                a_norm = norm_str(a)

                if a_norm == yes_token:
                    yes_count += 1
                else:
                    # assumiamo che tutto il resto sia "no"
                    no_count += 1
                    has_no = True

            sent_with_no += int(has_no)

    chr_question = (no_count / total_q) if total_q > 0 else 0.0
    chr_sentence = (sent_with_no / total_sent) if total_sent > 0 else 0.0

    return {
        "total_questions": total_q,
        "yes": yes_count,
        "no": no_count,
        "total_sentences": total_sent,
        "sentences_with_no": sent_with_no,
        "CHR_question": chr_question,
        "CHR_sentence": chr_sentence,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to JSONL with contrastive_questions and contrastive_answers_src")
    args = parser.parse_args()

    m = compute_metrics(args.input_path)

    print("\n====== CONTRASTIVE HALLUCINATION METRICS ======")
    print(f"Total contrastive questions : {m['total_questions']}")
    print(f"Yes                        : {m['yes']}")
    print(f"No                         : {m['no']}")
    print("---------------------------------------------")
    print(f"Total sentences            : {m['total_sentences']}")
    print(f"Sentences with >=1 'No'    : {m['sentences_with_no']}")
    print("---------------------------------------------")
    print(f"CHR_question (No / All Q)  : {m['CHR_question']:.4f}")
    print(f"CHR_sentence (sent OR)     : {m['CHR_sentence']:.4f}")
    print("=============================================\n")


if __name__ == "__main__":
    main()


