import json
import argparse


def norm_str(x):
    return x.strip().casefold() if isinstance(x, str) else ""  # case-insensitive [web:280]


def compute_metrics(input_path, dump_path=None, include_context=False, max_dump=None,
                          yes_token="yes", no_token="no", print_k=0):
    yes_token = yes_token.casefold()
    no_token = no_token.casefold()

    # Per-question (micro) su subset BT==YES
    total_q_gated = 0
    src_yes = 0
    src_no = 0

    # Per-sentence (macro/OR) su subset con >=1 gated Q
    total_sent_gated = 0
    sent_with_src_no = 0

    # Diagnostics
    total_sent_valid = 0
    sent_with_any_gate = 0

    dumped = 0
    printed = 0

    dump_f = open(dump_path, "w", encoding="utf-8") if dump_path else None

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            qs = data.get("contrastive_questions", [])
            a_src = data.get("contrastive_answers_src", [])
            a_bt = data.get("contrastive_answers_bt", [])

            if not (isinstance(qs, list) and isinstance(a_src, list) and isinstance(a_bt, list)):
                continue
            if not qs or not a_src or not a_bt:
                continue

            n = min(len(qs), len(a_src), len(a_bt))
            qs = qs[:n]
            a_src = a_src[:n]
            a_bt = a_bt[:n]

            total_sent_valid += 1

            # Gate: tieni solo le domande dove BT dice YES
            gated_idx = [i for i, bt_ans in enumerate(a_bt) if norm_str(bt_ans) == yes_token]
            if not gated_idx:
                continue

            sent_with_any_gate += 1
            total_sent_gated += 1

            has_src_no = False

            for i in gated_idx:
                total_q_gated += 1
                src_ans_norm = norm_str(a_src[i])

                if src_ans_norm == no_token:
                    src_no += 1
                    has_src_no = True

                    # Dump/print caso "NO solo su SRC" (dato che BT è YES per costruzione)
                    rec = {
                        "id": data.get("id"),
                        "i": i,
                        "q": qs[i],
                        "bt_answer": a_bt[i],
                        "src_answer": a_src[i],
                    }
                    if include_context:
                        for k in ["src", "mt", "perturbation", "pert_mt", "bt"]:
                            if k in data:
                                rec[k] = data[k]

                    if dump_f is not None:
                        if (max_dump is None) or (dumped < max_dump):
                            dump_f.write(json.dumps(rec, ensure_ascii=False) + "\n")  # JSONL [web:426]
                            dumped += 1

                    if print_k and printed < print_k:
                        print("\n[SRC_NO_BT_YES]")
                        print("ID :", rec["id"])
                        print("Q  :", rec["q"])
                        print("BT :", rec["bt_answer"])
                        print("SRC:", rec["src_answer"])
                        printed += 1

                else:
                    # assumiamo YES nel tuo setting
                    src_yes += 1

            sent_with_src_no += int(has_src_no)

    if dump_f is not None:
        dump_f.close()

    chr_gated_question = (src_no / total_q_gated) if total_q_gated > 0 else 0.0
    chr_gated_sentence = (sent_with_src_no / total_sent_gated) if total_sent_gated > 0 else 0.0

    return {
        "total_sent_valid": total_sent_valid,
        "sent_with_any_gate": sent_with_any_gate,
        "total_sent_gated": total_sent_gated,
        "total_q_gated": total_q_gated,
        "src_yes": src_yes,
        "src_no": src_no,
        "CHR_gated_question": chr_gated_question,   # P(SRC==NO | BT==YES)
        "CHR_gated_sentence": chr_gated_sentence,   # % frasi con >=1 gated Q che diventa NO su SRC
        "dumped": dumped,
        "dump_path": dump_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--dump_path", type=str, default=None,
                        help="If set, writes JSONL of cases where BT==YES and SRC==NO.")
    parser.add_argument("--include_context", action="store_true",
                        help="Include src/bt/pert_mt fields in dumped records.")
    parser.add_argument("--max_dump", type=int, default=None,
                        help="Max dumped mismatch records (None = all).")
    parser.add_argument("--print_k", type=int, default=0,
                        help="Print first K mismatches to stdout.")
    args = parser.parse_args()

    m = compute_metrics(
        args.input_path,
        dump_path=args.dump_path,
        include_context=args.include_context,
        max_dump=args.max_dump,
        print_k=args.print_k
    )

    print("\n====== CONTRASTIVE (GATED BY BT==YES) ======")
    print(f"Sentences (valid fields)              : {m['total_sent_valid']}")
    print(f"Sentences with >=1 BT==YES question   : {m['sent_with_any_gate']}")
    print(f"Sentences in gated denominator        : {m['total_sent_gated']}")
    print(f"Total gated questions (BT==YES)       : {m['total_q_gated']}")
    print(f"SRC answers YES (on gated)            : {m['src_yes']}")
    print(f"SRC answers NO  (on gated)            : {m['src_no']}")
    print("-------------------------------------------")
    print(f"CHR_gated_question = P(SRC==NO|BT==YES): {m['CHR_gated_question']:.4f}")
    print(f"CHR_gated_sentence (OR over questions): {m['CHR_gated_sentence']:.4f}")
    if m["dump_path"]:
        print(f"[DUMP] Wrote {m['dumped']} records -> {m['dump_path']}")
    print("===========================================\n")


if __name__ == "__main__":
    main()




