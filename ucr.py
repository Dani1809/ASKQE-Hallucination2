import json

def ucr_hallucination(answers_src, answers_bt, no_answer="No Answer"):
    assert len(answers_src) == len(answers_bt)

    hallucinated = 0
    for a_src, a_bt in zip(answers_src, answers_bt):
        if a_src.strip() == no_answer and a_bt.strip() != no_answer: 
            print(a_src)
            print(a_bt)
            hallucinated += 1

    return hallucinated / max(1, len(answers_src))


INPUT_PATH_src = "/content/ASKQE-Hallucination/data/qa_src.jsonl"  
INPUT_PATH_bt = "/content/ASKQE-Hallucination/data/qa_bt.jsonl"   


def load_data(path):
    print(f"[STEP] Loading dataset: {path}")
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            print(f"[OK] JSON list with {len(data)} samples")
            return data
        else:
            data = [json.loads(line) for line in f if line.strip()]
            print(f"[OK] JSONL with {len(data)} samples")
            return data

src = load_data(INPUT_PATH_src)
bt = load_data(INPUT_PATH_bt)

# Aggregate all answers into flat lists
all_answers_src_bt1 = []
all_answers_bt1 = []

for src_item, bt_item in zip(src, bt):
    # Extract and parse answers_src_bt1
    if "answers_src_bt1" in src_item and src_item["answers_src_bt1"]:
        # The value is a list containing a JSON string of a list
        json_list_str = src_item["answers_src_bt1"][0]
        try:
            parsed_list = json.loads(json_list_str)
            all_answers_src_bt1.extend(parsed_list)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse answers_src_bt1 for example ID {src_item.get('id', 'N/A')}")

    # Extract and parse answers_bt1
    if "answers_bt1" in bt_item and bt_item["answers_bt1"]:
        # The value is a list containing a JSON string of a list
        json_list_str = bt_item["answers_bt1"][0]
        try:
            parsed_list = json.loads(json_list_str)
            all_answers_bt1.extend(parsed_list)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse answers_bt1 for example ID {bt_item.get('id', 'N/A')}")


ucr_bt1 = ucr_hallucination(
    all_answers_src_bt1,
    all_answers_bt1
)
print(ucr_bt1)
