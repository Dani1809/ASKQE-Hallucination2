# AskQE-Hallucination
## Inverted QA-Based Framework for Hallucination Detection in Machine Translation

### Description

AskQE-Hallucination inverts the standard AskQE paradigm to explicitly detect hallucinations introduced during machine translation. While the original AskQE framework generates questions from the source text and evaluates whether information is preserved in the backtranslation, this extension generates questions from the backtranslation itself and validates whether the queried information is supported by the source.

---

## Core Idea

The framework detects hallucinations through three complementary approaches:

1. UCR (Unanswerable Content Rate): Generate questions from the backtranslation and answer them using both source and backtranslation. If the backtranslation can answer but the source cannot (returns "No Answer"), the information must have been added during translation.

2. BERTScore Validation: For answer pairs where both source and backtranslation provide responses, compute semantic similarity. Low scores indicate semantic divergence likely caused by hallucinations.

3. Yes/No Verification: Generate binary verification questions that decompose backtranslation answers into more specific factual claims. Ask the source to verify each claim. A "No" answer from the source to a claim supported by the backtranslation unambiguously identifies hallucinated content.

---

## Dataset

We evaluate on 50 sentences from the ContraTICo dataset with expansion_impact perturbations. Source language is English and target language is Spanish. The perturbation type adds modifiers, qualifiers, or new information.

Example perturbation:
Source: "We have the opportunity to escalate the data extraction to twice weekly if needed."
Perturbed MT: "Tenemos la oportunidad de aumentar la extracción de datos al doble por semana en caso de ser necesario y reducir costos."
Backtranslation: "We have the opportunity to double data extraction per week if necessary and reduce costs."

---

## Models Used

Backtranslation: facebook/nllb-200-distilled-600M
Question Generation and QA: Qwen/Qwen2.5-7B-Instruct
Semantic Similarity: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

---

## Project Structure

ASKQE-Hallucination2/
├── data/
│   ├── expansion_impact.jsonl           (original ContraTICo data)
│   ├── contratico_expansion.jsonl       (sampled 50 sentences)
│   ├── contratico_expansion_bt.jsonl    (backtranslated)
│   ├── contratico_expansion_bt_qg.jsonl (with questions)
│   ├── contratico_expansion_bt_qg_qa.jsonl (with answers)
│   ├── contratico_expansion_bt_qg_qa_checkqg.jsonl (with verification questions)
│   ├── contratico_expansion_bt_qg_qa_checkqg_checkqa.jsonl (with verification answers)
│   ├── mismatches.jsonl                 (BERTScore flagged pairs)
│   └── contrastive_src_no_bt_yes.jsonl  (Yes/No detections)
├── sampleSentences.py       (sample N sentences from dataset)
├── normalize_contratico.py  (normalize ContraTICo format)
├── backtranslate.py        (backtranslation with NLLB)
├── qg.py                   (question generation from BT)
├── qa.py                   (answer questions on SRC and BT)
├── checkqg.py              (generate Yes/No verification questions)
├── checkqa.py              (answer verification questions)
├── ucr.py                  (compute UCR metrics)
├── simScores.py            (compute BERTScore)
├── compute_contrastive_metrics.py (aggregate Yes/No results)
├── prompt.json             (prompt templates)
├── data.json               (configuration)
└── README.md

---

## Usage

Clone the repository:
git clone https://github.com/Dani1809/ASKQE-Hallucination2.git
cd ASKQE-Hallucination2

Step 1 - Sample 50 sentences from ContraTICo:
python sampleSentences.py --input_path data/expansion_impact.jsonl --output_path data/expansion_impact_50.jsonl --n 50

Step 2 - Normalize ContraTICo format:
python normalize_contratico.py --input_path data/expansion_impact_50.jsonl --output_path data/contratico_expansion.jsonl

Step 3 - Backtranslation:
python backtranslate.py --input_path data/contratico_expansion.jsonl --output_path data/contratico_expansion_bt.jsonl --source_language spa --target_language eng

Step 4 - Question Generation from Backtranslation:
python qg.py --input_path data/contratico_expansion_bt.jsonl --output_path data/contratico_expansion_bt_qg.jsonl --prompt_path prompt.json --prompt_key qg_prompt

Step 5 - Question Answering (answer each question using both source and backtranslation):
python qa.py --input_path data/contratico_expansion_bt_qg.jsonl --output_path data/contratico_expansion_bt_qg_qa.jsonl --prompt_path prompt.json --prompt_key qa_prompt

The QA model is explicitly instructed to return "No Answer" when information is absent.

Step 6 - Generate Yes/No Verification Questions:
python checkqg.py --input_path data/contratico_expansion_bt_qg_qa.jsonl --output_path data/contratico_expansion_bt_qg_qa_checkqg.jsonl --prompt_path prompt.json --prompt_key qg_prompt_check

Step 7 - Answer Verification Questions on Source:
python checkqa.py --input_path data/contratico_expansion_bt_qg_qa_checkqg.jsonl --output_path data/contratico_expansion_bt_qg_qa_checkqg_checkqa.jsonl --prompt_path prompt.json --prompt_key qa_prompt_check

---

## Evaluation

Step 8 - Compute UCR:
python ucr.py --input_path data/contratico_expansion_bt_qg_qa.jsonl

Output: Questions flagged: 20/165 (12.12%), Sentences flagged: 16/50 (32.00%)

Step 9 - Compute BERTScore and flag mismatches:
python simScores.py --input_path data/contratico_expansion_bt_qg_qa.jsonl --output_path data/mismatches.jsonl --tau 0.6 --only_mismatches --include_context

Output: Pairs with similarity < 0.6 saved to mismatches.jsonl for manual inspection.

Step 10 - Aggregate Yes/No Verification Results:
python compute_contrastive_metrics.py --input_path data/contratico_expansion_bt_qg_qa_checkqg_checkqa.jsonl --dump_path data/contrastive_src_no_bt_yes.jsonl

Output: Sentences with SRC=No and BT=Yes (hallucinations detected). Detection rate: >50% (27/48 sentences).

---

## Results

On 50 ContraTICo sentences with expansion_impact perturbation:

Method | Questions Detected | Sentences Detected
UCR | 20/165 (12.12%) | 16/50 (32.00%)
BERTScore < 0.6 | — | (flagged for inspection)
Yes/No Verification | 37/162 (22.84%) | 27/48 (56.25%)

The Yes/No verification approach achieves the highest detection rate by targeting atomic facts, making it effective even for subtle semantic shifts.

---

## Example

Source: "The fever started two days ago."
Perturbed MT: "La fiebre alta empezó hace dos días."
Backtranslation: "The high fever started two days ago."

Question from BT: "What is the symptom mentioned?"
BT Answer: "High fever"
Source Answer: "The fever"

Verification Question: "Is the symptom high fever?"
BT Answer: "Yes"
Source Answer: "No"

Result: Hallucination detected - The modifier "high" is not supported by the source.

---

## Contribution

This extension provides a complementary perspective to standard AskQE:
Standard AskQE measures information preservation (source → BT)
AskQE-Hallucination measures information addition (BT → source validation)

Together, they enable comprehensive quality estimation covering both omission and hallucination errors.
