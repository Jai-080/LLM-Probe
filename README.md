# LLM Probe

An Activation-Based Detection of Memorization and Hallucination in Large Language Models.

This project detects **memorization** (verbatim text reproduction) and **hallucination** (factually incorrect/unsupported generations) in Large Language Models (LLMs) by training lightweight classifiers (probes) on internal activation vectors. Hidden states are extracted directly from the target model during generation, eliminating the need for external database queries or fact-checking APIs.

---

## Methodology

1. **Activation Extraction**: Prompts and generated continuations are fed into the target model (`Phi-3-mini-4k-instruct` loaded in 4-bit quantization). Per-layer hidden states are extracted specifically for the continuation/response tokens.
2. **Layer Selection (Ranking)**: Cohen's $d$ effect size is computed across all 32 layers to identify which transformer layers exhibit the most distinct activation patterns for the target behavior.
3. **Probe Training**: Lightweight Logistic Regression classifiers (probes) are trained on the concatenated, scaled activations of the top 3 selected layers.
4. **Token-by-Token Visualizer**: A Gradio web app renders generated tokens color-coded by their predicted probability of being memorized or hallucinated.

---

## Project Structure

```text
llm_probe/
├── config.py              # model name, device, layer count constants
├── models/
│   └── loader.py          # load_model() and load_tokenizer() functions
├── core/
│   ├── activations.py     # extract_activations() for hidden states extraction
│   ├── layer_selection.py # cohens_d() and rank_layers() for layer ranking
│   └── probe.py           # train_probe() and evaluate_probe()
├── data/
│   ├── memorization/
│   │   ├── raw/           # raw source texts by category
│   │   └── labeled/       # final labeled dataset
│   └── hallucination/
│       └── labeled/       # TruthfulQA derived cleaned dataset (dataset.parquet)
├── scripts/
│   ├── data_collection/
│   │   ├── fetch_wiki_leads.py              # Wikipedia lead paragraph scraper
│   │   ├── fetch_book_openings.py           # Project Gutenberg openings fetcher
│   │   ├── fetch_code_snippets.py           # GitHub code snippets downloader
│   │   ├── fetch_more_snippets.py           # Supplemental code snippets fetcher
│   │   ├── merge_supplemental_snippets.py   # Code snippets merger
│   │   ├── migrate_txt_to_jsonl.py          # Format migration utility
│   │   └── extract_failed_wiki_topics.py    # Wikipedia retry scraper
│   ├── probe_1_memorization/
│   │   ├── build_memorization_dataset.py   # Response generation and ROUGE-L labeling
│   │   ├── review_borderline.py             # Re-score and filter borderline cases
│   │   ├── summarize_manual_review.py       # Final merge of manual decisions
│   │   ├── train_memorization_probe.py      # Layer selection and probe training
│   │   ├── verify_teacher_forcing.py        # Teacher forcing validation checks
│   │   ├── test_demo_prompts.py             # Live demo probe verification
│   │   └── rank_middle_layers.py            # Middle-layer activation explorer
│   └── probe_2_hallucination/
│       ├── build_hallucination_dataset.py   # Response generation and ROUGE-L recall labeling
│       ├── review_hallucination_labels.py   # Boilerplate stripping & Entity Guard cleanup
│       ├── export_reconsidered_for_review.py# Export reconsidered rows to CSV
│       ├── apply_reconsidered_decisions.py  # Validate and merge reconsidered rows
│       ├── export_newly_resolved_for_review.py # Export resolved ambiguous rows
│       ├── apply_newly_resolved_decisions.py  # Validate and merge resolved rows
│       ├── export_needs_manual_review_for_review.py # Export remaining ambiguous rows
│       ├── apply_needs_manual_review_decisions.py  # Validate and merge remaining rows
│       └── train_hallucination_probe.py     # Layer selection, training, & generalization tests
├── demo/
│   └── app.py             # Gradio interactive visualizer
├── results/
│   ├── memorization/      # Probe weights (.joblib), metrics report, rankings plot
│   └── hallucination/     # Probe weights (.joblib), metrics report, rankings plot
├── requirements.txt
└── README.md
```

---

## Setup & Verification

1. Create a local Python virtual environment:
   ```bash
   python -m venv llm_probe
   ```
2. Activate and install dependencies:
   ```bash
   .\llm_probe\Scripts\pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
   ```
3. Run the verification script:
   ```bash
   .\llm_probe\Scripts\python.exe -m llm_probe.verify_setup
   ```

---

## Reproducing & Training Probe 1 (Memorization)

1. **Scrape Wikipedia Leads**:
   ```bash
   .\llm_probe\Scripts\python.exe scripts/data_collection/fetch_wiki_leads.py
   ```
2. **Fetch Book & Code Openings**: Scrape Project Gutenberg book openings and GitHub code snippets using the collection utilities in `scripts/data_collection/`.
3. **Build & Label Dataset**: Query the model, score responses using ROUGE-L, and save to Parquet:
   ```bash
   .\llm_probe\Scripts\python.exe scripts/probe_1_memorization/build_memorization_dataset.py
   ```
4. **Train Memorization Probe**: Run activation extraction, layer selection, and probe training:
   ```bash
   .\llm_probe\Scripts\python.exe scripts/probe_1_memorization/train_memorization_probe.py
   ```

---

## Reproducing & Training Probe 2 (Hallucination)

1. **Build Generation Dataset**: Run inference on TruthfulQA and auto-label with ROUGE-L Recall:
   ```bash
   .\llm_probe\Scripts\python.exe scripts/probe_2_hallucination/build_hallucination_dataset.py
   ```
2. **Run Post-Processing Cleanup**: Strip boilerplate question prefixes and run the Entity-Based Guard to flag suspect labels:
   ```bash
   .\llm_probe\Scripts\python.exe scripts/probe_2_hallucination/review_hallucination_labels.py
   ```
3. **Review Flagged Subsets**: Run the export scripts, fill in final decisions in the generated CSVs (`grounded`/`hallucinated`/`exclude`), and merge them back to the primary dataset:
   ```bash
   # Reconsidered Rows (41)
   .\llm_probe\Scripts\python.exe scripts/probe_2_hallucination/export_reconsidered_for_review.py
   .\llm_probe\Scripts\python.exe scripts/probe_2_hallucination/apply_reconsidered_decisions.py

   # Newly Resolved Rows (91)
   .\llm_probe\Scripts\python.exe scripts/probe_2_hallucination/export_newly_resolved_for_review.py
   .\llm_probe\Scripts\python.exe scripts/probe_2_hallucination/apply_newly_resolved_decisions.py

   # Remaining Ambiguous Rows (268)
   .\llm_probe\Scripts\python.exe scripts/probe_2_hallucination/export_needs_manual_review_for_review.py
   .\llm_probe\Scripts\python.exe scripts/probe_2_hallucination/apply_needs_manual_review_decisions.py
   ```
4. **Train Hallucination Probe**: Perform layer selection (via absolute Cohen's $d$) and train standard and MLP probes on the final 813 clean examples:
   ```bash
   .\llm_probe\Scripts\python.exe scripts/probe_2_hallucination/train_hallucination_probe.py
   ```

---

## Results & Findings

### Probe 1: Memorization
*   **Best Representation**: `mean_pooled`
*   **Selected Probe Layers**: `[32, 1, 17]`
*   **In-Domain F1 Score**: `0.985` (Logistic Regression) / `0.988` (MLP)
*   **Out-of-Distribution Generalization F1**: `0.925` (trained on Wikipedia/Books, tested on code snippets)

### Probe 2: Hallucination
*   **Best Representation**: `mean_pooled`
*   **Selected Probe Layers**: `[32, 15, 17]`
*   **In-Domain F1 Score**: `0.636` (Logistic Regression) / `0.681` (MLP)
*   **Category Generalization F1 (Held-out)**:
    *   **Misconceptions (Held-out)**: F1 = `0.436` (Accuracy: `0.560`, Precision: `0.619`)
    *   **Law (Held-out)**: F1 = `0.597` (Accuracy: `0.516`, Precision: `0.882`)
