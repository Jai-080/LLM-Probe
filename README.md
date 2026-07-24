# LLM Probe

A project that detects **memorization** and **hallucination** in Large Language Models (LLMs) by training lightweight classifiers (probes) on internal activations. Hidden states are extracted directly from the model during generation, eliminating the need for external corpus lookup or fact-checking APIs.

## Methodology

1. **Activation Extraction**: Prompts and continuations are fed into the target model (e.g., `Phi-3-mini-4k-instruct` loaded in 4-bit quantization). Per-layer hidden states are extracted for the continuation tokens.
2. **Layer Selection (Ranking)**: Cohen's $d$ effect size is computed across layers to identify which transformer layers exhibit the most distinct activation patterns for the target behavior (memorization vs. hallucination).
3. **Probe Training**: Lightweight logistic regression classifiers are trained on the extracted activations of the selected layers.
4. **Token-by-Token Visualizer**: A Gradio web app renders generated tokens color-coded by their predicted probability of being memorized/hallucinated.

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
│   │   └── labeled/       # final labeled dataset (prompt, continuation, label, category)
│   └── hallucination/
│       └── labeled/       # TruthfulQA/HaluEval derived labeled dataset
├── scripts/
│   ├── build_memorization_dataset.py   # ROUGE-L verification pipeline
│   ├── build_hallucination_dataset.py  # model response generation and labeling
│   ├── train_memorization_probe.py     # full training and evaluation pipeline
│   └── train_hallucination_probe.py    # full training and evaluation pipeline
├── demo/
│   └── app.py             # Gradio interactive visualizer
├── results/
│   ├── memorization/      # saved probe (.pkl), metrics, plots
│   └── hallucination/
├── notebooks/
│   └── exploration.ipynb  # notebook for scratch testing
├── requirements.txt
└── README.md
```

## Setup & Verification

1. Create a local python virtual environment:
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
