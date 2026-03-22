# MCQ Generation Pipeline — VS Code Setup & Usage Guide

> **Paper:** "Psychometrically Robust MCQs: Integrating SBERT Distance Filtering and Multi-Source Distractors"  


---

## Project Structure

```
mcq_pipeline/
├── main.py                          # ← Start here: full end-to-end runner
├── phase1_document_processing.py    # PDF extraction, SBERT chunking, KeyBERT
├── phase2_question_answer_generation.py  # FLAN-T5 + grounding + DeBERTa QA
├── phase3_distractor_generation.py  # DistilBERT MLM + WordNet + T5 + KeyBERT
├── phase4_5_quality_psychometric.py # Quality metrics + CTT simulation
├── visualization.py                 # All paper figures (Fig 4, 5, 6 + more)
├── requirements.txt
└── README.md
```

---

## Step 1 — Environment Setup

### Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download spaCy language model (required once)

```bash
python -m spacy download en_core_web_sm
```

### Download NLTK data (required once)

```python
# Run in Python or add to your script:
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
```

---

## Step 2 — Models Downloaded Automatically

All transformer models are downloaded automatically from HuggingFace on first run.
**Required internet connection on first run (~5 GB total download):**

| Model | Size | Used in |
|---|---|---|
| `all-MiniLM-L6-v2` (SBERT) | 90 MB | Phases 1, 2, 3, 4 |
| `google/flan-t5-large` | 780 MB | Phase 2 — question generation |
| `deepset/deberta-v3-base-squad2` | 184 MB | Phase 2 — answer extraction |
| `distilbert-base-uncased` | 66 MB | Phase 3 — MLM distractors |
| `t5-small` | 60 MB | Phase 3 — paraphrase distractors |

---

## Step 3 — Running the Pipeline

### Option A: Demo mode (no PDF needed — built-in AI/ML text)

```bash
python main.py --demo
```

### Option B: Your own PDF

```bash
python main.py --pdf path/to/your_document.pdf
```

### Option C: Full options

```bash
python main.py \
  --pdf path/to/document.pdf \
  --output my_results/ \
  --theta 0.60 \
  --tau_ground 0.45 \
  --q_threshold 0.42
```

### Option D: Run individual phases for debugging

```bash
# Phase 1 only
python phase1_document_processing.py path/to/document.pdf

# Phase 2 only (with built-in dummy chunk)
python phase2_question_answer_generation.py

# Phase 3 only (with built-in dummy item)
python phase3_distractor_generation.py

# Phase 4+5 only (with dummy MCQ)
python phase4_5_quality_psychometric.py
```

---

## Step 4 — VS Code Launch Configuration

Create `.vscode/launch.json` in the project folder:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "MCQ Pipeline - Demo",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/main.py",
      "args": ["--demo", "--output", "demo_output"],
      "console": "integratedTerminal"
    },
    {
      "name": "MCQ Pipeline - PDF",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/main.py",
      "args": ["--pdf", "${input:pdfPath}", "--output", "results"],
      "console": "integratedTerminal"
    }
  ],
  "inputs": [
    {
      "id": "pdfPath",
      "type": "promptString",
      "description": "Path to the PDF file"
    }
  ]
}
```

---

## Key Parameters (Section 3.2 of paper)

| Parameter | Value | Meaning |
|---|---|---|
| `--theta` | 0.60 | SBERT chunking similarity threshold |
| `--tau_ground` | 0.45 | Grounding validation cutoff |
| `d_min / d_max` | 0.30 / 0.83 | Distractor distance band |
| `d_star` | 0.52 | Ideal distractor distance |
| `tau_div` | 0.85 | Max inter-distractor similarity |
| `--q_threshold` | 0.42 | Quality retention threshold |

---

## Expected Output

After running, the `--output` directory will contain:

```
mcq_output/
├── mcqs.json                        # Structured MCQ data
├── mcqs_readable.txt                # Human-readable MCQ bank
├── fig4_difficulty_discrimination.png
├── fig5_pointbiserial.png
├── fig6_correlation_matrix.png
├── bloom_distribution.png
└── quality_metrics.png
```

### Sample `mcqs.json` entry

```json
{
  "id": "MCQ-001",
  "bloom_level": "Understand",
  "question": "What does the self-attention mechanism allow a transformer to do?",
  "correct_answer": "weigh the importance of different words",
  "distractors": ["encode positional information", "apply dropout regularization", "stack residual layers"],
  "grounding_score": 0.71,
  "quality": {
    "pseudo_f1": 0.534,
    "novelty_index": 0.650,
    "distraction_index": 0.241,
    "plausibility": 0.424,
    "overall_q": 0.462,
    "label": "Good"
  },
  "psychometrics": {
    "p_value": 0.498,
    "d_value": 0.597,
    "r_pb": 0.447,
    "ctt_pass": true
  }
}
```

---

## Paper Results Benchmarks

Your output should approximate these targets (Table 11):

| Metric | Paper Result | Target Range |
|---|---|---|
| Mean P-value | 0.498 ± 0.066 | 0.30 – 0.70 ideal |
| % Ideal Difficulty | **94.2%** | ≥ 90% |
| Mean D-value | 0.597 ± 0.149 | ≥ 0.40 |
| % Strong Discrimination | **89.9%** | ≥ 85% |
| Mean r_pb | 0.447 ± 0.105 | ≥ 0.20 |
| % Valid Items | **98.6%** | ≥ 95% |
| % Functional Distractors | **96.7%** | ≥ 90% |

---

## Troubleshooting

**`spacy model not found`**  
→ Run: `python -m spacy download en_core_web_sm`

**`CUDA out of memory`**  
→ The pipeline auto-detects GPU/CPU. If OOM, set in each phase file: `device = "cpu"`

**`No MCQs passed quality threshold`**  
→ Lower `--q_threshold` to 0.35: `python main.py --demo --q_threshold 0.35`

**`Too few distractors`**  
→ Widen the distance band: edit `d_min=0.20, d_max=0.90` in `phase3_distractor_generation.py`

**First run is slow (~15 min)**  
→ Models are downloading from HuggingFace. Subsequent runs are fast (models cached in `~/.cache/huggingface/`).

---

## Hardware Requirements

| Setup | Time per MCQ | Recommended For |
|---|---|---|
| CPU only | ~4–8 min | Small documents (<10 pages) |
| GPU (4GB VRAM) | ~30–60 sec | Medium documents |
| GPU (8GB+ VRAM) | ~10–20 sec | Large documents (paper's 92 PDFs) |
