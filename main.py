"""
main.py — Full End-to-End MCQ Generation Pipeline
================================================
Reproduces the system from:
  "Psychometrically Robust MCQs: Integrating SBERT Distance Filtering
   and Multi-Source Distractors"

USAGE:
  python main.py --pdf path/to/document.pdf
  python main.py --pdf path/to/document.pdf --output results/
  python main.py --demo                        # Runs on built-in sample text

PIPELINE STAGES:
  Phase 1 → Document processing  (SBERT chunking, KeyBERT refinement)
  Phase 2 → Question & answer    (FLAN-T5-Large, SBERT grounding, DeBERTa)
  Phase 3 → Distractor gen.      (DistilBERT MLM, WordNet, T5 paraphrase, KeyBERT)
  Phase 4 → Quality evaluation   (Pseudo-F1, Novelty, Distraction Index, Q≥0.42)
  Phase 5 → Psychometric valid.  (CTT: P-value, D-value, r_pb)
"""

import argparse
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Dict


# ─────────────────────────────────────────────
# Setup check: print model load instructions
# ─────────────────────────────────────────────

SAMPLE_TEXT = """
Artificial intelligence (AI) is the simulation of human intelligence processes by machines,
especially computer systems. Specific applications of AI include expert systems,
natural language processing, speech recognition and machine vision.

Machine learning is a subset of artificial intelligence that provides systems the ability
to automatically learn and improve from experience without being explicitly programmed.
It focuses on developing computer programs that can access data and use it to learn for themselves.
The process begins with observations or data, such as examples, direct experience, or instruction.

Deep learning is part of a broader family of machine learning methods based on artificial
neural networks with representation learning. Learning can be supervised, semi-supervised
or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural
networks, convolutional neural networks and transformers have been applied to fields
including computer vision, speech recognition, natural language processing, and more.

Transformer architecture introduced the attention mechanism that allows the model to weigh
the importance of different words in a sentence when making predictions. The self-attention
mechanism computes attention scores between all pairs of tokens in a sequence, enabling
the model to capture long-range dependencies more effectively than recurrent neural networks.

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based
machine learning technique for natural language processing pre-training. It learns to
represent text by training on massive amounts of unlabeled text, predicting missing words
and understanding context from both directions simultaneously.
"""


def make_dummy_chunks_from_text(text: str, sbert_model) -> List[Dict]:
    """
    Converts raw text into pseudo-chunks (for --demo mode without PDF).
    Splits on double newline, encodes with SBERT.
    """
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if len(p.strip()) > 50]
    chunks = []
    for para in paragraphs:
        emb = sbert_model.encode(para, convert_to_numpy=True)
        # Simple keyword extraction using top words
        words = [w for w in para.lower().split() if len(w) > 5]
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:5]
        chunks.append({
            "text": para,
            "embedding": emb,
            "keywords": keywords
        })
    return chunks


def export_mcqs_json(mcqs: List[Dict], path: str):
    """Save final MCQs as a clean JSON file."""
    export = []
    for i, m in enumerate(mcqs):
        item = {
            "id": f"MCQ-{i+1:03d}",
            "bloom_level": m.get("bloom_level", ""),
            "question": m.get("question", ""),
            "correct_answer": m.get("answer", ""),
            "distractors": m.get("distractors", []),
            "grounding_score": m.get("grounding_score", 0),
            "quality": {
                "pseudo_f1": m.get("pseudo_f1", 0),
                "novelty_index": m.get("novelty_index", 0),
                "distraction_index": m.get("distraction_index", 0),
                "plausibility": m.get("plausibility", 0),
                "overall_q": m.get("quality_score", 0),
                "label": m.get("quality_label", "")
            },
            "psychometrics": {
                "p_value": m.get("p_value", 0),
                "d_value": m.get("d_value", 0),
                "r_pb": m.get("r_pb", 0),
                "ctt_pass": m.get("ctt_pass", False)
            }
        }
        # Remove non-serializable numpy types
        export.append(json.loads(json.dumps(item, default=lambda x: float(x) if hasattr(x, 'item') else str(x))))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"  Saved MCQs JSON: {path}")


def export_mcqs_readable(mcqs: List[Dict], path: str):
    """Save human-readable MCQ report."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("GENERATED MCQ BANK — Psychometrically Validated\n")
        f.write("=" * 70 + "\n\n")

        ctt_pass = sum(1 for m in mcqs if m.get("ctt_pass", False))
        f.write(f"Total MCQs: {len(mcqs)} | CTT Pass: {ctt_pass}\n\n")

        for i, m in enumerate(mcqs):
            f.write(f"─" * 60 + "\n")
            f.write(f"Item {i+1:>3}  |  Bloom: {m.get('bloom_level','N/A'):10}  |  "
                    f"Status: {'✓ CTT PASS' if m.get('ctt_pass') else '○ CTT INFO'}\n")
            f.write(f"\nQ: {m.get('question', 'N/A')}\n\n")

            distractors = m.get("distractors", [])
            options = [m.get("answer", "")] + distractors
            import random
            random.seed(i)
            random.shuffle(options)
            correct_letter = chr(65 + options.index(m.get("answer", "")))

            for j, opt in enumerate(options):
                marker = " ← CORRECT" if opt == m.get("answer", "") else ""
                f.write(f"  {chr(65+j)}. {opt}{marker}\n")

            f.write(f"\n  [Metrics] Q={m.get('quality_score',0):.3f} | "
                    f"P={m.get('p_value',0):.3f} | D={m.get('d_value',0):.3f} | "
                    f"r_pb={m.get('r_pb',0):.3f}\n\n")

    print(f"  Saved readable report: {path}")


def print_final_summary(mcqs: List[Dict], summary: Dict, elapsed: float):
    """Print the final summary table matching the paper's results."""
    print("\n" + "=" * 65)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Total MCQs generated          : {summary['n_items']}")
    print(f"  Pipeline time                 : {elapsed:.1f}s")
    print()
    print("  ── Psychometric Results (Table 11 equivalent) ──")
    print(f"  P-value mean ± SD             : {summary['p_mean']:.3f} ± {summary['p_sd']:.3f}")
    print(f"  % in ideal difficulty (0.30–0.70): {summary['pct_difficulty_ideal']:.1f}%  (paper: 94.2%)")
    print(f"  D-value mean ± SD             : {summary['d_mean']:.3f} ± {summary['d_sd']:.3f}")
    print(f"  % strong discrimination (≥0.40): {summary['pct_discrimination_strong']:.1f}%  (paper: 89.9%)")
    print(f"  r_pb mean ± SD                : {summary['rpb_mean']:.3f} ± {summary['rpb_sd']:.3f}")
    print(f"  % valid items (r_pb ≥ 0.20)   : {summary['pct_validity_pass']:.1f}%  (paper: 98.6%)")

    func_distractors = sum(1 for m in mcqs if m.get("num_distractors", 0) >= 3)
    print(f"  % functional distractors      : {func_distractors/max(len(mcqs),1)*100:.1f}%  (paper: 96.7%)")
    print("=" * 65)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MCQ Generation Pipeline (SBERT + FLAN-T5 + DeBERTa + DistilBERT)"
    )
    parser.add_argument("--pdf", type=str, default=None,
                        help="Path to input PDF file")
    parser.add_argument("--demo", action="store_true",
                        help="Run on built-in sample text (no PDF needed)")
    parser.add_argument("--output", type=str, default="mcq_output",
                        help="Output directory for results")
    parser.add_argument("--theta", type=float, default=0.60,
                        help="SBERT chunking threshold (default: 0.60)")
    parser.add_argument("--tau_ground", type=float, default=0.45,
                        help="Grounding validation threshold (default: 0.45)")
    parser.add_argument("--q_threshold", type=float, default=0.42,
                        help="Quality retention threshold (default: 0.42)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()

    if not args.demo and args.pdf is None:
        parser.print_help()
        print("\n  ERROR: Provide --pdf <path> or use --demo\n")
        return

    os.makedirs(args.output, exist_ok=True)
    start_time = time.time()

    print("\n" + "=" * 65)
    print("  MCQ Generation Pipeline")
    print("  Based on: 'Psychometrically Robust MCQs: Integrating SBERT")
    print("   Distance Filtering and Multi-Source Distractors'")
    print("=" * 65 + "\n")

    # ── Phase 1 ──
    from sentence_transformers import SentenceTransformer

    if args.demo:
        print("  [DEMO MODE] Using built-in sample text\n")
        print("=" * 60)
        print("PHASE 1: Document Processing")
        print("=" * 60)
        print("  Loading SBERT model...")
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        chunks = make_dummy_chunks_from_text(SAMPLE_TEXT, sbert_model)
        print(f"  ✓ Phase 1 complete. {len(chunks)} chunks created.\n")
    else:
        from phase1_document_processing import run_phase1
        chunks, sbert_model = run_phase1(
            args.pdf,
            theta_sim=args.theta,
            verbose=True
        )

    if not chunks:
        print("  ERROR: No chunks produced. Check the PDF or input text.")
        return

    # ── Phase 2 ──
    from phase2_question_answer_generation import run_phase2
    qa_items = run_phase2(chunks, sbert_model, tau_ground=args.tau_ground, verbose=True)

    if not qa_items:
        print("  ERROR: No QA items produced after grounding/answer filtering.")
        return

    # ── Phase 3 ──
    from phase3_distractor_generation import run_phase3
    mcq_items = run_phase3(qa_items, sbert_model, verbose=True)

    # ── Phase 4 ──
    from phase4_5_quality_psychometric import run_phase4, run_phase5, bloom_breakdown
    retained_mcqs = run_phase4(mcq_items, sbert_model, q_threshold=args.q_threshold, verbose=True)

    if not retained_mcqs:
        print("  WARNING: No MCQs passed quality threshold. Try lowering --q_threshold.")
        retained_mcqs = mcq_items  # Fall back to all items

    # ── Phase 5 ──
    final_mcqs, summary = run_phase5(retained_mcqs, n_students=50, verbose=True)
    bloom_breakdown(final_mcqs)

    # ── Export ──
    elapsed = time.time() - start_time
    print_final_summary(final_mcqs, summary, elapsed)

    json_path = os.path.join(args.output, "mcqs.json")
    txt_path = os.path.join(args.output, "mcqs_readable.txt")
    export_mcqs_json(final_mcqs, json_path)
    export_mcqs_readable(final_mcqs, txt_path)

    # ── Plots ──
    if not args.no_plots and len(final_mcqs) >= 3:
        from visualization import generate_all_plots
        generate_all_plots(final_mcqs, output_dir=args.output)

    print(f"\n  All outputs saved to: {os.path.abspath(args.output)}/")
    print("  Files:")
    for f in os.listdir(args.output):
        print(f"    • {f}")
    print()


if __name__ == "__main__":
    main()
