"""
Phase 4: Quality Evaluation
  - Pseudo-F1, Novelty Index, Distraction Index, Plausibility Score
  - Overall Quality Score Q = (PF1 + NI + DI + PL) / 4
  - Threshold: Q ≥ 0.42 to retain

Phase 5: Psychometric Validation (CTT)
  - Simulated P-value (item difficulty), D-value (discrimination), r_pb (point-biserial)
  - Bloom-level breakdown
  - Ablation study reporting
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import pointbiserialr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# Phase 4: Quality Metrics
# ─────────────────────────────────────────────

def pseudo_f1(
    answer_emb: np.ndarray,
    distractor_embs: List[np.ndarray]
) -> float:
    """
    Table 2 – Pseudo-F1:
      PF1 = (1/n) Σ [1 - cos(E(a), E(d_i))]
    Measures semantic distance between answer and distractors (higher = more distinct).
    """
    if not distractor_embs:
        return 0.0
    distances = []
    for d_emb in distractor_embs:
        sim = cosine_similarity(answer_emb.reshape(1, -1), d_emb.reshape(1, -1))[0, 0]
        distances.append(1.0 - float(sim))
    return float(np.mean(distances))


def novelty_index(distractor_embs: List[np.ndarray]) -> float:
    """
    Table 2 – Novelty Index:
      NI = (2 / n(n-1)) Σ_{i<j} [1 - cos(E(d_i), E(d_j))]
    Measures pairwise diversity among distractors.
    """
    n = len(distractor_embs)
    if n < 2:
        return 0.0
    pairwise_distances = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(
                distractor_embs[i].reshape(1, -1),
                distractor_embs[j].reshape(1, -1)
            )[0, 0]
            pairwise_distances.append(1.0 - float(sim))
    return float(np.mean(pairwise_distances))


def distraction_index(
    answer_emb: np.ndarray,
    chunk_emb: np.ndarray,
    distractor_embs: List[np.ndarray]
) -> float:
    """
    Table 2 – Distraction Index:
      DI = (1/n) Σ cos(E(d_i), E(c)) · [1 - cos(E(d_i), E(a))]
    Balances contextual plausibility with semantic separation from answer.
    """
    if not distractor_embs:
        return 0.0
    scores = []
    for d_emb in distractor_embs:
        sim_context = float(cosine_similarity(d_emb.reshape(1, -1), chunk_emb.reshape(1, -1))[0, 0])
        sim_answer = float(cosine_similarity(d_emb.reshape(1, -1), answer_emb.reshape(1, -1))[0, 0])
        scores.append(sim_context * (1.0 - sim_answer))
    return float(np.mean(scores))


def plausibility_score(distractors: List[str], answer: str) -> float:
    """
    Table 2 – Plausibility Score (PL): heuristic linguistic naturalness check.
    Simulates expert rating [0,5] normalized to [0,1].
    Checks: non-empty, similar length, no special chars, not identical to answer.
    """
    if not distractors:
        return 0.0

    ans_words = len(answer.split())
    scores = []
    for d in distractors:
        p = 0.0
        if len(d) > 2:
            p += 1.5
        d_words = len(d.split())
        if abs(d_words - ans_words) <= 3:
            p += 1.5
        if d.replace(" ", "").isalpha():
            p += 1.0
        if d.lower() != answer.lower():
            p += 1.0
        scores.append(min(p / 5.0, 1.0))

    return float(np.mean(scores))


def overall_quality_score(pf1: float, ni: float, di: float, pl: float) -> float:
    """
    Table 2 – Overall Quality: Q = (PF1 + NI + DI + PL) / 4
    """
    return (pf1 + ni + di + pl) / 4.0


def classify_quality(q: float) -> str:
    """Table 1 – Thresholds: Excellent ≥0.50, Good 0.42–0.49, Acceptable 0.35–0.41, Poor <0.35"""
    if q >= 0.50:
        return "Excellent"
    elif q >= 0.42:
        return "Good"
    elif q >= 0.35:
        return "Acceptable"
    else:
        return "Poor"


def evaluate_mcq_quality(
    mcq: Dict,
    sbert_model: SentenceTransformer,
    q_threshold: float = 0.42
) -> Dict:
    """
    Compute all 4 quality metrics for a single MCQ and decide retention.
    Returns enriched dict with metrics appended.
    """
    answer = mcq["answer"]
    chunk_text = mcq["chunk_text"]
    distractors = mcq.get("distractors", [])

    if not distractors:
        return {**mcq, "pseudo_f1": 0.0, "novelty_index": 0.0,
                "distraction_index": 0.0, "plausibility": 0.0,
                "quality_score": 0.0, "quality_label": "Poor", "retained": False}

    # Encode
    answer_emb = sbert_model.encode(answer, convert_to_numpy=True)
    chunk_emb = sbert_model.encode(chunk_text, convert_to_numpy=True)
    dist_embs = sbert_model.encode(distractors, convert_to_numpy=True)
    dist_emb_list = [dist_embs[i] for i in range(len(distractors))]

    # Compute metrics
    pf1 = pseudo_f1(answer_emb, dist_emb_list)
    ni = novelty_index(dist_emb_list)
    di = distraction_index(answer_emb, chunk_emb, dist_emb_list)
    pl = plausibility_score(distractors, answer)
    q = overall_quality_score(pf1, ni, di, pl)

    return {
        **mcq,
        "pseudo_f1": round(pf1, 4),
        "novelty_index": round(ni, 4),
        "distraction_index": round(di, 4),
        "plausibility": round(pl, 4),
        "quality_score": round(q, 4),
        "quality_label": classify_quality(q),
        "retained": q >= q_threshold
    }


def run_phase4(
    mcq_items: List[Dict],
    sbert_model: SentenceTransformer,
    q_threshold: float = 0.42,
    verbose: bool = True
) -> List[Dict]:
    """
    Phase 4: Evaluate and filter MCQs by quality score Q ≥ 0.42.
    """
    print("=" * 60)
    print("PHASE 4: Quality Evaluation")
    print("=" * 60)

    evaluated = []
    for idx, mcq in enumerate(mcq_items):
        scored = evaluate_mcq_quality(mcq, sbert_model, q_threshold)
        evaluated.append(scored)

    retained = [m for m in evaluated if m["retained"]]
    rejected = [m for m in evaluated if not m["retained"]]

    if verbose:
        print(f"\n  {'ID':>4}  {'PF1':>6}  {'NI':>6}  {'DI':>6}  {'PL':>6}  {'Q':>6}  {'Label':12}  Status")
        print("  " + "-" * 72)
        for idx, m in enumerate(evaluated):
            status = "RETAIN" if m["retained"] else "REJECT"
            print(f"  {idx+1:>4}  {m['pseudo_f1']:>6.3f}  {m['novelty_index']:>6.3f}  "
                  f"{m['distraction_index']:>6.3f}  {m['plausibility']:>6.3f}  "
                  f"{m['quality_score']:>6.3f}  {m['quality_label']:12}  {status}")

    print(f"\n  ── Phase 4 Summary ──")
    print(f"     Items evaluated : {len(evaluated)}")
    print(f"     Items retained  : {len(retained)} ({len(retained)/max(len(evaluated),1)*100:.1f}%)")
    print(f"     Items rejected  : {len(rejected)}")

    if retained:
        qs = [m["quality_score"] for m in retained]
        print(f"     Mean Q (retained): {np.mean(qs):.3f} ± {np.std(qs):.3f}")

    print("\n  ✓ Phase 4 complete.\n")
    return retained


# ─────────────────────────────────────────────
# Phase 5: Psychometric Validation (CTT)
# ─────────────────────────────────────────────

def simulate_learner_responses(
    mcq: Dict,
    n_students: int = 50,
    ability_mean: float = 0.5,
    ability_std: float = 0.15,
    seed: int = None
) -> np.ndarray:
    """
    Section 3.3: Rasch-type simulation.
      - θ ~ TruncNormal(0.5, 0.15²)
      - Item difficulty b_j = 1 - Q_j
      - P(correct) = logistic(θ - b_j), adjusted for distractor plausibility
    Returns binary response array of shape (n_students,).
    """
    rng = np.random.default_rng(seed)
    ability = rng.normal(ability_mean, ability_std, n_students)
    ability = np.clip(ability, 0.01, 0.99)

    quality = mcq.get("quality_score", 0.45)
    b_j = 1.0 - quality  # Item difficulty parameter

    # Logistic model
    p_correct = 1.0 / (1.0 + np.exp(-(ability - b_j) * 3.0))

    # Plausibility adjustment: plausible distractors reduce P(correct) slightly
    pl = mcq.get("plausibility", 0.4)
    p_correct = p_correct * (1.0 - 0.15 * pl)
    p_correct = np.clip(p_correct, 0.05, 0.95)

    responses = rng.binomial(1, p_correct, n_students)
    return responses


def compute_ctt_metrics(responses: np.ndarray) -> Dict:
    """
    Classical Test Theory (Section 2.6):
      - P-value = C_i / N (proportion correct)
      - D-value = X̄_top27 − X̄_bottom27 (discrimination)
      - r_pb = point-biserial correlation with total score
    """
    n = len(responses)
    p_value = float(np.mean(responses))

    # D-value (top/bottom 27% groups)
    upper_cut = int(np.ceil(0.27 * n))
    lower_cut = int(np.ceil(0.27 * n))
    sorted_responses = np.sort(responses)[::-1]
    top_group = sorted_responses[:upper_cut]
    bottom_group = sorted_responses[n - lower_cut:]
    d_value = float(np.mean(top_group) - np.mean(bottom_group))

    # Point-biserial correlation (item vs. total)
    # For single item, use item score vs item-as-total proxy
    if np.std(responses) == 0:
        r_pb = 0.0
    else:
        # Simulate a test score: item score + noise representing other items
        rng = np.random.default_rng(42)
        total_score = responses + rng.binomial(1, 0.5, n) * 4
        try:
            r_pb, _ = pointbiserialr(responses, total_score)
            r_pb = float(r_pb)
        except Exception:
            r_pb = 0.0

    return {
        "p_value": round(p_value, 4),
        "d_value": round(max(d_value, 0.0), 4),
        "r_pb": round(max(r_pb, 0.0), 4)
    }


def ctt_filter(ctt: Dict) -> bool:
    """Standard CTT thresholds (Section 2.6)."""
    return (
        0.30 <= ctt["p_value"] <= 0.70 and  # Ideal difficulty range
        ctt["d_value"] >= 0.40 and            # Strong discrimination
        ctt["r_pb"] >= 0.20                   # Construct validity
    )


def run_phase5(
    retained_mcqs: List[Dict],
    n_students: int = 50,
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Phase 5: Simulate student responses and compute CTT metrics for each MCQ.
    Returns (final_mcqs, summary_stats).
    """
    print("=" * 60)
    print("PHASE 5: Psychometric Validation (CTT)")
    print("=" * 60)

    final_mcqs = []
    p_values, d_values, r_pbs = [], [], []

    for idx, mcq in enumerate(retained_mcqs):
        responses = simulate_learner_responses(mcq, n_students=n_students, seed=idx)
        ctt = compute_ctt_metrics(responses)

        mcq_with_ctt = {
            **mcq,
            "p_value": ctt["p_value"],
            "d_value": ctt["d_value"],
            "r_pb": ctt["r_pb"],
            "ctt_pass": ctt_filter(ctt)
        }
        final_mcqs.append(mcq_with_ctt)

        p_values.append(ctt["p_value"])
        d_values.append(ctt["d_value"])
        r_pbs.append(ctt["r_pb"])

    # Summary statistics (matching Table 11)
    p_arr = np.array(p_values)
    d_arr = np.array(d_values)
    r_arr = np.array(r_pbs)

    pct_diff = np.mean((p_arr >= 0.30) & (p_arr <= 0.70)) * 100
    pct_disc = np.mean(d_arr >= 0.40) * 100
    pct_valid = np.mean(r_arr >= 0.20) * 100

    summary = {
        "n_items": len(final_mcqs),
        "p_mean": round(float(np.mean(p_arr)), 3),
        "p_sd": round(float(np.std(p_arr)), 3),
        "pct_difficulty_ideal": round(pct_diff, 1),
        "d_mean": round(float(np.mean(d_arr)), 3),
        "d_sd": round(float(np.std(d_arr)), 3),
        "pct_discrimination_strong": round(pct_disc, 1),
        "rpb_mean": round(float(np.mean(r_arr)), 3),
        "rpb_sd": round(float(np.std(r_arr)), 3),
        "pct_validity_pass": round(pct_valid, 1),
    }

    if verbose:
        print(f"\n  {'ID':>4}  {'Bloom':10}  {'P-val':>6}  {'D-val':>6}  {'r_pb':>6}  {'CTT Pass':8}")
        print("  " + "-" * 56)
        for idx, m in enumerate(final_mcqs):
            status = "✓ PASS" if m["ctt_pass"] else "✗ FAIL"
            print(f"  {idx+1:>4}  {m['bloom_level']:10}  {m['p_value']:>6.3f}  "
                  f"{m['d_value']:>6.3f}  {m['r_pb']:>6.3f}  {status}")

        print(f"\n  ── CTT Summary (Table 11 equivalent) ──")
        print(f"     Total items         : {summary['n_items']}")
        print(f"     Mean P-value        : {summary['p_mean']} ± {summary['p_sd']}")
        print(f"     % in ideal diff.    : {summary['pct_difficulty_ideal']}%   (paper: 94.2%)")
        print(f"     Mean D-value        : {summary['d_mean']} ± {summary['d_sd']}")
        print(f"     % strong discrim.   : {summary['pct_discrimination_strong']}%   (paper: 89.9%)")
        print(f"     Mean r_pb           : {summary['rpb_mean']} ± {summary['rpb_sd']}")
        print(f"     % valid items       : {summary['pct_validity_pass']}%   (paper: 98.6%)")

    print("\n  ✓ Phase 5 complete.\n")
    return final_mcqs, summary


# ─────────────────────────────────────────────
# Bloom-level breakdown (Table 10 equivalent)
# ─────────────────────────────────────────────

def bloom_breakdown(final_mcqs: List[Dict]) -> None:
    """Print psychometric statistics per Bloom level."""
    print("\n  ── Bloom-Level Breakdown (Table 10 equivalent) ──")
    levels = {}
    for m in final_mcqs:
        lvl = m.get("bloom_level", "Unknown")
        levels.setdefault(lvl, []).append(m)

    header = f"  {'Level':12} {'Count':>6} {'%':>6} {'Mean P':>8} {'Mean D':>8} {'Mean rpb':>10} {'Mean Q':>8}"
    print(header)
    print("  " + "-" * 68)
    total = len(final_mcqs)
    for lvl, items in sorted(levels.items()):
        cnt = len(items)
        pct = cnt / max(total, 1) * 100
        mp = np.mean([m["p_value"] for m in items])
        md = np.mean([m["d_value"] for m in items])
        mr = np.mean([m["r_pb"] for m in items])
        mq = np.mean([m["quality_score"] for m in items])
        print(f"  {lvl:12} {cnt:>6} {pct:>6.1f} {mp:>8.3f} {md:>8.3f} {mr:>10.3f} {mq:>8.3f}")


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    dummy_mcqs = [
        {
            "chunk_text": "Gradient descent is an optimization algorithm used to minimize a loss function by iteratively updating model parameters in the direction of the negative gradient.",
            "answer": "gradient descent",
            "distractors": ["stochastic ascent", "backpropagation", "Newton's method"],
            "bloom_level": "Remember",
            "question": "What optimization algorithm minimizes a loss function by updating parameters along the negative gradient?",
            "grounding_score": 0.72,
            "quality_score": 0.0  # Will be computed
        }
    ]
    # Compute embeddings
    for m in dummy_mcqs:
        m["chunk_embedding"] = sbert.encode(m["chunk_text"], convert_to_numpy=True)

    # Phase 4
    retained = run_phase4(dummy_mcqs, sbert, q_threshold=0.35)

    # Phase 5
    if retained:
        final, summary = run_phase5(retained, n_students=50)
        bloom_breakdown(final)
