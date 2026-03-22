"""
Phase 3: Distractor Generation
- Multi-source generation: DistilBERT MLM, WordNet, T5-Small paraphrase, KeyBERT
- SBERT semantic filtering: 0.30 ≤ D(a, d_i) ≤ 0.83
- Optimal distance targeting D* = 0.52
- Inter-distractor diversity: cos(d_i, d_j) < τ_div = 0.85
"""

import re
import numpy as np
import torch
import nltk
from typing import List, Dict, Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    pipeline
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download WordNet once
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

from nltk.corpus import wordnet


# ─────────────────────────────────────────────
# 1. MLM Distractor Generator (DistilBERT)
# ─────────────────────────────────────────────

class MLMDistractorGenerator:
    """
    Section 2.4.1 – Equation (7):
      Replace correct answer with [MASK] and predict top-k tokens via DistilBERT.
      C_MLM = {top-k t_i | t_i = argmax P(t | c_mask)}
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        print("  Loading DistilBERT MLM...")
        self.device = 0 if torch.cuda.is_available() else -1
        self.fill_mask = pipeline(
            "fill-mask",
            model=model_name,
            device=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, answer: str, chunk_text: str, top_k: int = 10) -> List[str]:
        """
        Mask the first occurrence of `answer` in `chunk_text` and predict replacements.
        """
        # Escape regex special chars in answer
        pattern = re.escape(answer)
        masked_text = re.sub(pattern, self.tokenizer.mask_token, chunk_text, count=1)

        if self.tokenizer.mask_token not in masked_text:
            # Answer not found verbatim – mask first noun-like token as fallback
            words = chunk_text.split()
            if words:
                words[0] = self.tokenizer.mask_token
                masked_text = " ".join(words)

        try:
            predictions = self.fill_mask(masked_text[:512], top_k=top_k)
            candidates = []
            for pred in predictions:
                token_str = pred.get("token_str", "").strip()
                if token_str and token_str.lower() != answer.lower() and len(token_str) > 1:
                    candidates.append(token_str)
            return candidates
        except Exception:
            return []


# ─────────────────────────────────────────────
# 2. WordNet Lexical Distractor Generator
# ─────────────────────────────────────────────

def wordnet_distractors(answer: str) -> List[str]:
    """
    Section 2.4.1 – Equation (8):
      C_WN = {ℓ | ℓ ∈ lemmas(s), s ∈ synsets(a)}
    Returns synonyms, hypernyms, and coordinate terms.
    """
    candidates = set()
    answer_lower = answer.lower().replace(" ", "_")

    synsets = wordnet.synsets(answer_lower)
    for syn in synsets:
        # Lemmas (synonyms)
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != answer.lower():
                candidates.add(name)

        # Hypernyms (broader terms)
        for hypernym in syn.hypernyms():
            for lemma in hypernym.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != answer.lower():
                    candidates.add(name)

        # Coordinate terms (same hypernym, different synset)
        for hypernym in syn.hypernyms():
            for hyponym in hypernym.hyponyms():
                if hyponym != syn:
                    for lemma in hyponym.lemmas():
                        name = lemma.name().replace("_", " ")
                        if name.lower() != answer.lower():
                            candidates.add(name)

    return list(candidates)[:20]  # Cap to avoid explosion


# ─────────────────────────────────────────────
# 3. T5-Small Paraphrase Generator
# ─────────────────────────────────────────────

class T5Paraphraser:
    """
    Section 2.4.1: Paraphrase the correct answer using T5-Small.
      C_PAR = {p_i = T5_small(a)}
    """

    def __init__(self, model_name: str = "t5-small"):
        print("  Loading T5-Small paraphraser...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def paraphrase(self, text: str, num_variants: int = 4) -> List[str]:
        """Generate syntactically diverse variants that preserve core meaning."""
        prompt = f"paraphrase: {text}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                num_return_sequences=num_variants,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                early_stopping=True
            )

        paraphrases = []
        for out in outputs:
            decoded = self.tokenizer.decode(out, skip_special_tokens=True).strip()
            if decoded.lower() != text.lower() and len(decoded) > 3:
                paraphrases.append(decoded)
        return paraphrases


# ─────────────────────────────────────────────
# 4. KeyBERT Contextual Keyphrases
# ─────────────────────────────────────────────

def keybert_distractors(chunk_keywords: List[str], answer: str) -> List[str]:
    """
    Section 2.4.1 – Equation (9): Use pre-extracted chunk keywords as distractor candidates.
    Filters out the correct answer itself.
    """
    return [kw for kw in chunk_keywords if kw.lower() != answer.lower()]


# ─────────────────────────────────────────────
# 5. SBERT Semantic Filtering & Ranking
# ─────────────────────────────────────────────

def cosine_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Equation (10): D(a, d_i) = 1 - (E(a)ᵀ E(d_i)) / (||E(a)||₂ · ||E(d_i)||₂)
    """
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(emb_a, emb_b) / (norm_a * norm_b))


def sbert_filter_and_rank(
    answer: str,
    candidates: List[str],
    sbert_model: SentenceTransformer,
    d_min: float = 0.30,
    d_max: float = 0.83,
    d_star: float = 0.52,
    tau_div: float = 0.85,
    num_distractors: int = 3
) -> List[Dict]:
    """
    Section 2.4.2:
      1. Embed answer and all candidates
      2. Retain candidates where d_min ≤ D(a, d_i) ≤ d_max
      3. Score by deviation from D* (Equation 11): S(d_i) = |D(a, d_i) − D*|
      4. Select top-k with inter-distractor diversity cos(d_i, d_j) < τ_div
    """
    if not candidates:
        return []

    # Deduplicate
    unique_candidates = list(dict.fromkeys(
        c.strip() for c in candidates
        if c.strip() and c.strip().lower() != answer.strip().lower()
    ))
    if not unique_candidates:
        return []

    # Encode
    answer_emb = sbert_model.encode(answer, convert_to_numpy=True)
    candidate_embs = sbert_model.encode(unique_candidates, convert_to_numpy=True)

    # Step 1: Filter by distance band
    valid = []
    for i, cand in enumerate(unique_candidates):
        dist = cosine_distance(answer_emb, candidate_embs[i])
        if d_min <= dist <= d_max:
            valid.append({
                "text": cand,
                "embedding": candidate_embs[i],
                "distance": dist,
                "deviation": abs(dist - d_star)  # Equation (11)
            })

    if not valid:
        return []

    # Step 2: Sort by deviation from D* (ascending = closer to ideal)
    valid.sort(key=lambda x: x["deviation"])

    # Step 3: Greedy selection with inter-distractor diversity
    selected = []
    for candidate in valid:
        if len(selected) >= num_distractors:
            break

        # Check diversity against already-selected distractors
        if selected:
            selected_embs = np.vstack([s["embedding"] for s in selected])
            sims = cosine_similarity(
                candidate["embedding"].reshape(1, -1), selected_embs
            ).flatten()
            if sims.max() >= tau_div:
                continue  # Too similar to an already-selected distractor

        selected.append(candidate)

    # If fewer than 3 found, pad with next-best admissible candidates
    if len(selected) < num_distractors:
        for candidate in valid:
            if len(selected) >= num_distractors:
                break
            if candidate not in selected:
                selected.append(candidate)

    return selected[:num_distractors]


# ─────────────────────────────────────────────
# 6. Full Phase 3 Pipeline
# ─────────────────────────────────────────────

def run_phase3(
    qa_items: List[Dict],
    sbert_model: SentenceTransformer,
    d_min: float = 0.30,
    d_max: float = 0.83,
    d_star: float = 0.52,
    tau_div: float = 0.85,
    verbose: bool = True
) -> List[Dict]:
    """
    For each QA item:
      1. Generate distractor pool from 4 sources
      2. Apply SBERT semantic filtering and ranking
    Returns enriched MCQ dicts with distractors added.
    """
    print("=" * 60)
    print("PHASE 3: Distractor Generation")
    print("=" * 60)

    mlm_gen = MLMDistractorGenerator()
    paraphraser = T5Paraphraser()

    mcq_items = []
    functional_count = 0

    for idx, item in enumerate(qa_items):
        answer = item["answer"]
        chunk_text = item["chunk_text"]
        keywords = item.get("keywords", [])

        # ── Source 1: MLM (DistilBERT) ──
        c_mlm = mlm_gen.generate(answer, chunk_text, top_k=10)

        # ── Source 2: WordNet ──
        c_wn = wordnet_distractors(answer)

        # ── Source 3: T5-Small Paraphrase ──
        c_par = paraphraser.paraphrase(answer, num_variants=4)

        # ── Source 4: KeyBERT keyphrases ──
        c_kb = keybert_distractors(keywords, answer)

        # ── Union & deduplicate ──
        all_candidates = list(dict.fromkeys(c_mlm + c_wn + c_par + c_kb))

        # ── SBERT filtering & ranking ──
        selected_distractors = sbert_filter_and_rank(
            answer=answer,
            candidates=all_candidates,
            sbert_model=sbert_model,
            d_min=d_min,
            d_max=d_max,
            d_star=d_star,
            tau_div=tau_div,
            num_distractors=3
        )

        distractor_texts = [d["text"] for d in selected_distractors]
        distractor_distances = [d["distance"] for d in selected_distractors]

        if len(distractor_texts) >= 1:
            functional_count += 1

        mcq_items.append({
            **item,
            "distractors": distractor_texts,
            "distractor_distances": distractor_distances,
            "candidate_pool_size": len(all_candidates),
            "num_distractors": len(distractor_texts)
        })

        if verbose:
            status = "✓" if len(distractor_texts) >= 3 else f"~({len(distractor_texts)}/3)"
            dists_str = ", ".join(f"{d:.3f}" for d in distractor_distances)
            print(f"  [item {idx+1:>3}] {status} | Distractors: {distractor_texts} "
                  f"| Distances: [{dists_str}]")

    print(f"\n  ── Phase 3 Summary ──")
    print(f"     QA items processed        : {len(qa_items)}")
    print(f"     Items with ≥1 distractor  : {functional_count} "
          f"({functional_count/max(len(qa_items),1)*100:.1f}%)")
    full_3 = sum(1 for m in mcq_items if m["num_distractors"] >= 3)
    print(f"     Items with 3 distractors  : {full_3} "
          f"({full_3/max(len(qa_items),1)*100:.1f}%)")
    print("\n  ✓ Phase 3 complete.\n")
    return mcq_items


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    dummy_items = [{
        "chunk_text": (
            "Transformer architectures rely on self-attention mechanisms to model "
            "long-range dependencies in sequential data without recurrence."
        ),
        "chunk_embedding": sbert.encode(
            "Transformer architectures rely on self-attention mechanisms.", convert_to_numpy=True
        ),
        "keywords": ["self-attention", "transformer", "sequential data", "recurrence"],
        "bloom_level": "Understand",
        "question": "What mechanism do transformer architectures use to model long-range dependencies?",
        "answer": "self-attention mechanisms",
        "answer_score": 0.82,
        "grounding_score": 0.71
    }]

    mcqs = run_phase3(dummy_items, sbert, verbose=True)
    print("\nSample MCQ:")
    m = mcqs[0]
    print(f"  Q: {m['question']}")
    print(f"  A: {m['answer']}")
    for i, d in enumerate(m["distractors"]):
        print(f"  D{i+1}: {d}")
