"""
Phase 1: Document Processing
- Text extraction & preprocessing
- Sentence segmentation (spaCy)
- SBERT-based semantic chunking (θ=0.60, max 7 sentences)
- KeyBERT chunk refinement (MMR, λ=0.5)
"""

import re
import pdfplumber
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────
# 1. Text Extraction & Preprocessing
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF using pdfplumber, preserving reading order."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return "\n".join(pages_text)


def preprocess_text(raw_text: str) -> str:
    """
    Three-stage preprocessing pipeline (Section 2.2.1):
      1. Remove hyphenated line-break artifacts
      2. Merge newlines into single spaces
      3. Collapse multiple whitespaces
    """
    # Stage 1: remove hyphen + optional whitespace + newline
    text = re.sub(r"-\s*\n", "", raw_text)
    # Stage 2: merge remaining newlines (with surrounding whitespace) to a space
    text = re.sub(r"\s*\n\s*", " ", text)
    # Stage 3: collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# 2. Sentence Segmentation
# ─────────────────────────────────────────────

def segment_sentences(text: str, nlp) -> List[str]:
    """
    Use spaCy's dependency parser for accurate sentence boundary detection
    (Section 2.2.2). Returns S = {s1, s2, ..., sn}.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
    return sentences


# ─────────────────────────────────────────────
# 3. SBERT Semantic Chunking
# ─────────────────────────────────────────────

def encode_sentences(sentences: List[str], sbert_model: SentenceTransformer) -> np.ndarray:
    """Encode each sentence into a 384-dim SBERT embedding."""
    embeddings = sbert_model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    return embeddings  # shape: (n_sentences, 384)


def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Equation (1): cosine similarity between two embedding vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def semantic_chunking(
    sentences: List[str],
    embeddings: np.ndarray,
    theta_sim: float = 0.60,
    max_chunk_size: int = 7
) -> List[str]:
    """
    Algorithm 1 from the paper (Section 2.2.3):
      - Start new chunk when cosine similarity < θ_sim (0.60)
      - OR when current chunk reaches max_chunk_size (7)
    Returns C = {c1, c2, ..., ck} as joined strings.
    """
    if not sentences:
        return []

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i], embeddings[i - 1])

        if sim >= theta_sim and len(current_chunk) < max_chunk_size:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    # Append final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ─────────────────────────────────────────────
# 4. KeyBERT Chunk Refinement
# ─────────────────────────────────────────────

def mmr_keyword_selection(
    chunk_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidates: List[str],
    top_n: int = 5,
    lambda_val: float = 0.5
) -> List[str]:
    """
    Maximal Marginal Relevance (MMR) – Equation (3):
      MMR(k) = λ·cos(E(k), E(c)) − (1−λ)·max_j<k cos(E(k), E(k_j))
    Balances relevance vs. diversity among selected keyphrases.
    """
    if len(candidates) == 0:
        return []

    # Relevance scores: cos(E(k), E(c))
    chunk_emb_2d = chunk_embedding.reshape(1, -1)
    relevance_scores = cosine_similarity(candidate_embeddings, chunk_emb_2d).flatten()

    selected_indices = []
    remaining_indices = list(range(len(candidates)))

    for _ in range(min(top_n, len(candidates))):
        if not remaining_indices:
            break

        if not selected_indices:
            # Pick highest relevance first
            best_idx = remaining_indices[int(np.argmax(relevance_scores[remaining_indices]))]
        else:
            # MMR score for each remaining candidate
            mmr_scores = []
            selected_embs = candidate_embeddings[selected_indices]
            for idx in remaining_indices:
                rel = relevance_scores[idx]
                sim_to_selected = cosine_similarity(
                    candidate_embeddings[idx].reshape(1, -1), selected_embs
                ).max()
                mmr = lambda_val * rel - (1 - lambda_val) * sim_to_selected
                mmr_scores.append(mmr)
            best_idx = remaining_indices[int(np.argmax(mmr_scores))]

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]


def keybert_chunk_refinement(
    chunks: List[str],
    kw_model: KeyBERT,
    sbert_model: SentenceTransformer,
    top_n_keywords: int = 5,
    lambda_val: float = 0.5
) -> List[Dict]:
    """
    Section 2.2.4: For each chunk, extract keyphrases using KeyBERT + MMR.
    Returns list of dicts: {chunk_text, keywords, chunk_embedding}.
    """
    refined_chunks = []
    for chunk_text in chunks:
        if not chunk_text.strip():
            continue

        # KeyBERT extracts top-N keyphrases with MMR
        keywords_with_scores = kw_model.extract_keywords(
            chunk_text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            use_mmr=True,
            diversity=0.5,
            top_n=top_n_keywords
        )
        keywords = [kw for kw, _ in keywords_with_scores]

        chunk_embedding = sbert_model.encode(chunk_text, convert_to_numpy=True)

        refined_chunks.append({
            "text": chunk_text,
            "keywords": keywords,
            "embedding": chunk_embedding
        })

    return refined_chunks


# ─────────────────────────────────────────────
# 5. Full Phase 1 Pipeline
# ─────────────────────────────────────────────

def run_phase1(
    pdf_path: str,
    theta_sim: float = 0.60,
    max_chunk_size: int = 7,
    verbose: bool = True
) -> List[Dict]:
    """
    End-to-end Phase 1 execution:
      PDF → preprocess → segment → SBERT chunks → KeyBERT refinement
    Returns list of chunk dicts ready for Phase 2.
    """
    print("=" * 60)
    print("PHASE 1: Document Processing")
    print("=" * 60)

    # Load models once
    print("  [1/5] Loading models (spaCy, SBERT, KeyBERT)...")
    nlp = spacy.load("en_core_web_sm")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    kw_model = KeyBERT(model=sbert_model)

    # Step 1: Extract text
    print(f"  [2/5] Extracting text from: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"        Raw characters: {len(raw_text):,}")

    # Step 2: Preprocess
    print("  [3/5] Preprocessing text...")
    clean_text = preprocess_text(raw_text)
    print(f"        Clean characters: {len(clean_text):,}")

    # Step 3: Sentence segmentation
    print("  [4/5] Segmenting sentences with spaCy...")
    sentences = segment_sentences(clean_text, nlp)
    print(f"        Total sentences: {len(sentences)}")

    # Step 4: SBERT chunking
    print(f"  [5/5] Semantic chunking (θ={theta_sim}, max_size={max_chunk_size})...")
    embeddings = encode_sentences(sentences, sbert_model)
    raw_chunks = semantic_chunking(sentences, embeddings, theta_sim, max_chunk_size)
    print(f"        Chunks created: {len(raw_chunks)}")

    # Step 5: KeyBERT refinement
    print("        Refining chunks with KeyBERT + MMR...")
    refined_chunks = keybert_chunk_refinement(raw_chunks, kw_model, sbert_model)

    # Compute intra-chunk coherence stats
    if verbose:
        chunk_sizes = [len(c["text"].split()) for c in refined_chunks]
        print(f"\n  ── Chunk Statistics ──")
        print(f"     Mean word count/chunk : {np.mean(chunk_sizes):.1f}")
        print(f"     Median word count     : {np.median(chunk_sizes):.1f}")
        print(f"     Min/Max word count    : {min(chunk_sizes)}/{max(chunk_sizes)}")
        print(f"     Total chunks          : {len(refined_chunks)}")

    print("\n  ✓ Phase 1 complete.\n")
    return refined_chunks, sbert_model


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python phase1_document_processing.py <path_to_pdf>")
        sys.exit(1)

    chunks, sbert = run_phase1(sys.argv[1])

    print(f"\nSample chunk (first):\n{'-'*40}")
    print(chunks[0]["text"][:300], "...")
    print(f"Keywords: {chunks[0]['keywords']}")
