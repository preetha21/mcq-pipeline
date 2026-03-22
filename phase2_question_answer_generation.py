"""
Phase 2: Question & Answer Generation
- Bloom-aligned question generation with FLAN-T5-Large (T=0.8, top_p=0.93)
- SBERT grounding validation (τ=0.45)
- Extractive answer identification with DeBERTa-v3-base (score≥0.45, len≥4 words)
"""
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering
)
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# Bloom Taxonomy Prompt Templates
# ─────────────────────────────────────────────

BLOOM_PROMPTS = {
    "Remember": (
        "Generate a factual recall question from the following text. "
        "The question should ask the student to remember a specific fact, term, or definition.\n\n"
        "Text: {chunk}\n\nQuestion:"
    ),
    "Understand": (
        "Generate a comprehension question from the following text. "
        "The question should test whether the student understands the concept or can explain it.\n\n"
        "Text: {chunk}\n\nQuestion:"
    ),
    "Apply": (
        "Generate an application question from the following text. "
        "The question should ask the student to apply a concept to a new situation or example.\n\n"
        "Text: {chunk}\n\nQuestion:"
    ),
}

# Distribution: Remember=28.1%, Understand=36.0%, Apply=36.0% (Table 10)
BLOOM_DISTRIBUTION = ["Remember", "Understand", "Apply"]


def get_bloom_level(index: int) -> str:
    """Cycle through Bloom levels to match the paper's 28/36/36% distribution."""
    weights = [0.281, 0.360, 0.359]
    rng = np.random.default_rng(index)
    return rng.choice(BLOOM_DISTRIBUTION, p=weights)


# ─────────────────────────────────────────────
# 1. Question Generation – FLAN-T5-Large
# ─────────────────────────────────────────────

class QuestionGenerator:
    """
    Section 2.3.1: Bloom-conditioned question generation using FLAN-T5-Large.
    Parameters: T=0.8, top_p=0.93, max_new_tokens=60
    """

    def __init__(self, model_name: str = "google/flan-t5-large", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Loading FLAN-T5-Large on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate(
        self,
        chunk_text: str,
        bloom_level: str = "Understand",
        temperature: float = 0.8,
        top_p: float = 0.93,
        max_new_tokens: int = 60,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Equation (4): q = T5_θ(p) = decode(generate(p; T=0.8, p_top=0.93, L_max=60))
        """
        prompt_template = BLOOM_PROMPTS.get(bloom_level, BLOOM_PROMPTS["Understand"])
        prompt = prompt_template.format(chunk=chunk_text[:800])  # Fit within token budget

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                early_stopping=True
            )

        questions = [
            self.tokenizer.decode(out, skip_special_tokens=True).strip()
            for out in outputs
        ]
        return [q for q in questions if len(q) > 10]  # Filter trivial outputs


# ─────────────────────────────────────────────
# 2. Grounding Validation – SBERT
# ─────────────────────────────────────────────

def compute_grounding_score(
    question: str,
    chunk_text: str,
    sbert_model: SentenceTransformer
) -> float:
    """
    Equation (5): G(q, c) = (E(q)·E(c)) / (||E(q)|| · ||E(c)||)
    Threshold τ_ground = 0.45 (Section 2.3.2).
    """
    emb_q = sbert_model.encode(question, convert_to_numpy=True)
    emb_c = sbert_model.encode(chunk_text, convert_to_numpy=True)

    norm_q = np.linalg.norm(emb_q)
    norm_c = np.linalg.norm(emb_c)
    if norm_q == 0 or norm_c == 0:
        return 0.0

    return float(np.dot(emb_q, emb_c) / (norm_q * norm_c))


def validate_grounding(
    question: str,
    chunk_text: str,
    sbert_model: SentenceTransformer,
    tau_ground: float = 0.45
) -> Tuple[bool, float]:
    """
    Returns (is_grounded, grounding_score).
    A question is accepted only if G(q,c) ≥ τ_ground (0.45).
    """
    score = compute_grounding_score(question, chunk_text, sbert_model)
    return score >= tau_ground, score


# ─────────────────────────────────────────────
# 3. Answer Extraction – DeBERTa-v3-base
# ─────────────────────────────────────────────

class AnswerExtractor:
    """
    Section 2.3.3: Extractive QA with DeBERTa-v3-base (SQuAD2-fine-tuned).
    Filters: min_answer_words=4, confidence≥0.45
    """

    def __init__(
        self,
        model_name: str = "deepset/deberta-v3-base-squad2",
        device: int = 0 if torch.cuda.is_available() else -1
    ):
        print(f"  Loading DeBERTa-v3-base QA model...")
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer as QATokenizer
        import torch

        self.qa_tokenizer = QATokenizer.from_pretrained(model_name)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.qa_model.eval()

    def extract(self, question, context, min_answer_words=4, confidence_threshold=0.45):
        try:
            inputs = self.qa_tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )

            with torch.no_grad():
                outputs = self.qa_model(**inputs)

            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

            # Get best start/end positions
            start_idx = int(torch.argmax(start_logits))
            end_idx = int(torch.argmax(end_logits))

        # end must be after start
            if end_idx < start_idx:
                end_idx = start_idx + 3

        # Decode answer tokens
            input_ids = inputs["input_ids"][0]
            answer_tokens = input_ids[start_idx: end_idx + 1]
            answer = self.qa_tokenizer.decode(
            answer_tokens,
            skip_special_tokens=True
            ).strip()

        # Confidence approximation
            import torch.nn.functional as F
            start_prob = float(F.softmax(start_logits, dim=0)[start_idx])
            end_prob = float(F.softmax(end_logits, dim=0)[end_idx])
            score = (start_prob + end_prob) / 2.0

            word_count = len(answer.split())
            if word_count < min_answer_words or score < confidence_threshold:
                return None

            return {
                "answer": answer,
                "score": score,
                "start": start_idx,
                "end": end_idx
            }

        except Exception as e:
            return None


# ─────────────────────────────────────────────
# 4. Full Phase 2 Pipeline
# ─────────────────────────────────────────────

def run_phase2(
    chunks: List[Dict],
    sbert_model: SentenceTransformer,
    tau_ground: float = 0.45,
    verbose: bool = True
) -> List[Dict]:
    """
    For each chunk:
      1. Generate Bloom-aligned question (FLAN-T5-Large)
      2. Validate grounding with SBERT (τ=0.45)
      3. Extract answer with DeBERTa-v3-base
    Returns list of QA dicts for Phase 3.
    """
    print("=" * 60)
    print("PHASE 2: Question & Answer Generation")
    print("=" * 60)

    question_gen = QuestionGenerator()
    answer_ext = AnswerExtractor()

    qa_items = []
    grounding_pass = 0
    answer_pass = 0

    for idx, chunk in enumerate(chunks):
        chunk_text = chunk["text"]
        bloom_level = get_bloom_level(idx)

        # Step 1: Generate question
        questions = question_gen.generate(chunk_text, bloom_level=bloom_level)
        if not questions:
            continue
        question = questions[0]

        # Step 2: Grounding validation
        is_grounded, g_score = validate_grounding(question, chunk_text, sbert_model, tau_ground)
        if not is_grounded:
            if verbose:
                print(f"  [chunk {idx+1:>3}] ✗ Grounding {g_score:.3f} < {tau_ground} — rejected")
            continue
        grounding_pass += 1

        # Step 3: Answer extraction
        answer_result = answer_ext.extract(question, chunk_text)
        if answer_result is None:
            if verbose:
                print(f"  [chunk {idx+1:>3}] ✗ No valid answer extracted")
            continue
        answer_pass += 1

        qa_items.append({
            "chunk_text": chunk_text,
            "chunk_embedding": chunk["embedding"],
            "keywords": chunk.get("keywords", []),
            "bloom_level": bloom_level,
            "question": question,
            "answer": answer_result["answer"],
            "answer_score": answer_result["score"],
            "grounding_score": g_score
        })

        if verbose:
            print(f"  [chunk {idx+1:>3}] ✓ Bloom:{bloom_level:10s} | G:{g_score:.3f} | "
                  f"Ans:'{answer_result['answer'][:30]}...' (conf:{answer_result['score']:.3f})")

    print(f"\n  ── Phase 2 Summary ──")
    print(f"     Chunks processed       : {len(chunks)}")
    print(f"     Grounding pass (81.6%) : {grounding_pass} ({grounding_pass/max(len(chunks),1)*100:.1f}%)")
    print(f"     Answer extraction pass : {answer_pass} ({answer_pass/max(len(chunks),1)*100:.1f}%)")
    print(f"     QA pairs generated     : {len(qa_items)}")
    print("\n  ✓ Phase 2 complete.\n")
    return qa_items


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # Minimal test with a dummy chunk
    dummy_chunks = [{
        "text": (
            "SBERT (Sentence-BERT) generates dense sentence embeddings by fine-tuning "
            "BERT with siamese networks on natural language inference tasks. It enables "
            "efficient semantic similarity comparisons using cosine distance, making it "
            "well-suited for tasks like clustering, information retrieval, and semantic search."
        ),
        "keywords": ["SBERT", "sentence embeddings", "cosine distance"],
        "embedding": None
    }]

    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    # Compute embedding for dummy chunk
    dummy_chunks[0]["embedding"] = sbert.encode(dummy_chunks[0]["text"], convert_to_numpy=True)

    results = run_phase2(dummy_chunks, sbert, verbose=True)
    if results:
        print(f"\nSample QA item:")
        print(f"  Question : {results[0]['question']}")
        print(f"  Answer   : {results[0]['answer']}")
        print(f"  Bloom    : {results[0]['bloom_level']}")
        print(f"  Grounding: {results[0]['grounding_score']:.3f}")
