from __future__ import annotations

"""M4: RAGAS-style evaluation (faithfulness, answer_relevancy,
context_precision, context_recall).

Wraps ``ragas`` >= 0.1.10 via the ``datasets.Dataset`` API and uses
``config.JUDGE_MODEL`` (default ``gpt-4o-mini``) as the LLM judge.

If the ``OPENAI_API_KEY`` is missing or ragas is not installed, returns
a fully-populated zero-score result so callers can always proceed.
"""

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import JUDGE_MODEL, OPENAI_API_KEY  # noqa: E402


# ─── Result dataclass (mirrors RagasResult in phase_a_ragas) ─────────────────

@dataclass
class RagasEvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4.0


# ─── Lazy / optional ragas imports ────────────────────────────────────────────

def _ragas_available() -> bool:
    try:
        import ragas  # noqa: F401
        from datasets import Dataset  # noqa: F401
        return True
    except Exception:
        return False


def _build_dataset(questions, answers, contexts, ground_truths):
    from datasets import Dataset
    return Dataset.from_dict({
        "question":     list(questions),
        "answer":       list(answers),
        "contexts":     [list(c) if c else [] for c in contexts],
        "ground_truth": list(ground_truths),
    })


def _empty_results(questions, answers, contexts, ground_truths) -> list[RagasEvalResult]:
    return [
        RagasEvalResult(
            question=q, answer=a, contexts=list(c or []), ground_truth=gt,
        )
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
    ]


def _safe_metric(dataset, metric_name: str) -> list[float]:
    """Run a single ragas metric and return aligned scores, or zeros."""
    try:
        from ragas.metrics import (  # type: ignore
            faithfulness, answer_relevancy, context_precision, context_recall,
        )
        metrics_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        metric = metrics_map[metric_name]
        from ragas import evaluate  # type: ignore
        result = evaluate(dataset, metrics=[metric])
        # result is EvaluationResult; scores are in result[metric_name]
        try:
            scores = list(result[metric_name])
        except Exception:
            scores = list(result.scores[metric_name])
        return [float(s) if s is not None else 0.0 for s in scores]
    except Exception:
        return [0.0] * len(dataset)


# ─── Public API ──────────────────────────────────────────────────────────────

def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str],
                   model: str | None = None) -> dict:
    """Evaluate a batch of RAG predictions with RAGAS metrics.

    Returns:
        {
          "per_question":  list[RagasEvalResult]   # one per question
          "aggregate":     {metric: float}         # mean across questions
          "model":         str                     # judge model used
          "ran":           bool                    # True if ragas actually ran
        }
    """
    n = len(questions)
    if n == 0:
        return {"per_question": [], "aggregate": {},
                "model": model or JUDGE_MODEL, "ran": False}

    # ── Graceful fallback: no API key or no ragas → zeros, no crash ────────
    if not OPENAI_API_KEY or not _ragas_available():
        return {
            "per_question": _empty_results(questions, answers, contexts, ground_truths),
            "aggregate": {
                "faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0,
            },
            "model": model or JUDGE_MODEL,
            "ran": False,
        }

    judge_model = model or JUDGE_MODEL
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

    try:
        dataset = _build_dataset(questions, answers, contexts, ground_truths)
        f_scores = _safe_metric(dataset, "faithfulness")
        ar_scores = _safe_metric(dataset, "answer_relevancy")
        cp_scores = _safe_metric(dataset, "context_precision")
        cr_scores = _safe_metric(dataset, "context_recall")

        per_q = []
        for i, (q, a, c, gt) in enumerate(zip(questions, answers, contexts, ground_truths)):
            per_q.append(RagasEvalResult(
                question=q, answer=a, contexts=list(c or []), ground_truth=gt,
                faithfulness=float(f_scores[i]) if i < len(f_scores) else 0.0,
                answer_relevancy=float(ar_scores[i]) if i < len(ar_scores) else 0.0,
                context_precision=float(cp_scores[i]) if i < len(cp_scores) else 0.0,
                context_recall=float(cr_scores[i]) if i < len(cr_scores) else 0.0,
            ))

        def _mean(xs: list[float]) -> float:
            return round(sum(xs) / len(xs), 4) if xs else 0.0

        return {
            "per_question": per_q,
            "aggregate": {
                "faithfulness":      _mean(f_scores),
                "answer_relevancy":  _mean(ar_scores),
                "context_precision": _mean(cp_scores),
                "context_recall":    _mean(cr_scores),
            },
            "model": judge_model,
            "ran": True,
        }
    except Exception:
        # Any failure → return zeros; never crash the pipeline.
        return {
            "per_question": _empty_results(questions, answers, contexts, ground_truths),
            "aggregate": {
                "faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0,
            },
            "model": judge_model,
            "ran": False,
        }


if __name__ == "__main__":
    out = evaluate_ragas(
        questions=["Bao nhiêu ngày phép năm?"],
        answers=["15 ngày."],
        contexts=[["Mỗi nhân viên chính thức được hưởng 15 ngày phép năm."]],
        ground_truths=["15 ngày phép năm theo chính sách v2024."],
    )
    print("Ran:", out["ran"], "Model:", out["model"])
    if out["per_question"]:
        r = out["per_question"][0]
        print(f"  F={r.faithfulness} AR={r.answer_relevancy} "
              f"CP={r.context_precision} CR={r.context_recall}")
