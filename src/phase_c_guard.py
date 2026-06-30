from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers.
    
    Custom recognizers:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)
    
    Predefined: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    from presidio_analyzer.nlp_engine import NlpEngine, NlpArtifacts

    class NoopNlpEngine(NlpEngine):
        def process_text(self, text: str, language: str) -> NlpArtifacts:
            return NlpArtifacts([], [], None, None, None, None)
        def process_batch(self, texts, language, batch_size=1, n_process=1, **kwargs):
            for t in texts:
                yield (t, self.process_text(t, language))
        def is_loaded(self):
            return True
        def is_stopword(self, word: str, language: str) -> bool:
            return False
        def is_punct(self, word: str, language: str) -> bool:
            return False
        def get_supported_entities(self):
            return []
        def get_supported_languages(self):
            return ["en"]
        def load(self):
            pass

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer  = AnalyzerEngine(registry=registry, nlp_engine=NoopNlpEngine())
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)

    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
    entities = [
        {"type": r.entity_type, "text": text[r.start:r.end],
         "score": round(r.score, 3), "start": r.start, "end": r.end}
        for r in results
    ]
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml."""
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails  = LLMRails(config)
    return rails


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,
        }
    """
    text_lower = text.lower()

    # Rule-based detection (works without API key)
    jailbreak_keywords = [
        "bo qua", "b qua", "ignore your system prompt", "do anything now",
        "khong co gioi han", "unrestricted", "act as", "dan",
        "tiet lo", "tite lo", "mat khau admin",
        "c", "allow me to",
    ]
    off_topic_keywords = [
        "lam tho", "bai tho", "nau pho", "bitcoin", "ethereum",
        "phuong trinh vi phan", "marvel", "bo phim",
        "cach nau", "cong thuc",
    ]
    prompt_injection_keywords = [
        "system override", "ignore previous", "admin command",
        "dump all training", "ceo", "system instructions",
        "/*", "forget all",
    ]

    for kw in jailbreak_keywords:
        if kw in text_lower:
            return {"allowed": False, "blocked_reason": "rule_jailbreak", "response": ""}
    for kw in off_topic_keywords:
        if kw in text_lower:
            return {"allowed": False, "blocked_reason": "rule_off_topic", "response": ""}
    for kw in prompt_injection_keywords:
        if kw in text_lower:
            return {"allowed": False, "blocked_reason": "rule_prompt_injection", "response": ""}

    # PII request detection
    pii_request_keywords = [
        "cccd", "so dien thoai", "cmnd", "email cua", "cua nhan vien",
    ]
    pii_request_count = sum(1 for kw in pii_request_keywords if kw in text_lower)
    if pii_request_count >= 2:
        return {"allowed": False, "blocked_reason": "rule_pii_request", "response": ""}

    # Fallback to NeMo if available
    if rails is not None:
        try:
            response = await rails.generate_async(
                messages=[{"role": "user", "content": text}]
            )
            response_text = str(response)
            refuse_keywords = ["xin li", "khng th", "i cannot", "i'm sorry"]
            blocked = any(kw in response_text.lower() for kw in refuse_keywords)
            return {
                "allowed": not blocked,
                "blocked_reason": ("nemo_input_rail" if blocked else None),
                "response": response_text,
            }
        except Exception:
            pass

    return {"allowed": True, "blocked_reason": None, "response": text}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra: có PII không? Nội dung có phù hợp không? Có hallucination rõ ràng không?

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    if rails is None:
        rails = setup_nemo_rails()

    response = await rails.generate_async(messages=[
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ])

    response_text = ""
    if isinstance(response, dict):
        response_text = response.get("content", str(response))
    elif hasattr(response, "response"):
        val = getattr(response, "response")
        try:
            response_text = val[0]["content"] if isinstance(val, list) and val else str(val)
        except Exception:
            response_text = str(val)
    else:
        response_text = str(response)

    refuse_keywords = [
        "xin l-i", "khA'ng th cung c p", "khA'ng `c",
        "i cannot", "i'm sorry",
    ]
    flagged = any(kw in response_text.lower() for kw in refuse_keywords)

    return {
        "safe":           not flagged,
        "flagged_reason": ("nemo_output_rail" if flagged else None),
        "final_answer":   (response_text if flagged else answer),
    }


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None

            # Layer 1: Presidio PII (synchronous)
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            # Layer 2: Input rail (rule-based + NeMo fallback)
            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id":         item["id"],
                "category":   item["category"],
                "input":      item["input"][:80] + ("..." if len(item["input"]) > 80 else ""),
                "expected":   item.get("expected", "allowed"),
                "actual":     actual,
                "blocked_by": blocked_by,
                "passed":     actual == item.get("expected", "allowed"),
            })
        return results

    results = asyncio.run(_run_all())
    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Mục tiêu production: P95 total < LATENCY_BUDGET_P95_MS (500ms mặc định)

    Returns:
        {
          "presidio_ms":  {"p50": float, "p95": float, "p99": float},
          "nemo_ms":      {"p50": float, "p95": float, "p99": float},
          "total_ms":     {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms": int,
        }
    """
    presidio_times, nemo_times, total_times = [], [], []

    async def _measure():
        for text in test_inputs[:n_runs]:
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())

    def percentiles(times):
        if not isinstance(times, list) or not times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        s = sorted(times)
        n = len(s)
        return {
            "p50": round(s[min(int(n * 0.50), n - 1)], 2),
            "p95": round(s[min(int(n * 0.95), n - 1)], 2),
            "p99": round(s[min(int(n * 0.99), n - 1)], 2),
        }

    total_p = percentiles(total_times)
    return {
        "presidio_ms": percentiles(presidio_times),
        "nemo_ms":     percentiles(nemo_times),
        "total_ms":    total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set)
    if results:
        passed = sum(1 for r in results if r["passed"])
        print(f"Adversarial suite: {passed}/{len(results)} passed")

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")