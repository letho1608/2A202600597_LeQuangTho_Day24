# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Le Quang Tho  
**MSSV:** 2A202600597  
**Ngày:** 30/06/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~1.2ms P50 / 3.5ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~245ms P50 / 412ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 1.2 | 3.5 | 8.1 | <10ms ✅ |
| NeMo Input Rail | 245 | 412 | 710 | <300ms ⚠️ |
| RAG Pipeline | 850 | 1200 | 1800 | <2000ms ✅ |
| NeMo Output Rail | 180 | 310 | 520 | <300ms ⚠️ |
| **Total Guard** | **1,276** | **1,925** | **3,038** | **<500ms ❌** |

**Budget OK?** [ ] Yes / [X] No  
**Comment:** NeMo input rail (LLM API call) là bottleneck chính — chiếm ~80% total latency.  
Trong production, cần tối ưu: (1) dùng model nhỏ hơn cho judge rail, (2) cache frequent queries, (3) tune timeout.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
name: RAG Eval + Guardrail CI
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install deps
        run: |
          pip install -r requirements.txt
          python -m spacy download en_core_web_lg

      - name: RAGAS Quality Gate
        run: python src/phase_a_ragas.py
        env:
          MIN_FAITHFULNESS: 0.75
          MIN_AVG_SCORE: 0.65

      - name: Guardrail Gate
        run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
        # phải ≥ 15/20 (75%)

      - name: Latency Gate
        run: |
          python -c "
from src.phase_c_guard import measure_p95_latency;
import json
with open('adversarial_set_20.json') as f:
    inputs = json.load(f)
    latency = measure_p95_latency([i['input'] for i in inputs], n_runs=5)
    assert latency['total_ms']['p95'] < 500, 'P95 > 500ms'
          "
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model / reduce model size |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.65 (trung bình 4 metrics) |
| Worst metric | context_recall (0.48) |
| Dominant failure distribution | multi_hop (cross-document queries) |
| Cohen's κ | 0.58 (moderate agreement — cần thêm data) |
| Adversarial pass rate | 16 / 20 (80%) |
| Guard P95 latency | 412 ms (Presidio + NeMo input) |

---

## Nhận xét & Cải tiến

**Điều hoạt động tốt:**
- Presidio PII scan cực kỳ nhanh (<10ms) — phát hiện chính xác CCCD 12 số và số điện thoại VN
- NeMo guardrails xử lý tốt các input off-topic và jailbreak bằng Colang rules
- RAGAS evaluation có insight rõ ràng về failure distribution (multi_hop là điểm yếu)

**Điều cần cải thiện:**
- NeMo latency vượt ngưỡng 500ms khi dùng gpt-4o-mini — cần cân nhắc model nhẹ hơn hoặc async pipeline
- Adversarial pass rate 16/20 vẫn chưa đạt ngưỡng bonus (18/20) — cần cải thiện rails.co để bắt thêm kiểu tấn công
- Cohen's κ 0.58 cần thêm dữ liệu label để đạt substantial agreement

**Nếu deploy production thực sự:**
1. Thay Presidio bằng solution có tiếng Việt native (chứ không chỉ regex pattern)
2. Dùng model cascade: phiếu nhỏ cho input rail, phiếu lớn hơn cho RAG
3. Thêm caching layer cho frequent HR queries
4. CI/CD pipeline cần đo latency thực tế trên mỗi PR, không chỉ mỗi lần release