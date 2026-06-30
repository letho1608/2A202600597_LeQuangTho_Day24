# Failure Cluster Analysis — Phase A

**Sinh viên:** [Họ Tên]  
**Ngày:** [Ngày làm lab]

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | ? | ? | ? |
| answer_relevancy | ? | ? | ? |
| context_precision | ? | ? | ? |
| context_recall | ? | ? | ? |
| **avg_score** | ? | ? | ? |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| ... | | | | |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | | | | |
| answer_relevancy | | | | |
| context_precision | | | | |
| context_recall | | | | |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** [factual / multi_hop / adversarial]  
**Dominant metric:** [faithfulness / answer_relevancy / context_precision / context_recall]

**Lý do phân tích:**

> [Viết 3-5 câu giải thích tại sao distribution này hay bị failure, 
>  tại sao metric này thấp nhất trong corpus HR policy tiếng Việt]

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| context_recall | Missing relevant chunks | Improve chunking or add BM25 |
| context_precision | Too many irrelevant chunks | Add reranking or metadata filter |
| answer_relevancy | Answer doesn't match question | Improve prompt template |

---

## 6. Nhận xét về Adversarial Distribution

> [So sánh avg_score của adversarial vs factual vs multi_hop.
>  Pipeline có bị "nhầm" bởi version conflicts (v2023 vs v2024) không?
>  Câu nào trong bottom 10 rơi vào adversarial? Tại sao?]