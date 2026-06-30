# LLM Judge Bias Report — Phase B

**Sinh viên:** [Họ Tên]  
**Ngày:** [Ngày làm lab]  
**Judge model:** gpt-4o-mini

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() trên ít nhất 5 cặp answers)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| ... | | | |

---

## 2. Swap-and-Average Results

*(Chạy swap_and_average() trên cùng các cặp)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |

**Position bias rate:** ?% (= số case NOT consistent / tổng)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)  
**Judge labels:** [kết quả chạy judge trên 10 câu tương ứng]

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | | | |
| 5 | | | |
| 12 | | | |
| 21 | | | |
| 23 | | | |
| 29 | | | |
| 33 | | | |
| 41 | | | |
| 46 | | | |
| 50 | | | |

**Cohen's κ:** ?  
**Interpretation:** [poor / slight / fair / moderate / substantial / almost perfect]

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: ? / ? cases
- B thắng + B dài hơn A: ? / ? cases  
- **Verbosity bias rate:** ?%

**Kết luận:** [LLM có xu hướng chọn answer dài hơn không? Tại sao điều này là vấn đề?]

---

## 5. Nhận xét chung

> [Viết 3-5 câu nhận xét:
>  - κ > 0.6 chưa? LLM judge đáng tin không?
>  - Position bias đáng lo ngại không (>30%)?
>  - Swap-and-average có thực sự giúp ích không?
>  - Trong môi trường production, nên dùng judge như thế nào?]