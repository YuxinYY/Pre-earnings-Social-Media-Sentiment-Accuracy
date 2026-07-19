# Project Summary and Plan


---

## Project Summary

### Original Framework

We are building a deep learning system that:
- Takes static social media text data (Reddit posts from 20xx–20xx via Academic Torrent)
- Encodes them using BERT embeddings
- Passes them through a Hierarchical Attention Network (HAN) with two attention layers:
  - **Post-level attention**: learns to weigh individual posts within a day
  - **Temporal attention**: learns to weigh important days within a time window
- Outputs a binary classification: whether a stock's future volatility will be high or low

The model architecture is inspired by the paper: *"Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction"* (WSDM 2018).

### Core Narrative

The project's framing is:

> **"Can a Hierarchical Attention Network, which simulates how humans filter key signals from massive social media noise, predict abnormal stock volatility using only Reddit text?"**

This is a legitimate and interesting research question. The HAN structure is designed precisely for this: filtering important information from a large pool of noisy documents. Reddit is a NOISY source, which makes the "filtering" story *stronger*, not weaker.

### Key Concern Identified

The main weakness of the current project is **not** the narrative — it is **experimental design**. Currently, you have a single model. Even with a 55% accuracy, you cannot answer:
- "Is the text providing any signal at all?"
- "Is the HAN structure better than a simpler text model?"
- "Is the model just learning a proxy for historical volatility (e.g., 'more discussion = recent high vol = future high vol')?"

---

## Revised Project Plan

### Phase 1: Fix the Experimental Design (Priority: Highest)

This is the minimum viable improvement before anything else.

#### 1a. Change the Label Definition

Use **relative abnormal volatility** instead of a fixed threshold:

- For each stock, compute the realized volatility for the next 5 days.
- Label = 1 if this volatility exceeds the stock's own past 20-day (or 60-day) mean + 1.5σ.
- This is computed **per stock, on a rolling basis**.
- Expect ~10–20% positive samples. Use AUC / F1 for evaluation, not accuracy.

#### 1b. Add Two Types of Baselines

| Baseline Type | What | Why |
|---|---|---|
| **Pure Market Baseline** | Logistic regression using only past realized volatility (5/10/20 days) + possibly volume/range | Answers: "Does text add anything beyond what historical vol already tells us?" |
| **Simple Text Baseline** | Average BERT embeddings per day → MLP or TF-IDF + Random Forest | Answers: "Does the HAN attention mechanism add value over simple aggregation?" |

**Ideal result**: HAN > Simple Text Baseline > Market Baseline → proves both "text has signal" and "HAN filtering is valuable."

#### 1c. Ticker Matching Filter (Simple Version)

Reddit posts mentioning "GME" may just be memes. Add a lightweight filter:
- Require ticker to appear with `$` prefix (e.g., `$GME`), or
- Require ticker to appear in the title, not just the body, or
- Remove posts from meme subreddits (like `wallstreetbets` if you want serious analysis)

This is not perfect but reduces noise significantly.

---

### Phase 2: Strengthen the Narrative & Analysis (Priority: Medium-High)

#### 2a. Visualize Attention Weights

This is the HAN's greatest selling point and your strongest evidence for the "filtering" narrative:
- For a correctly classified high-volatility prediction, show:
  - Which days had highest temporal attention
  - Which posts within those days had highest post-level attention
- This gives you concrete, visual proof that "the model focused on the right signals."

#### 2b. Add Interpretability Analyses

- **Attention vs. sentiment**: Are high-attention posts more emotionally extreme (positive or negative)?
- **Attention vs. volume**: Does the model assign higher weight to days with unusually high post volume (viral events)?
- **Error analysis**: On misclassified samples, did the model focus on the wrong posts/days?

#### 2c. (Optional) FinBERT Enhancement

You can keep **pure text input** (maximally faithful to the original paper's spirit), but if you want a clean boost:
- Replace BERT with FinBERT (financially pre-trained BERT)
- This requires no change to model architecture, only the embedding layer

---

### Phase 3: Experimentation & Exploration (Priority: Low-Medium)

#### 3a. Sub-period Analysis

Leverage your 20-year data span:
- 2008 Financial Crisis
- 2020 COVID crash
- 2021 Meme stock frenzy
- 2022–2023 rising rate environment
- Does the model perform differently across regimes? This is an interesting finding in itself.

#### 3b. Stock-level Heterogeneity

- Large cap vs. small cap vs. meme stocks
- Does social media predict volatility better for small-cap stocks?
- Does the HAN attention pattern differ by stock type?

---

### What NOT to Change (Unless You Want To)

- **Model structure**: HAN + BiGRU (or BiLSTM) is fine. It's lightweight, interpretable, and suits the task.
- **Pure text input**: Keeping only text is a legitimate design choice — it preserves the clean "can text alone predict?" research question.
- **No need for market features**: The whole point is to isolate text signal. Market features would muddy the question.

---

## Immediate Action Items (To-Do List)

1. **Redefine labels** using per-stock rolling abnormal volatility threshold (1.5σ above 20-day mean).

2. **Implement baseline 1**: Logistic regression on historical realized volatility only.

3. **Implement baseline 2**: Daily average BERT embedding → MLP (or TF-IDF + RF).

4. **Add ticker filter**: At minimum, require `$TICKER` format or title appearance.

5. **Retrain HAN** with new labels → compare all three models using AUC.

6. **Visualize attention weights** for correctly classified high-volatility cases.

7. **Document everything** in a clear README with the refined narrative.

---

## Architecture Considerations: HAN vs. Transformer vs. Memory-Caching Transformer

A natural follow-up question is whether to compare the current HAN architecture against a pure Transformer or a Transformer with memory caching (e.g., Titans, Infini-attention).

### Correcting a Common Misconception

**Transformer is not "mainly for text generation."** BERT itself is a Transformer encoder, and the embeddings already used in this project are Transformer outputs. Encoder-only Transformers are a standard and highly effective choice for classification tasks. Generation is just one application of decoder-based models; Transformers are fundamentally sequence modelers, and classification is well within their scope.

### Why Transformer + Memory Caching Is Not Appropriate Here

Memory-caching mechanisms (e.g., Titans, Infini-attention) are designed to solve a specific problem: **extremely long contexts**—thousands, tens of thousands, or even millions of tokens—where standard attention's quadratic complexity becomes prohibitive or the context window cannot fit everything.

The temporal module in this project processes a **daily-level sequence** of length T, where T is roughly 10–30 days. At this scale:

- A standard Transformer encoder handles full attention over the entire sequence with no difficulty.
- The Bi-GRU "fixed-state compression bottleneck" is largely irrelevant for sequences of only ~20 steps; that bottleneck becomes painful at hundreds or thousands of steps.
- Memory caching has no meaningful role to play here.

Running a three-way comparison of HAN vs. pure Transformer vs. Transformer + memory caching would likely show **no meaningful difference** among them, because the experiment would not stress the design goal of the memory mechanism. Such a comparison would not yield a valid conclusion and could suggest a misunderstanding of why these architectures exist.

### Reasonable Architecture Ablation

The clean, useful comparison is:

> **Replace the temporal Bi-GRU + attention with a small Transformer encoder (e.g., 2 layers, 4 heads), leaving everything else unchanged.**

This directly addresses the question: *"For a short daily sequence, is recurrence + attention better than pure self-attention?"*

It is cheap, clean, and well-motivated.

### Priority Reminder

This architecture comparison is **secondary**. The core unresolved question remains whether text provides any signal beyond historical volatility. The correct sequence is:

1. Run the market baseline (historical volatility only).
2. Confirm that HAN beats the market baseline.
3. Only then compare HAN against the small Transformer encoder.

### When Would Memory Become Relevant?

Memory mechanisms would only become relevant if the lookback window were extended to hundreds of days—for example, asking whether discussion patterns from six months ago still influence today's volatility. However, this is unlikely to be a compelling story in finance: recent information dominates volatility prediction, and distant history would likely add noise rather than signal.

**Bottom line:** Skip the memory-caching comparison. If time permits after establishing the baseline signal, run one simple HAN vs. small Transformer encoder ablation.
