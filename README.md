# CFPB Consumer Complaint Classifier

The Consumer Financial Protection Bureau (CFPB) is a US government agency that collects complaints from people who have problems with financial products like mortgages, credit cards, and bank accounts. Those complaints are published publicly in a database.

This project takes 3,000 of those complaints and tries to automatically classify each one into the correct category using a decision tree. The three categories are mortgage complaints, credit report complaints, and fraud complaints. The interesting part is comparing two approaches: feeding the raw text features directly into the tree vs first compressing them with PCA and then training the tree. PCA cuts the error rate from 12.2% down to 3.3%.

---

## The Data

3,000 complaints pulled from the public CFPB database, 1,000 per category:

| Category | What these complaints are about |
|---|---|
| `mortgage` | Escrow disputes, servicer problems, refinancing issues, foreclosure |
| `credit_report` | Incorrect entries on credit reports, identity disputes, FCRA violations |
| `fraud` | Unauthorized charges, account takeovers, scam transactions |

The raw complaint text is not used directly as input to the model. Instead, each complaint was first passed through OpenAI's text embedding model, which converts the words into a list of 1,536 numbers. Think of it as translating each complaint into a point in a very high-dimensional space where complaints that are about similar things end up close to each other. The decision trees in this project train on those numbers, not on the original text.

---

## Word Cloud

Before any modeling, it helps to just look at what words appear most often across all 3,000 complaints. Common stopwords and CFPB's placeholder strings (like XXXX used to redact account numbers) were removed first.

![Word Cloud](charts/01_wordcloud.png)

Words like *consumer*, *information*, *report*, *section*, and *mortgage* dominate. These are broad enough to appear in all three complaint types, which is actually a hint that a simple word-counting approach would not separate the categories well. The embedding model captures meaning rather than just word frequency, which is why it works better.

---

## Step 1: Decision Tree on Raw Embedding Features

The simplest starting point. Take all 1,536 embedding dimensions and feed them directly into a decision tree. The data is split 80% for training and 20% for testing.

The tree is first grown fully, then pruned using cross-validation to find the size that minimizes misclassification error on held-out data. The chart below shows how error changes as tree size increases.

**Cross-validation curve:**

![CV Raw Tree](charts/02_cv_raw_tree.png)

The error bottoms out and then starts creeping back up as the tree gets too large and starts memorizing the training data rather than learning general patterns. The dashed line marks the optimal size.

**The pruned tree:**

![Pruned Raw Tree](charts/03_pruned_raw_tree.png)

Each internal node shows a split condition like `X746 < -0.00184`. These variable names (X746, X1071, etc.) refer to specific dimensions in the 1,536-dimensional embedding space. They do not have human-readable names because embeddings are not designed to be interpretable that way — each dimension is a learned combination of patterns across thousands of words.

**Confusion matrix:**

![Confusion Matrix Raw](charts/04_confusion_raw.png)

Reading the confusion matrix: the diagonal shows correct predictions. Off-diagonal cells are mistakes. Fraud is the hardest category — 42 out of 201 fraud complaints get misclassified, mostly confused with mortgage or credit report. That is a 20.9% fraud miss rate.

**Raw features results:**
- Overall error rate: **12.2%**
- Fraud miss rate: **20.9%** — about 1 in 5 fraud complaints get incorrectly classified

---

## Why 1,536 Features is a Problem

A decision tree works by finding the best split at each node. With 1,536 dimensions to choose from, many of which are highly correlated (similar embedding dimensions tend to move together), the tree wastes splits on noisy or redundant features. It ends up making many small distinctions that do not generalize to new data.

The fix is to compress the 1,536 dimensions down to a smaller set of dimensions that capture most of the useful variation. That is what PCA does.

---

## Step 2: PCA, then Decision Tree

PCA (Principal Component Analysis) takes the 1,536 embedding dimensions and creates a new set of dimensions called principal components. Each principal component is a direction in the data that captures as much variation as possible, and they are uncorrelated with each other by construction. You can then keep only the first N components that together explain enough of the total variance.

The chart below shows how many components are needed to reach 50% of the total variance explained:

![PCA Variance](charts/05_pca_variance.png)

The curve flattens quickly. Just 29 components out of 1,536 get us to 50% of the variance. This means most of the information in the original 1,536 dimensions can be summarized in 29 numbers. The rest is mostly noise.

**Cross-validation on PCA features:**

![CV PCA Tree](charts/06_cv_pca_tree.png)

The PCA tree converges to a much lower error faster and stays there. The optimal tree is also simpler.

**The pruned PCA tree:**

![Pruned PCA Tree](charts/07_pruned_pca_tree.png)

This tree is far simpler than the raw one. It only needs two splits: first on PC1 (the single most important direction in the embedding space), then on PC2. If PC1 is below a threshold, the complaint is classified as credit report. If not, PC2 determines whether it is fraud or mortgage. Clean and interpretable.

**Confusion matrix:**

![Confusion Matrix PCA](charts/08_confusion_pca.png)

Almost everything lands on the diagonal. Mortgage complaints in particular are perfectly classified — 200 out of 200. Fraud misses drop from 42 to just 8.

**PCA results:**
- Overall error rate: **3.3%**
- Fraud miss rate: **4.0%** — about 1 in 25

---

## PCA Scatter Plot

Plotting the first two principal components and coloring by category shows why PCA works so well here. The three groups are already fairly well separated in just two dimensions.

![PCA Scatter](charts/09_pca_scatter.png)

Credit report complaints (dark blue) cluster to the left along PC1. Mortgage (teal) and fraud (orange) sit to the right but split along PC2 — mortgage goes up, fraud goes down. That structure is exactly what the simple two-split PCA tree is exploiting.

---

## Results Summary

| Model | Features Used | Overall Error | Fraud Miss Rate |
|---|---|---|---|
| Decision tree on raw embeddings | 1,536 dimensions | 12.2% | 20.9% |
| Decision tree on PCA features | 29 components | **3.3%** | **4.0%** |

Reducing from 1,536 correlated dimensions to 29 uncorrelated ones cuts the error rate by roughly 4x. The PCA tree is also much simpler and easier to understand — two splits instead of a deep branching structure.

---

## Files

```
cfpb-complaint-classifier/
├── cfpb_classification.R      full analysis with model outputs
├── generate_charts.R          saves all charts as PNGs
├── charts/
│   ├── 01_wordcloud.png
│   ├── 02_cv_raw_tree.png
│   ├── 03_pruned_raw_tree.png
│   ├── 04_confusion_raw.png
│   ├── 05_pca_variance.png
│   ├── 06_cv_pca_tree.png
│   ├── 07_pruned_pca_tree.png
│   ├── 08_confusion_pca.png
│   └── 09_pca_scatter.png
└── README.md
```

`CFPB_Complaints.csv` is not in the repo. The raw complaint data is publicly available through the [CFPB complaint database](https://www.consumerfinance.gov/data-research/consumer-complaints/).

## How to Run

```r
install.packages(c("tm", "wordcloud", "tree"))
# Then run generate_charts.R to produce the charts
# Then run cfpb_classification.R for the full model output
```
