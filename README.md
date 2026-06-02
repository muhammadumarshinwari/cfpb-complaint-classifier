# CFPB Complaint Classification

Classifying consumer financial complaints using OpenAI text embeddings and decision trees in R. The data comes from the Consumer Financial Protection Bureau (CFPB), which collects written complaints from Americans about financial products and services. The goal is to predict which category a complaint belongs to based solely on its text.

---

## The Data

The dataset contains 3,000 complaints, evenly split across three categories:

| Category | Count | Description |
|---|---|---|
| `mortgage` | 1,000 | Escrow issues, servicer disputes, refinance problems |
| `credit_report` | 1,000 | Incorrect entries, identity disputes, FCRA violations |
| `fraud` | 1,000 | Unauthorized charges, account takeovers, scams |

Each complaint was passed through OpenAI's text embedding model, which converts the raw text into a 1,536-dimensional numeric vector. These embedding features are what the models actually train on — the raw text is not used directly in the tree models.

| Metric | Value |
|---|---|
| Total complaints | 3,000 |
| Embedding dimensions | 1,536 |
| Average complaint length | 143 tokens |
| Min / Max tokens | 70 / 279 |

---

## Step 0 — Word Cloud

Before modeling, a word cloud was built to see which terms appear most often across all 3,000 complaints. Standard pre-processing: lowercase conversion, punctuation and number removal, English stopwords, and CFPB redaction tokens (XXXX).

```r
corpus <- Corpus(VectorSource(CFPB$message))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, removeWords, c("xxxx", "xxx", "xx"))
corpus <- tm_map(corpus, stripWhitespace)
```

![Word Cloud](charts/01_wordcloud.png)

Common terms like *account*, *report*, *received*, and *told* dominate — fairly generic vocabulary shared across all three complaint types, which is why raw text frequency alone does not separate categories cleanly.

---

## Step 1 — Decision Tree on Raw Embedding Features

The 1,536 embedding dimensions were used directly as features in an 80/20 train/test split.

```r
set.seed(1)
trainindex <- sample(nrow(CFPB), size = 0.8 * nrow(CFPB))
CFPB_train <- CFPB[trainindex, ]
CFPB_test  <- CFPB[-trainindex, ]

tree_model <- tree(issue ~ ., data = CFPB_train)
```

Cross-validation was used to find the optimal number of terminal nodes before pruning.

```r
cv_results   <- cv.tree(tree_model, FUN = prune.misclass)
optimal_size <- cv_results$size[which.min(cv_results$dev)]
pruned_tree  <- prune.misclass(tree_model, best = optimal_size)
```

![CV Raw Tree](charts/02_cv_raw_tree.png)

![Pruned Raw Tree](charts/03_pruned_raw_tree.png)

![Confusion Matrix Raw](charts/04_confusion_raw.png)

**Results — Raw Features Tree:**
- Overall error rate: **12.2%**
- Fraud miss rate: **20.9%** (roughly 1 in 5 fraud complaints misclassified)

---

## Step 2 — PCA, then Decision Tree

With 1,536 correlated embedding dimensions, the tree struggles to find clean splits. PCA extracts uncorrelated components and keeps those that together explain at least 50% of the total variance.

```r
pca_result     <- prcomp(embedding_features, center = TRUE, scale = TRUE)
PVE            <- pca_result$sdev^2 / sum(pca_result$sdev^2)
num_components <- which(cumsum(PVE) >= 0.5)[1]  # 29 components
```

![PCA Variance](charts/05_pca_variance.png)

29 principal components are enough to capture 50% of the variance in the 1,536-dimensional embedding space. The tree is then trained on these 29 features instead of the original 1,536.

```r
train_pca <- data.frame(
  issue = CFPB_train$issue,
  pca_result$x[, 1:num_components]
)
tree_pca         <- tree(issue ~ ., data = train_pca)
pruned_tree_pca  <- prune.misclass(tree_pca, best = optimal_size_pca)
```

![CV PCA Tree](charts/06_cv_pca_tree.png)

![Pruned PCA Tree](charts/07_pruned_pca_tree.png)

![Confusion Matrix PCA](charts/08_confusion_pca.png)

**Results — PCA Features Tree:**
- Overall error rate: **3.3%**
- Fraud miss rate: **4.0%** (roughly 1 in 25)

---

## Results Summary

| Model | Overall Error | Fraud Miss Rate |
|---|---|---|
| Tree on raw embeddings (1,536 features) | 12.2% | 20.9% |
| Tree on PCA features (29 components) | **3.3%** | **4.0%** |

PCA cuts the overall error by roughly 4x. The improvement comes from dimensionality reduction: rather than navigating 1,536 correlated axes, the tree splits on 29 orthogonal directions that capture the genuine structure in the embedding space.

---

## PCA Scatter Plot

The first two principal components plotted against each other, colored by complaint category. Visible separation between the three types confirms that the embedding space does encode meaningful differences between mortgage, credit report, and fraud complaints.

![PCA Scatter](charts/09_pca_scatter.png)

---

## How to Reproduce

1. Clone the repo and place `CFPB_Complaints.csv` in the root directory.
2. Install packages if needed:

```r
install.packages(c("tm", "wordcloud", "tree"))
```

3. Run `generate_charts.R` to produce all charts, then `cfpb_classification.R` for the full analysis.

**R version:** 4.3+  
**Key packages:** `tm`, `wordcloud`, `tree`

---

## File Structure

```
cfpb-complaint-classifier/
├── cfpb_classification.R     # full analysis script
├── generate_charts.R         # saves all plots as PNGs
├── charts/                   # generated chart images
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

> `CFPB_Complaints.csv` is excluded from the repo due to file size (56 MB). The dataset can be sourced from the CFPB public complaints database.
