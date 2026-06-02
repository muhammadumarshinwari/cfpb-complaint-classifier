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

**A few numbers on the text:**
- Average complaint length: 143 tokens
- Shortest complaint: 70 tokens
- Longest complaint: 279 tokens

---

## Approach

### Step 0 — Word Cloud

Before modeling, a word cloud was built to see which terms appear most often across all 3,000 complaints. Standard pre-processing was applied: lowercase conversion, punctuation and number removal, English stopwords, and redaction tokens (XXXX, XXX) that CFPB uses to anonymize personal data.

```r
library(tm)
library(wordcloud)

corpus <- Corpus(VectorSource(CFPB$message))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, removeWords, c("xxxx", "xxx", "xx"))
corpus <- tm_map(corpus, stripWhitespace)

tdm <- TermDocumentMatrix(corpus)
```

Common terms: *account*, *report*, *information*, *received*, *time*, *told* — fairly generic, which is why the raw text alone does not discriminate cleanly between categories without embedding.

---

### Step 1 — Decision Tree on Raw Embedding Features

The 1,536 embedding dimensions were used directly as features. The data was split 80/20 into training and test sets.

```r
set.seed(1)
trainindex <- sample(nrow(CFPB), size = 0.8 * nrow(CFPB))
CFPB_train <- CFPB[trainindex, ]
CFPB_test  <- CFPB[-trainindex, ]

# Remove message and num_tokens columns
CFPB_train <- CFPB_train[, -c(1, 3)]
CFPB_test  <- CFPB_test[, -c(1, 3)]

tree_model <- tree(issue ~ ., data = CFPB_train)
```

Cross-validation was used to find the optimal number of terminal nodes, and the tree was pruned accordingly.

```r
cv_results  <- cv.tree(tree_model, FUN = prune.misclass)
optimal_size <- cv_results$size[which.min(cv_results$dev)]
pruned_tree  <- prune.misclass(tree_model, best = optimal_size)
```

---

### Step 2 — PCA, then Decision Tree

With 1,536 correlated features, a single decision tree struggles to find meaningful splits. PCA was applied to the embedding dimensions to extract uncorrelated components, retaining those that together explain at least 50% of the total variance.

```r
embedding_features <- CFPB_train[, -1]
pca_result <- prcomp(embedding_features, center = TRUE, scale = TRUE)

# Cumulative variance
total_var <- sum(pca_result$sdev^2)
PVE       <- pca_result$sdev^2 / total_var
num_components <- which(cumsum(PVE) >= 0.5)[1]

train_pca <- data.frame(
  issue = CFPB_train$issue,
  pca_result$x[, 1:num_components]
)
```

The same cross-validation and pruning procedure was then repeated on the PCA-reduced training set.

---

## Results

| Model | Overall Error Rate | Fraud Miss Rate |
|---|---|---|
| Tree on raw embeddings | 12.2% | 20.9% |
| Tree on PCA features | **3.3%** | **4.0%** |

The PCA tree cuts the overall error rate by roughly 4x and reduces the fraud miss rate from 1-in-5 to about 1-in-25. The improvement comes from dimensionality reduction: the tree can now split on directions that capture genuine variance in the data rather than trying to navigate 1,536 correlated axes one at a time.

---

## Charts

**1. Word Cloud**
Most frequent terms after stopword removal. Redaction tokens (XXXX) are excluded. Dominated by service and account-related vocabulary shared across all three complaint types.

**2. Cross-Validation — Raw Features Tree**
Misclassification deviance plotted against tree size. Used to identify the optimal number of terminal nodes before pruning.

**3. Pruned Tree — Raw Features**
Final tree structure after pruning to the optimal size selected by cross-validation.

**4. Cross-Validation — PCA Tree**
Same procedure applied to the PCA feature set. Noticeably lower deviance at the optimum compared to the raw features tree.

**5. Pruned Tree — PCA Features**
Final PCA tree. Cleaner structure than the raw features tree — fewer splits needed to achieve much lower error.

**6. PCA Scatter Plot**
Training observations plotted on the first two principal components, colored by issue category. Shows visible separation between the three complaint types in the PCA space, which explains why the PCA tree performs well.

---

## How to Reproduce

1. Clone the repo and open `cfpb_classification.R` in RStudio.
2. Make sure `CFPB_Complaints.csv` is in the same directory.
3. Install the required packages if not already present:

```r
install.packages(c("tm", "wordcloud", "tree"))
```

4. Run the script top to bottom. Each `dev.new()` call opens a new plot window.

**R version used:** 4.3+  
**Key packages:** `tm`, `wordcloud`, `tree`

---

## File Structure

```
cfpb-complaint-classifier/
├── cfpb_classification.R     # full analysis script
├── CFPB_Complaints.csv       # complaint text + OpenAI embeddings (3,000 rows x 1,539 cols)
└── README.md
```
