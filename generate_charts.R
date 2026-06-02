## generate_charts.R
## Runs all analysis and saves plots as PNGs for the README

setwd("E:/CC/cfpb-complaint-classifier")

CFPB <- read.csv("CFPB_Complaints.csv")

dir.create("charts", showWarnings = FALSE)

# ── Chart 1: Word Cloud ────────────────────────────────────────────────────────
library(tm)
library(wordcloud)

corpus <- Corpus(VectorSource(CFPB$message))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, removeWords, c("money","bank","account","credit","card",
                                        "loan","payment","service","company",
                                        "customer","issue"))
corpus <- tm_map(corpus, removeWords, c("xxxx","xxx","xx","xxxxx","xxxxxx"))
corpus <- tm_map(corpus, stripWhitespace)

tdm <- TermDocumentMatrix(corpus)
m   <- as.matrix(tdm)
v   <- sort(rowSums(m), decreasing = TRUE)
df  <- data.frame(word = names(v), freq = v, row.names = NULL)
df  <- subset(df, !grepl("^x+$", df$word))

png("charts/01_wordcloud.png", width = 900, height = 700, res = 120)
par(mar = c(0, 0, 3, 0))
set.seed(42)
wordcloud(words = df$word, freq = df$freq, min.freq = 20,
          max.words = 100, random.order = FALSE, rot.per = 0.25,
          colors = brewer.pal(8, "Dark2"), scale = c(4, 0.5))
title(main = "Most Frequent Words in CFPB Complaints", cex.main = 1.2, line = 1)
dev.off()
cat("Chart 1 done\n")

# ── Pre-processing ─────────────────────────────────────────────────────────────
CFPB$issue <- factor(CFPB$issue)
set.seed(1)
trainindex  <- sample(nrow(CFPB), size = 0.8 * nrow(CFPB))
CFPB_train  <- CFPB[trainindex, ]
CFPB_test   <- CFPB[-trainindex, ]
CFPB_train  <- CFPB_train[, -c(1, 3)]
CFPB_test   <- CFPB_test[, -c(1, 3)]

# ── Exercise 1: Raw-features tree ─────────────────────────────────────────────
library(tree)
tree_model <- tree(issue ~ ., data = CFPB_train)
cv_results  <- cv.tree(tree_model, FUN = prune.misclass)

# Chart 2: CV for raw tree
png("charts/02_cv_raw_tree.png", width = 800, height = 550, res = 120)
plot(cv_results$size, cv_results$dev, type = "b",
     pch = 19, col = "#2E4057",
     xlab = "Tree Size (Terminal Nodes)",
     ylab = "CV Misclassification Error",
     main = "Cross-Validation: Raw Embedding Features Tree",
     lwd = 2)
optimal_size <- cv_results$size[which.min(cv_results$dev)]
abline(v = optimal_size, col = "#C9A86A", lty = 2, lwd = 2)
legend("topright", legend = paste("Optimal size =", optimal_size),
       col = "#C9A86A", lty = 2, lwd = 2, bty = "n")
dev.off()
cat("Chart 2 done\n")

# Chart 3: Pruned raw tree
pruned_tree <- prune.misclass(tree_model, best = optimal_size)
png("charts/03_pruned_raw_tree.png", width = 1600, height = 1000, res = 130)
par(mar = c(2, 2, 4, 2))
plot(pruned_tree)
text(pruned_tree, pretty = 0, cex = 0.62, col = "#2E4057")
title(main = "Pruned Decision Tree — Raw Embedding Features", cex.main = 1.1)
dev.off()
cat("Chart 3 done\n")

# Evaluate raw tree
predictions   <- predict(pruned_tree, newdata = CFPB_test, type = "class")
Actual.class  <- CFPB_test$issue
overall.error <- sum(predictions != Actual.class) / nrow(CFPB_test)
fraud_idx     <- which(CFPB_test$issue == "fraud")
missing_fraud_rate <- sum(predictions[fraud_idx] != "fraud") / length(fraud_idx)
cat(sprintf("Raw tree — overall error: %.1f%%, fraud miss: %.1f%%\n",
            overall.error*100, missing_fraud_rate*100))

# Chart 4: Confusion matrix raw tree
conf_raw <- table(Actual = Actual.class, Predicted = predictions)
png("charts/04_confusion_raw.png", width = 700, height = 550, res = 120)
par(mar = c(5, 5, 4, 2))
image(1:3, 1:3, t(apply(conf_raw, 1, rev)),
      col = colorRampPalette(c("white", "#2E4057"))(50),
      xaxt = "n", yaxt = "n",
      xlab = "Predicted", ylab = "Actual",
      main = "Confusion Matrix — Raw Features Tree")
axis(1, at = 1:3, labels = colnames(conf_raw), cex.axis = 0.9)
axis(2, at = 1:3, labels = rev(rownames(conf_raw)), cex.axis = 0.9)
for (i in 1:3) for (j in 1:3)
  text(j, 4-i, conf_raw[i,j], col = ifelse(conf_raw[i,j] > 100, "white", "black"), cex = 1.1)
dev.off()
cat("Chart 4 done\n")

# ── Exercise 2: PCA tree ───────────────────────────────────────────────────────
embedding_features <- CFPB_train[, -1]
pca_result <- prcomp(embedding_features, center = TRUE, scale = TRUE)
total_var  <- sum(pca_result$sdev^2)
PVE        <- pca_result$sdev^2 / total_var
num_components <- which(cumsum(PVE) >= 0.5)[1]
cat(sprintf("PCA components for 50%% variance: %d\n", num_components))

# Chart 5: Scree / cumulative variance
png("charts/05_pca_variance.png", width = 800, height = 550, res = 120)
par(mar = c(5, 5, 4, 2))
plot(cumsum(PVE)[1:100], type = "l", lwd = 2, col = "#2E4057",
     xlab = "Number of Principal Components",
     ylab = "Cumulative Proportion of Variance Explained",
     main = "PCA: Cumulative Variance Explained")
abline(h = 0.5, col = "#C9A86A", lty = 2, lwd = 2)
abline(v = num_components, col = "#C9A86A", lty = 2, lwd = 2)
points(num_components, cumsum(PVE)[num_components],
       pch = 19, col = "#C9A86A", cex = 1.5)
legend("bottomright",
       legend = c("Cumulative variance", sprintf("50%% threshold (PC %d)", num_components)),
       col = c("#2E4057", "#C9A86A"), lty = c(1, 2), lwd = 2, bty = "n")
dev.off()
cat("Chart 5 done\n")

# Build PCA tree
train_pca <- data.frame(issue = CFPB_train$issue,
                        pca_result$x[, 1:num_components])
tree_pca  <- tree(issue ~ ., data = train_pca)
cv_pca    <- cv.tree(tree_pca, FUN = prune.misclass)
min_dev_pca     <- min(cv_pca$dev)
idx_min_pca     <- which(cv_pca$dev == min_dev_pca)
optimal_size_pca <- min(cv_pca$size[idx_min_pca])
pruned_tree_pca  <- prune.misclass(tree_pca, best = optimal_size_pca)

# Chart 6: CV for PCA tree
png("charts/06_cv_pca_tree.png", width = 800, height = 550, res = 120)
plot(cv_pca$size, cv_pca$dev, type = "b",
     pch = 19, col = "#2E4057",
     xlab = "Tree Size (Terminal Nodes)",
     ylab = "CV Misclassification Error",
     main = "Cross-Validation: PCA Features Tree",
     lwd = 2)
abline(v = optimal_size_pca, col = "#C9A86A", lty = 2, lwd = 2)
legend("topright", legend = paste("Optimal size =", optimal_size_pca),
       col = "#C9A86A", lty = 2, lwd = 2, bty = "n")
dev.off()
cat("Chart 6 done\n")

# Chart 7: Pruned PCA tree
png("charts/07_pruned_pca_tree.png", width = 1000, height = 700, res = 120)
plot(pruned_tree_pca)
text(pruned_tree_pca, pretty = 0, cex = 0.75)
title(main = "Pruned Decision Tree — PCA Features")
dev.off()
cat("Chart 7 done\n")

# Evaluate PCA tree
embedding_features_test <- CFPB_test[, -1]
pca_test <- predict(pca_result, newdata = embedding_features_test)
test_pca <- data.frame(issue = CFPB_test$issue,
                       pca_test[, 1:num_components])
predictions_pca    <- predict(pruned_tree_pca, newdata = test_pca, type = "class")
Actual.class_pca   <- test_pca$issue
overall.error_pca  <- sum(predictions_pca != Actual.class_pca) / nrow(test_pca)
fraud_idx_pca      <- which(test_pca$issue == "fraud")
missing_fraud_rate_pca <- sum(predictions_pca[fraud_idx_pca] != "fraud") / length(fraud_idx_pca)
cat(sprintf("PCA tree  — overall error: %.1f%%, fraud miss: %.1f%%\n",
            overall.error_pca*100, missing_fraud_rate_pca*100))

# Chart 8: Confusion matrix PCA tree
conf_pca <- table(Actual = Actual.class_pca, Predicted = predictions_pca)
png("charts/08_confusion_pca.png", width = 700, height = 550, res = 120)
par(mar = c(5, 5, 4, 2))
image(1:3, 1:3, t(apply(conf_pca, 1, rev)),
      col = colorRampPalette(c("white", "#2E4057"))(50),
      xaxt = "n", yaxt = "n",
      xlab = "Predicted", ylab = "Actual",
      main = "Confusion Matrix — PCA Features Tree")
axis(1, at = 1:3, labels = colnames(conf_pca), cex.axis = 0.9)
axis(2, at = 1:3, labels = rev(rownames(conf_pca)), cex.axis = 0.9)
for (i in 1:3) for (j in 1:3)
  text(j, 4-i, conf_pca[i,j], col = ifelse(conf_pca[i,j] > 100, "white", "black"), cex = 1.1)
dev.off()
cat("Chart 8 done\n")

# Chart 9: PCA scatter plot (PC1 vs PC2)
colors_map <- c(credit_report = "#2E4057", fraud = "#E07B39", mortgage = "#3A9E82")
point_colors <- colors_map[as.character(CFPB_train$issue)]
png("charts/09_pca_scatter.png", width = 950, height = 680, res = 130)
par(mar = c(5, 5, 4, 2))
plot(pca_result$x[, 1], pca_result$x[, 2],
     col = adjustcolor(point_colors, alpha.f = 0.55),
     pch = 16, cex = 0.65,
     xlab = "Principal Component 1",
     ylab = "Principal Component 2",
     main = "PCA of OpenAI Embeddings — Colored by Complaint Category")
legend("bottomright",
       legend = c("Credit Report", "Fraud", "Mortgage"),
       col = colors_map, pch = 16, pt.cex = 1.4,
       bty = "o", bg = "white", box.col = "grey80",
       cex = 0.95, y.intersp = 1.3)
dev.off()
cat("Chart 9 done\n")

cat("\nAll charts saved to charts/\n")
