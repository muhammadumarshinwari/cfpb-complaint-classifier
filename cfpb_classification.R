#######################################################
# Classifying CFPB complaints using complaints text and OpenAI Embeddings
#######################################################

CFPB <-read.csv("CFPB Complaints.csv")

# Check for dimensions and column names
dim(CFPB)
names(CFPB)
View(CFPB)

# The first column is the raw text of the complaint, and the second column is the issue category. 
# The third column is the number of tokens in the complaint text. 
# The remaining columns are the embedding features extracted from the complaint text.

#######################################################
# Exercise 0: Creating a world cloud of the most common words in the complaints
#######################################################

# We will use two new packages "tm" and "wordcloud" to create a word cloud of the most common words in the complaints.
library(tm)
library(wordcloud)
# create a corpus from the raw text column
corpus <- Corpus(VectorSource(CFPB$message))
# preprocess the text: convert to lowercase, remove punctuation, remove numbers, remove stopwords,
# remove redactions (XX, XXX, etc.), and strip whitespace
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, removeWords, c("money", "bank", "
account", "credit", "card", "loan", "payment", "service", "company", "customer", "issue"))
corpus <- tm_map(corpus, removeWords, c("xxxx", "xxx", "xx", "xxxxx", "xxxxxx"))
corpus <- tm_map(corpus, stripWhitespace)
# create a term-document matrix
tdm <- TermDocumentMatrix(corpus)
m <- as.matrix(tdm)
v <- sort(rowSums(m), decreasing = TRUE)
df <- data.frame(word = names(v), freq = v, row.names = NULL)
# drop redaction-like tokens: "x", "xx", "xxx", etc.
df <- subset(df, !grepl("^x+$", word))
# create a wordcloud of the most frequent words in the complaints
dev.new()
wordcloud(words = df$word, freq = df$freq, min.freq = 20
          , max.words = 100, random.order = FALSE, rot.per = 0.35,
          colors = brewer.pal(8, "Dark2"))

#######################################################
# Exercise 1: Building a classification tree on the embedding features to predict the issue category
#######################################################


#### 
# Pre-processing steps:
####

# Step 1: convert the "issue" column to a categorical variable (factor in R)
CFPB$issue <- factor(CFPB$issue)

# Split the data into training and testing sets (80/20 split)
set.seed(1)  # for reproducibility
trainindex <- sample(nrow(CFPB), size = 0.8 * nrow(CFPB))
CFPB_train <- CFPB[trainindex, ]
CFPB_test <- CFPB[-trainindex, ]

# Step 2: remove the "message" and "num_tokens" columns from  training and testing sets

CFPB_train <- CFPB_train[, -c(1, 3)]  # remove
CFPB_test <- CFPB_test[, -c(1, 3)]  # remove



# Build a tree model to predict the issue category based on the embedding features
library(tree)
tree_model <- tree(issue ~ ., data = CFPB_train)
summary(tree_model)

# Perform cross-validation to determine the optimal tree size
cv_results <- cv.tree(tree_model, FUN = prune.misclass)

# Plot the cross-validation results
dev.new()
plot(cv_results$size, cv_results$dev, type = "b",
     xlab = "Tree Size (Number of Terminal Nodes)",
     ylab = "Cross-Validation Misclassification Deviance",
     main = "Cross-Validation for Tree Size Selection")
# Choose the optimal tree size (the one with the minimum error)
optimal_size <- cv_results$size[which.min(cv_results$dev)]
print(paste("Optimal tree size:", optimal_size))
# Prune the tree to the optimal size
pruned_tree <- prune.misclass(tree_model, best = optimal_size)

# plot the pruned tree
dev.new()
plot(pruned_tree)
text(pruned_tree, pretty = 0)


# Evaluate the pruned tree on the test set
predictions <- predict(pruned_tree, newdata = CFPB_test, type = "class")
# Calculate the confusion matrix and overall error rate
Actual.class <- CFPB_test$issue
table(Actual.class, predictions)
# compute overall error rate
overall.error <- sum(predictions != Actual.class)/nrow(CFPB_test)
overall.error # 0.1216667


# Compute the rate of missing fraud 
fraud_idx <- which(CFPB_test$issue == "fraud")
missing_fraud_rate <- sum(predictions[fraud_idx] != "fraud")
missing_fraud_rate <- missing_fraud_rate / length(fraud_idx)
missing_fraud_rate # 0.2089552


#######################################################
# Exercise 2: Building a classification tree on the PCA features to predict the issue category
#######################################################

# Now, we want to do PCA and build a classification tree on the PCA features. 
# We will keep PCA components that contain 50% of the variance.
# Perform PCA on the embedding features (excluding the issue column)
embedding_features <- CFPB_train[, -1]  # exclude the issue column
pca_result <- prcomp(embedding_features, center = TRUE, scale = TRUE)

# Calculate the cumulative proportion of variance contained

total_var <-sum(pca_result$sdev^2)
PVE <-pca_result$sdev^2/total_var
cumsum(PVE)

# Determine the number of components needed to explain at least 50% of the variance
num_components <- which(cumsum(PVE) >= 0.5)[1]
print(paste("Number of PCA components to retain:", num_components))
# Create a new data frame with the retained PCA components and the issue variable
train_pca <- data.frame(
  issue = CFPB_train$issue,
  pca_result$x[, 1:num_components]  # retain only the selected PCA components
)
# Build a tree model using the PCA features
tree_pca <- tree(issue ~ ., data = train_pca)
summary(tree_pca)
# Perform cross-validation to determine the optimal tree size for the PCA tree
cv_pca <- cv.tree(tree_pca, FUN = prune.misclass)
# Plot the cross-validation results for the PCA tree
dev.new()
plot(cv_pca$size, cv_pca$dev, type = "b",
     xlab = "Tree Size (Number of Terminal Nodes)",
     ylab = "Cross-Validation Misclassification Deviance",
     main = "Cross-Validation for PCA Tree Size Selection")
# Choose the optimal tree size for the PCA tree; 
# if there are multiple sizes with the same minimum deviance, we can choose the smallest tree for simplicity
min_dev_pca <- min(cv_pca$dev)
idx_min_pca <- which(cv_pca$dev == min_dev_pca)
optimal_size_pca <- min(cv_pca$size[idx_min_pca])
print(paste("Optimal tree size for PCA tree:", optimal_size_pca))
# Prune the PCA tree to the optimal size
pruned_tree_pca <- prune.misclass(tree_pca, best = optimal_size_pca)
# plot the pruned PCA tree
dev.new()
plot(pruned_tree_pca)
text(pruned_tree_pca, pretty = 0)


# Evaluate the pruned PCA tree on the test set
# First, we need to apply the same PCA transformation to the test set
embedding_features_test <- CFPB_test[, -1]  # exclude the issue column
pca_test <- predict(pca_result, newdata = embedding_features_test)
test_pca <- data.frame(
  issue = CFPB_test$issue,
  pca_test[, 1:num_components]  # retain only the selected PCA components
)
# Now we can make predictions using the pruned PCA tree
predictions_pca <- predict(pruned_tree_pca, newdata = test_pca, type = "class")

# Calculate the confusion matrix and overall error rate for the PCA tree
Actual.class_pca <- test_pca$issue
table(Actual.class_pca, predictions_pca)
# compute overall error rate for the PCA tree
overall.error_pca <- sum(predictions_pca != Actual.class_pca)/nrow(test_pca)
overall.error_pca # 0.03333333
# Compute the rate of missing fraud for the PCA tree 
fraud_idx_pca <- which(test_pca$issue == "fraud")
missing_fraud_rate_pca <- sum(predictions_pca[fraud_idx_pca] != "fraud")
missing_fraud_rate_pca <- missing_fraud_rate_pca / length(fraud_idx_pca)
missing_fraud_rate_pca # 0.039801

# Create a table to summarize the results of the pruned tree on raw features and the pruned tree on PCA features
results_summary <- data.frame(
  Model = c("Raw-features tree", "PCA-features tree"),
  Overall_Error_Rate = c(overall.error, overall.error_pca),
  Missing_Fraud_Rate = c(missing_fraud_rate, missing_fraud_rate_pca))

results_summary

# Now visualize the first two PCA components and color the points by the issue category to see if there is any separation between the categories in the PCA space.
dev.new()
plot(pca_result$x[, 1], pca_result$x[, 2], col =
     factor(CFPB_train$issue), pch = 16,
     xlab = "PCA Component 1", ylab = "PCA Component 2",
     main = "PCA of Embedding Features")
legend("topleft", legend = levels(CFPB_train$issue), col =1:length(levels(CFPB_train$issue)), pch = 16)


