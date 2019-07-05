tweets_raw <- read.csv("tweets.csv", stringsAsFactors = FALSE)
str(tweets_raw)
head(tweets_raw)
library(descr)
str(tweets_raw$label)
table(tweets_raw$label)
freq(tweets_raw$label)
library(tm)
tweets_corpus <- VCorpus(VectorSource(tweets_raw$tweet))
print(tweets_corpus)
inspect(tweets_corpus[1:2])
as.character(tweets_corpus[[1]])
lapply(tweets_corpus[1:2], as.character)
tweets_corpus_clean <- tm_map(tweets_corpus, content_transformer(tolower))
tweets_corpus_clean <- tm_map(tweets_corpus_clean, removeNumbers)
tweets_corpus_clean <- tm_map(tweets_corpus_clean, removeWords, stopwords())

# remove punctuation
tweets_corpus_clean <- tm_map(tweets_corpus_clean, removePunctuation) 

lapply(tweets_corpus[1:3], as.character)


library(SnowballC)

tweets_corpus_clean <- tm_map(tweets_corpus_clean, stemDocument)

as.character(tweets_corpus[[1]])
as.character(tweets_corpus_clean[[1]])

# eliminate not needed whitespace
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

# examine the final clean corpus
lapply(tweets_corpus[1:3], as.character)
lapply(tweets_corpus_clean[1:3], as.character)

# create a document-term sparse matrix
tweets_dtm <- DocumentTermMatrix(Corpus(VectorSource(tweets_corpus_clean)))

class(tweets_dtm)
as.character(tweets_dtm[[1]])

# Create training and test data sets
library(caTools)
set.seed("123")

split = sample.split(tweets_dtm, SplitRatio = 0.7)
tweets_dtm_train <- subset(tweets_dtm, split == T)
tweets_dtm_test <- subset(tweets_dtm, split == F)

# Create training and test data sets for label
split = sample.split(tweets_raw,SplitRatio = 0.7)
tweets_train_label <- subset(tweets_raw$label, split == T)
tweets_test_label <- subset(tweets_raw$label, split == F)

# check that the proportion of hate tweets are similar
prop.table(table(tweets_train_label))
prop.table(table(tweets_test_label))

# word cloud visualization
library(wordcloud)
library(NLP)
wordcloud(tweets_corpus, min.freq = 50, random.order = FALSE)

# subset the training data into Normal and Abnormal groups
Normal <- subset(tweets_raw, label == 0)
Abnormal  <- subset(tweets_raw, label == 1)

wordcloud(Normal$tweet, max.words = 100, scale = c(3, 0.5))
wordcloud(Abnormal$tweet, max.words = 100, scale = c(3, 0.5))

tweets_dtm_freq_train <- removeSparseTerms(tweets_dtm_train, 0.999)
tweets_dtm_freq_train

# save frequently-appearing terms to a character vector
tweets_freq_words <- findFreqTerms(tweets_dtm_train, 5)
str(tweets_freq_words)

# create DTMs with only the frequent terms
tweets_dtm_freq_train <- tweets_dtm_train[ , tweets_freq_words]
tweets_dtm_freq_test <- tweets_dtm_test[ , tweets_freq_words]

# convert counts to a factor
convert_counts <- function(x) 
  
{
  x <- ifelse(x > 0, "Yes", "No")
}

# apply() convert_counts() to columns of train/test data
tweets_train <- apply(tweets_dtm_freq_train, MARGIN = 2, convert_counts)
tweets_test  <- apply(tweets_dtm_freq_test, MARGIN = 2, convert_counts)

# Training a model on the data ----
library(e1071)
tweets_classifier <- naiveBayes(tweets_train, tweets_train_labels)

#Evaluating model performance ----
tweets_test_pred <- predict(tweets_classifier, tweets_test)

library(gmodels)
CrossTable(tweets_test_pred, tweets_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

# Improving model performance ----
tweets_classifier2 <- naiveBayes(tweets_train, tweets_train_labels, laplace = 1)
tweets_test_pred2 <- predict(tweets_classifier2, tweets_test)
CrossTable(tweets_test_pred2, tweets_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
