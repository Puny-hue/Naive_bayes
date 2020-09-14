# 1) Prepare a classification model using Naive Bayes for salary data

## Training data ##

salary_train <- SalaryData_Train
View(salary_train)
summary(salary_train)
str(salary_train)
names(salary_train)
levels(salary_train$Salary)

# Test Data
salary_test <- SalaryData_Test
View(salary_test)
str(salary_test)

#Graphical representation
barplot(table(as.factor(salary_train[,14]),as.factor(salary_train[,2])),legend=c("<=50K",">50"))

# Naive Bayes
install.packages("e1071")        
library(e1071)
model<-naiveBayes(salary_train$Salary~., data = salary_train)
summary(model)
model$levels
model$apriori
model$isnumeric
model$tables
model$call
pred<- predict(model, newdata = salary_test[,14])


# Accuracy

Accuracy <- mean(pred==salary_test$Salary)
#0.819

#crosstable
library(gmodels)
ct <- CrossTable(x = salary_test$Salary, y= pred,prop.chisq = FALSE)

# confusion Matrix
library(caret)
confusion <- confusionMatrix(salary_test$Salary,pred)
x<- as.data.frame(pred)
x<- cbind(salary_test$Salary,x)
View(x)


#Accuracy

Accuracy <- sum(diag(ct$t))/sum(ct$t)


# 2) Building a Naive Bayes model on the data set for classifying the ham and spam

sms_raw <- sms_raw_NB_1_
View(sms_raw)
sms_raw$type <- factor(sms_raw$type)

# Examine the type variable more carefully
str(sms_raw$type)
table(sms_raw$type)

#Graphical representation
barplot(table(sms_raw$type))

#Build a carpus using the text mining (tm) package
install.packages("tm")
library(tm)
sms_corpus <- Corpus(VectorSource(sms_raw$text))
sms_corpus <- tm_map(sms_corpus, function(x) iconv(enc2utf8(x), sub = 'byte'))
class(sms_corpus)

# clean up the corpus using tm_map()
Corpus_clean <- tm_map(sms_corpus, tolower)
Corpus_clean<- tm_map(Corpus_clean, removeNumbers)
Corpus_clean<- tm_map(Corpus_clean, removeWords, stopwords())
Corpus_clean<- tm_map(Corpus_clean, removePunctuation)
Corpus_clean<- tm_map(Corpus_clean, stripWhitespace)

# create a document-term sparse matrix
sms_dtm <- DocumentTermMatrix(Corpus_clean)
class(sms_dtm)

# creating Training and test datasets
library(caret)

sms_raw_train <- sms_raw[1:4169,]
sms_raw_test <- sms_raw[4170:5559,]

sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test <- sms_dtm[4170:5559,]

sms_corpus_train <- Corpus_clean[1:4169]
sms_corpus_test <- Corpus_clean[4170:5559]

#check that the proportion of spam is similar
prop.table(table(sms_raw_train$type))
prop.table(table(sms_corpus_test$type))

# Indicator features for frequent words
# Dictionary of words which are used more than 5 times
 
sms_dict <- findFreqTerms(sms_dtm_train, 5)
sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary = sms_dict))
sms_test  <- DocumentTermMatrix(sms_corpus_test, list(dictionary = sms_dict))
inspect(sms_corpus_train[1:100])

# convert counts to a factor
# custom function: if a word is used more than 0 times then mention 1 else mention 0
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
# Margin = 2 is for columns
# Margin = 1 is for rows
sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_test, MARGIN = 2, convert_counts)
View(sms_train)

##  Training a model on the data ----
install.packages("e1071")
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)
sms_classifier
##  Evaluating model performance ----
sms_test_pred <- predict(sms_classifier, sms_test)
table(sms_test_pred)
prop.table(table(sms_test_pred))


library(gmodels)
CrossTable(sms_test_pred, sms_raw_test$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
confusionMatrix(sms_test_pred, sms_raw_test$type)
#Accuracy
mean(sms_test_pred==sms_raw_test$type) 
sms_raw_test$type

