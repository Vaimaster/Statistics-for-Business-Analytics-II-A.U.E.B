library("readxl")
library('pgmm')
library('nnet')
library('class')
library('e1071')	#first install class and then this one
library('penalizedLDA')
library('MASS')
library('heplots')
library('tree')
library('mclust')
library(aod)
library(dplyr)
library(janitor)

my.statistics = function(Actual,Predicted) {
  confusion.table = table(Actual=Actual,Predicted=Predicted)
  output = list(confusion.table=confusion.table)
  TN = confusion.table[1]
  FN = confusion.table[2]
  FP = confusion.table[3]
  TP = confusion.table[4]
  output$accuracy = (TP+TN)/sum(confusion.table)
  output$precission = (TP)/(TP+FP)
  output$sensitivity = (TP)/(TP+FN)
  output$specificity = (TN)/(TN+FP)
  output$FPR = (FP)/(TN+FP)
  output$ARI = adjustedRandIndex(Predicted, Actual)
  
  return(output)
}

#setwd() # <-- set your working directory here
getwd()

churn_data <- read_excel("churn.xls",sheet='churn')

churn_data$Churn <- factor(churn_data$Churn)
levels(churn_data$Churn) <- list("No" = "0", "Yes" = "1")
churn_data$`Int'l Plan` <- factor(churn_data$`Int'l Plan`)
levels(churn_data$`Int'l Plan`) <- list("No" = "0", "Yes" = "1")
churn_data$`VMail Plan` <- factor(churn_data$`VMail Plan`)
levels(churn_data$`VMail Plan`) <- list("No" = "0", "Yes" = "1")
churn_data$State <- factor(churn_data$State)
churn_data$`Area Code` <- factor(churn_data$`Area Code`)
churn_data$Gender <- factor(churn_data$Gender)

#Feature Enrichment;
# The features like total day calls and total eve calls measure frequency of usage whereas
# features such as total day minutes and total eve minutes measure volume of usage.
# Another interesting feature to look at would be the average minutes per call.
# We can measure the average by dividing the total minutes by total calls, for example,
# the feature average minutes per day call = total day minutes / total day calls and 
# similarly,average minutes per eve call = total eve minutes/ total eve calls.
churn_data$avg.minute.day <- churn_data$`Day Mins`/churn_data$`Day Calls`
churn_data$avg.minute.eve <- churn_data$`Eve Mins`/churn_data$`Eve Calls`

churn_data$avg.minute.night <- churn_data$`Night Mins`/churn_data$`Night Calls`
churn_data$avg.minute.intl <- churn_data$`Intl Mins`/churn_data$`Intl Calls`

churn_data[is.na(churn_data)] <- 0
churn_data <- churn_data %>% clean_names(., case= c('upper_camel'))

#churn_data <- sapply(churn_data, as.character)
# churn_data <- as.data.frame(sapply(churn_data, as.numeric))
# churn_data$Churn <- churn_data$Churn-1
# churn_data$`Int'l Plan` <- churn_data$`Int'l Plan`-1
# churn_data$`VMail Plan` <- churn_data$`VMail Plan`-1
# churn_data$State <- churn_data$State-1
# churn_data$`Area Code` <- churn_data$`Area Code`-1
# churn_data$Gender <- churn_data$Gender-1
#View(churn_data)

# Get some good predictive covariates from the first project's step

mylogit = glm(Churn ~ . , data = churn_data, family = "binomial")

summary(mylogit)

X = model.matrix(mylogit)[]
lasso = cv.glmnet(X, churn_data$Churn, alpha = 1, family="binomial")

plot(lasso)

min = coef(lasso, s = "lambda.min")
lse = coef(lasso, s = "lambda.1se")

plot(lasso$glmnet.fit, xvar = "lambda")
abline(v=log(c(lasso$lambda.min, lasso$lambda.1se)), lty =2)

selected = min[min[,1]!=0,]
selectedNames = c(names(selected)[-1],'Churn')

# selectedNames = c(names(selected)[-1],"hilary_elected") -> Try removing all states
churn_data_lasso = churn_data[, which(names(churn_data) %in% selectedNames)]

summary(churn_data_lasso)

# This is for Variable selection for classification
library(mlbench)
library(caret)

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(churn_data[,-8], churn_data$Churn, sizes=10:20, rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors <-predictors(results)
# plot the results
plot(results, type=c("g", "o"))
predictors <- append(predictors,'Churn')

churn_data_rfe <- churn_data[,predictors]

churn_data_class_num <- select_if(churn_data_class, is.numeric)
churn_data_class_num$Churn <- as.numeric(churn_data_class$Churn)-1
churn_data_class_num$IntlPlan <- as.numeric(churn_data_class$IntlPlan)-1
churn_data_class_num$VMailPlan <- as.numeric(churn_data_class$VMailPlan)-1

# Creating train and test sets
set.seed(123)
trainIndex <- createDataPartition(churn_data_class_num$Churn, p = 0.8, list = FALSE)
trainingclass <- churn_data_class_num[trainIndex, ]
testingclass <- churn_data_class_num[-trainIndex, ]
# traininglasso <- churn_data_lasso[trainIndex, ]
# testinglasso <- churn_data_lasso[-trainIndex, ]

# K-nn
library('class')
#Scaled01ChurnData[,20] <- as.numeric(Scaled01ChurnData[,20])-1
# 
i=1
k.optm=1
for (i in seq(from=1, to=61, by=2)){
  knn.mod <- knn(train = trainingclass[,-16], test = testingclass[,-16], cl = trainingclass[,16], k = i)
  k.optm[i] <- 100 * sum(testingclass[,16] == knn.mod)/NROW(testingclass[,16])
  k=i
  cat(k,'=',k.optm[i],'')
}
plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")
# As we can see from the graph, only 7nn is the best with 89.04 Accuracy

knn7<-knn(train = trainingclass[,1:17], test = testingclass[,1:17], cl = trainingclass[,18], k = which.max(k.optm))
table(testingclass[,18],knn7)
# Scaled01ChurnData[,20] <- as.factor(Scaled01ChurnData[,20])
# levels(Scaled01ChurnData[,20]) <- list("No" = "0", "Yes" = "1")
# levels(knn3) <- list("No" = "0", "Yes" = "1")
confusionMatrix(Scaled01ChurnData[,20],knn7)



library('nnet')
library(scales)
# scale the data to [0, 1] range

ChurnNumData <- churn_data%>% select_if(is.numeric)

# Correlation Matrix
par(mfrow=c(1,1))
library(corrplot)
corrplot(round(cor(ChurnNumData), 2), tl.cex=0.5)

Scaled01ChurnData <- as.data.frame(rescale(as.matrix(ChurnNumData)))
Scaled01ChurnData <- Scaled01ChurnData %>% mutate(Churn = churn_data$Churn, IntlPlan = churn_data$IntlPlan, VMailPlan = churn_data$VMailPlan, State = churn_data$State, AreaCode = churn_data$AreaCode, Gender = churn_data$Gender)

# Load libraries
library(dplyr)
library(tidyr)
library(caret)
library(glmnet)
library(coefplot)


# Split dataset into training and test sets

set.seed(123)
trainIndexnum <- createDataPartition(Scaled01ChurnData$Churn, p = 0.8, list = FALSE)
trainingnum <- Scaled01ChurnData[trainIndexnum, ]
testingnum <- Scaled01ChurnData[-trainIndexnum, ]
set.seed(123)
trainIndex <- createDataPartition(churn_data$Churn, p = 0.8, list = FALSE)
training <- churn_data[trainIndex, ]
testing <- churn_data[-trainIndex, ]

#%>% 
#  pivot_longer(-Species, names_to = "Variable", values_to = "Value") %>% 
#  pivot_wider(names_from = "Species", values_from = "Value")
# ChurnNumData$Churn <- as.numeric(churn_data$Churn)-1
# ChurnNumData$IntlPlan <- as.numeric(churn_data$IntlPlan)-1
# ChurnNumData$VMailPlan <- as.numeric(churn_data$VMailPlan)-1
# ChurnNumData$State <- as.numeric(churn_data$State)-1
# ChurnNumData$AreaCode <- as.numeric(churn_data$AreaCode)-1
# ChurnNumData$Gender <- as.numeric(churn_data$Gender)-1




#Scaled01ChurnData <- as.data.frame(apply(ChurnNumData, 2, function(x) (x - min(x)) / (max(x) - min(x))))
#library(caret)
# view the scaled data
mult <- multinom(Churn~., Scaled01ChurnData)
summary(mult)
head(mult$fitted.values)
mult.class <- factor(ifelse(mult$fitted.values > 0.5, 1, 0))#, levels = c("No", "Yes"))
levels(mult.class) <- list("No" = "0", "Yes" = "1")
#mult.class<- apply(mult$fitted.values,1,which.max)
table(Scaled01ChurnData$Churn, mult.class)
confusionMatrix(Scaled01ChurnData$Churn, mult.class)

library('e1071')
library(caTools)
library(caret)

set.seed(123)  # Setting Seed
nbm <- naiveBayes(Churn ~ ., data = training)
nbm_pred <- predict(nbm, newdata = testing[,-20])
nbm_cm <- table(testing$Churn, nbm_pred)
confusionMatrix(nbm_cm)



# Load libraries
library(dplyr)
library(tidyr)
library(caret)
library(glmnet)
library(coefplot)


# Split dataset into training and test sets

set.seed(123)
trainIndex <- createDataPartition(churn_data$Churn, p = 0.8, list = FALSE)
training <- churn_data[trainIndex, ]
testing <- churn_data[-trainIndex, ]

# Fit logistic regression model with L1 regularization
model <- cv.glmnet(x = model.matrix(Churn ~ ., data = training)[,-8],
                   y = as.numeric(training$Churn)-1,
                   family = "binomial", alpha = 1, nfolds = 5)

# Extract non-zero coefficients
predictors <- extract.coef(model)[-1,]
predictors <- predictors$Coefficient
#predictors <- gsub("`","",as.character(predictors))
predictors <- gsub("Yes.*", "", predictors)
predictors <- gsub("State.*", "State", predictors)
predictors <- gsub("Gender.*", "Gender", predictors)
predictors <- unique(append(predictors,'Churn'))

# Fit logistic regression model on selected predictors
model <- glm(Churn ~ ., data = training %>% select(all_of(predictors)),
             family = "binomial")

# Predict on test set
predictions <- predict(model, newdata = testing %>% select(all_of(predictors)),
                       type = "response")
#table(testing$Churn, predictions)

predictions <- factor(ifelse(predictions > 0.5, 1, 0))#, levels = c("No", "Yes"))
levels(predictions) <- list("No" = "0", "Yes" = "1")
# table(testing$Churn, predictions)
# length(predictions)
# length(testing$Churn)
# Evaluate performance
confusionMatrix(predictions, testing$Churn)

library(randomForest)

# Fit random forest model
model1 <- randomForest(Churn ~ ., data = training, importance = TRUE)

# Select important predictors
importance <- importance(model1, type = 1)
predictors1 <- row.names(importance)[importance > 0]
predictors1 <- unique(append(predictors1,'Churn'))

# Fit random forest model on selected predictors
model1 <- randomForest(Churn ~ ., data = training %>% select(all_of(predictors1)))

# Predict on test set
predictions <- predict(model1, newdata = testing %>% select(all_of(predictors1)))

# Evaluate performance
confusionMatrix(predictions, testing$Churn)

# compare performance according to Adjusted Rand Index wrt the 
#	true class

allmodels<- cbind(mult.class,nbclass, lda1, mq2, km1, km2, tr, svm.pred)
colnames(allmodels) <- c('M-Logistic', 'Naive Bayes', 
                         'LDA', 'QDA', 'K-means 4', 'K-means 7', 'Tree', 'SVM')
#	compute adjusted Rand Index for all models
ari <- apply(allmodels,2,function(x){adjustedRandIndex(x, wines$Type)} )

par(mar = c(4,6,3,1))
barplot(ari[order(ari, decreasing = T)], 
        horiz=TRUE, las = 2, xlab = 'Adjusted Rand Index', xlim = c(0,1))