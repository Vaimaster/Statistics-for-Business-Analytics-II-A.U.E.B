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
require(glmnet)
library(aod)

setwd('C:/Users/User/Desktop/Μεταπτυχιακό/2) Statistics for Business Analytics II/0) Graded Assignments/Project I 2022-2023')
getwd()

training_dataset <- read_excel("churn.xls",sheet='churn')

training_dataset$Churn <- as.factor(training_dataset$Churn)
levels(training_dataset$Churn) <- list("No" = "0", "Yes" = "1")
training_dataset$`Int'l Plan` <- as.factor(training_dataset$`Int'l Plan`)
levels(training_dataset$`Int'l Plan`) <- list("No" = "0", "Yes" = "1")
training_dataset$`VMail Plan` <- as.factor(training_dataset$`VMail Plan`)
levels(training_dataset$`VMail Plan`) <- list("No" = "0", "Yes" = "1")
training_dataset$State <- as.factor(training_dataset$State)
training_dataset$`Area Code` <- as.factor(training_dataset$`Area Code`)
training_dataset$Gender <- as.factor(training_dataset$Gender)

# Checking for NAs 
sum(apply(training_dataset,2, is.nan))
sum(apply(training_dataset,2, is.na))
lapply(training_dataset, function(x){length(which(is.na(x)))})

# Exploratory training_dataset Analysis
str(training_dataset)

library(psych)
index <- sapply(training_dataset, class) == "numeric" | sapply(training_dataset, class) == "integer" 
training_dataset_num <- training_dataset[,index]
training_dataset_num[] <- sapply(training_dataset_num, as.numeric)
str(training_dataset_num)
training_dataset_fac <- training_dataset[,!index]
str(training_dataset_fac)
summary(training_dataset)

# Visual Analysis for numerical variables
library(Hmisc)
hist.data.frame(training_dataset_num)

# Visual Analysis for factor variables
training_dataset_fac_2_levels <- training_dataset_fac[,-c(4,5)]

par(mfrow=c(1,1)); n <- nrow(training_dataset_fac_2_levels)
barplot(sapply(training_dataset_fac_2_levels,table)/n, horiz=T, las=1, col=2:3, cex.names=1, mgp = c(3, 0, 0))
legend('topright', fil=2:3, legend=c('No','Yes','Female','Male'), ncol=2, bty='n',cex=0.5)

training_dataset_fac_mul_lvls <- training_dataset_fac[,-c(2,3,6)]
training_dataset_fac_mul_lvls_Churn_0<-training_dataset_fac_mul_lvls[training_dataset_fac_mul_lvls$Churn=='No',]
training_dataset_fac_mul_lvls_Churn_1<-training_dataset_fac_mul_lvls[training_dataset_fac_mul_lvls$Churn=='Yes',]
par(mfrow=c( 3,2)); n <- nrow(training_dataset_fac_mul_lvls_Churn_0)
for (i in 1:3){
  barplot(table(training_dataset_fac_mul_lvls_Churn_0[i]), las=1, col=2:5, cex.names=1,main=names(training_dataset_fac_mul_lvls_Churn_0)[i])
  barplot(table(training_dataset_fac_mul_lvls_Churn_1[i]), las=1, col=2:5, cex.names=1,main=names(training_dataset_fac_mul_lvls_Churn_1)[i])}

# Correlation Matrix
par(mfrow=c(1,1))
library(corrplot)
corrplot(round(cor(training_dataset_num), 2), tl.cex=0.5)

# Start with Models
# Plan is to first do a LASSO to get a subset of Variables that matter and then go for step-wise processes
# Start with a model that includes everything 

mylogit <- glm(Churn ~ ., data = training_dataset, family = "binomial")

summary(mylogit)

#LASSO in order to have less variables in our model

X = model.matrix(mylogit)[]
lasso <- cv.glmnet(X, training_dataset$Churn, alpha = 1, family="binomial")

plot(lasso)

min = coef(lasso, s = "lambda.min")
log(lasso$lambda.min)
lasso$lambda.min
lse = coef(lasso, s = "lambda.1se")
log(lasso$lambda.1se)
lasso$lambda.1se

plot(lasso$glmnet.fit, xvar = "lambda")
abline(v=log(c(lasso$lambda.min, lasso$lambda.1se)), lty =2)

selected = min[min[,1]!=0,]
selectedNames = c(names(selected)[-1],"Churn")
collapsedNames = paste(selectedNames, collapse= " ")

existsIn = function (item, array, arrayAsString){
  if(item %in% array | grepl(item, arrayAsString)){
    return(item)
  }
}

newSelectedNames = sapply(colnames(training_dataset), existsIn, selectedNames, collapsedNames)
newSelectedNames = newSelectedNames[!sapply(newSelectedNames, is.null)]

library(dplyr)
lassoModel = glm(Churn ~ ., data = select(training_dataset,names(newSelectedNames)), family = "binomial")
summary(lassoModel)

# Stepwise Method

aicmodel1 <- step(lassoModel,trace=TRUE, direction = 'both')
summary(aicmodel1)

# Final Model

finalModel1 = glm(Churn ~ `Eve Mins` + `CustServ Calls` + `Int'l Plan` + `VMail Plan` + `Day Charge` + `Night Charge` + `Intl Calls` + `Intl Charge`, data = training_dataset, family='binomial')

summary(finalModel1)

vcov(finalModel1)

for (i in 1:(finalModel1$rank)){
print(wald.test(b = coef(finalModel1), Sigma = vcov(finalModel1), Terms = i))
}
# Low p-value, statistically important

# Goodness of Fit
with(finalModel1, pchisq(deviance, df.residual, lower.tail = FALSE)) # just checking that it's better than the null 

confint(finalModel1) # With a 95% confidence we don't necessarily reject H0 if 0 is included 

with(finalModel1, null.deviance - deviance)
with(finalModel1, df.null - df.residual)
with(finalModel1, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE)) # Significant diff between our model and null model

df_final<-finalModel1$model
par(mfrow = c(3,3))
for (i in 2:ncol(df_final)){
  plot(df_final[,i], resid(finalModel1, type = 'pearson'), ylab = 'Residuals (Pearson)', xlab = names(df_final)[i], cex.lab = 1.5, cex.axis = 1.5, pch = 16, col = 'blue', cex = 1.5)
}

par(mfrow = c(2,2))
plot(finalModel1)

par(mfrow = c(3,3))
for (i in 2:ncol(df_final)){
  plot(df_final[,i], resid(finalModel1, type = 'deviance'), ylab = 'Residuals (Deviance)', xlab = names(df_final)[i], cex.lab = 1.5, cex.axis = 1.5, pch = 16, col = 'red', cex = 1.5)
}

with(summary(finalModel1), 1 - deviance/null.deviance)
library(pscl)
pR2(finalModel1)['McFadden']