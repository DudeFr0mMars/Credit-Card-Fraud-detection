rm(list=ls())

library(readr)
library(dplyr)
library(randomForest)
library(ggplot2)
library(Hmisc)
library(party)
library(caret)
library(MLmetrics)
library(mice)
library(VIM)

set.seed(42)

setwd("path to csv file")
creditData <- read_csv('creditcardFraudMiss.csv')
summary(creditData)
glimpse(creditData)

mice_imputes <- mice(creditData, m=5, maxit=3)
mice_imputes$method
creditDataFinal <- complete(mice_imputes,4)


creditDataFinal$class <- factor(creditDataFinal$class)
random=sample(nrow(creditDataFinal),nrow(creditDataFinal)*0.7)
train<-creditDataFinal[random,]
test<-creditDataFinal[-random,]

train %>%
  select(class) %>%
  group_by(class) %>%
  summarise(count = n()) %>%
  glimpse
test %>%
  select(class) %>%
  group_by(class) %>%
  summarise(count = n()) %>%
  glimpse

rfModel <- randomForest(class ~ . , data = train)
test$predicted <- predict(rfModel, test)


confusionMatrix(test$class, test$predicted)

F1_all <- F1_Score(test$class, test$predicted)
F1_all

options(repr.plot.width=5, repr.plot.height=4)
varImpPlot(rfModel,
           sort = T,
           n.var=10,
           main="Top 10 Most Important Variables")

rfModelTrim1 <- randomForest(class ~  V17, 
                             data = train)

test$predictedTrim1 <- predict(rfModelTrim1, test)

F1_1 <- F1_Score(test$class, test$predictedTrim1)
F1_1

rfModelTrim2 <- randomForest(class ~  V17 + V14, 
                             data = train)

test$predictedTrim2 <- predict(rfModelTrim2, test)

F1_2 <- F1_Score(test$class, test$predictedTrim2)
F1_2

rfModelTrim5 <- randomForest(class ~  V17 + V14 + V14 + V10 + V11, 
                             data = train)

test$predictedTrim5 <- predict(rfModelTrim5, test)

F1_5 <- F1_Score(test$class, test$predictedTrim5)
F1_5

rfModelTrim10 <- randomForest(class ~  V17 + V14 + V12 + V10 + V11 
                              + V16 + V18 + V4 + V9 + V7, 
                              data = train)

test$predictedTrim10 <- predict(rfModelTrim10, test)

F1_10 <- F1_Score(test$class, test$predictedTrim10)
F1_10

numVariables <- c(1,2,5,10,17)
F1_Score <- c(F1_1, F1_2, F1_5, F1_10, F1_all)
variablePerf <- data.frame(numVariables, F1_Score)

options(repr.plot.width=4, repr.plot.height=3)
ggplot(variablePerf, aes(numVariables, F1_Score)) + geom_point() + labs(x = "Number of Variables", y = "F1 Score", title = "F1 Score Performance")

rf10 = randomForest(class ~  V17 + V14 + V12 + V10 + V11 
                    + V16 + V18 + V4 + V9 + V7,  
                    ntree = 1000,
                    data = train)

options(repr.plot.width=6, repr.plot.height=4)
plot(rf10)

options(repr.plot.width=6, repr.plot.height=4)
plot(rf10, xlim=c(0,100))

