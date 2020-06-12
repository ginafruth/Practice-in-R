#Load useful libraries
library(tidyr)
library(caret) #Data partition
library(ggplot2) #Graphics
library(rpart) #Decision tree model
library(rpart.plot) #Decision tree plot
library(e1071) #confusion matrix
library(caTools)
library(randomForest)
library(data.table) #fread


#Load in the Titanic Datasets
setwd("C:\\Users\\RavenGrey\\Desktop\\R")
TitanicTrain <- fread("TitanicTrain.csv",
                      #Convert Character Columns to Factors
                      stringsAsFactors= T,
                      #Checks to ensure variable names are valid
                      check.names= T)

TitanicTest <- fread("TitanicTest.csv", stringsAsFactors = T, check.names = T)

#Take a look at the structure, summary statistics, and the first/last few rows of the datasets
str(TitanicTrain)
summary(TitanicTrain)
head(TitanicTrain)
tail(TitanicTrain)

str(TitanicTest)
summary(TitanicTest)
head(TitanicTest)
tail(TitanicTest)

#COMBINE 2 DATASETS TO CLEAN DATA
#Create new column to distinguish between Train or Test set
TitanicTrain$IsTrainSet <- TRUE
TitanicTest$IsTrainSet <- FALSE

#Check to see that both datasets have the same column names
names(TitanicTrain)
names(TitanicTest)

#create new Column "Survived" that was originally missing from Test set
TitanicTest$Survived <- NA

#Combine 2 datasets
TitanicFull <- rbind(TitanicTrain, TitanicTest)

#Find and fix missing values
summary(TitanicFull)

#Replace 2 missing "Embarked" values with the mode ("S")
TitanicFull[TitanicFull$Embarked=="", "Embarked"] <- "S"
summary(TitanicFull)

#Change Pclass and SibSp to factors
TitanicFull$Pclass <- as.factor(TitanicFull$Pclass)
TitanicFull$SibSp <- as.factor(TitanicFull$SibSp)
str(TitanicFull)
summary(TitanicFull)

#Graphs of the significant variables
ggplot(data = TitanicFull, aes(x=Age)) + geom_histogram() + facet_grid(Pclass~.)
ggplot(data = TitanicFull, aes(x=Age)) + geom_histogram() + facet_grid(Sex~.)
ggplot(data = TitanicFull, aes(x=Age)) + geom_histogram() + facet_grid(SibSp~.)

#Impute Missing Ages
Pclass1 <-TitanicFull[TitanicFull$Pclass == "1", "Age"]
median1 <- median(Pclass1$Age, na.rm = T)

Pclass2 <-TitanicFull[TitanicFull$Pclass == "2", "Age"]
median2 <- median(Pclass2$Age, na.rm = T)

Pclass3 <-TitanicFull[TitanicFull$Pclass == "3", "Age"]
median3 <- median(Pclass3$Age, na.rm = T)

TitanicFull[TitanicFull$Pclass=="1", "PredictAge"] <- 39
TitanicFull[TitanicFull$Pclass=="2", "PredictAge"] <- 29
TitanicFull[TitanicFull$Pclass=="3", "PredictAge"] <- 24

#Replace missing ages with predictions
TitanicFull[is.na(TitanicFull$Age), "Age"] <- TitanicFull[is.na(TitanicFull$Age), "PredictAge"]
summary(TitanicFull)

#Impute missing fare value
medianFare <- median(na.rm = T, TitanicFull$Fare)
medianFare
TitanicFull[is.na(TitanicFull$Fare), "Fare"] <- medianFare
summary(TitanicFull)

#Separate Test and Train Set
TitanicTrain <- TitanicFull[TitanicFull$IsTrainSet == T, ]
TitanicTest <- TitanicFull[TitanicFull$IsTrainSet == F, ]

#Change 'Survived' to factor
TitanicTrain$Survived <- as.factor(TitanicTrain$Survived)
str(TitanicTrain)

#DATA VISUALIZATION
ggplot(data = TitanicTrain, aes(x=Pclass, fill = Survived)) + geom_bar()
ggplot(data = TitanicTrain, aes(x=Sex, fill = Survived)) + geom_bar()
ggplot(data = TitanicTrain, aes(x=Pclass, fill = Survived)) + geom_bar() + facet_grid(Sex~.)

ggplot(data = TitanicTrain, aes(x=Age, fill = Survived)) + geom_histogram()
ggplot(data = TitanicTrain, aes(x=SibSp, fill = Survived)) + geom_bar()
ggplot(data = TitanicTrain, aes(x=Embarked, fill = Survived)) + geom_bar()

ggplot(data = TitanicTrain, aes(x=Fare, fill = Survived)) + geom_histogram()
ggplot(data = TitanicTrain, aes(x=Pclass, y= Fare, color=Survived)) +geom_boxplot() + geom_jitter() + facet_grid(~Survived) +
  ggtitle("Titanic Survival by Class")

#Look at some of the data by table
table(TitanicTrain$Survived)
table(TitanicTrain$Survived, TitanicTrain$Pclass)
table(TitanicTrain$Survived, TitanicTrain$Sex)
table(TitanicTrain$Survived, TitanicTrain$Embarked)

#Make a simple decision tree, just for fun
dt <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = TitanicTrain)
dt
#Look at the decision tree
rpart.plot(dt) #default
rpart.plot(dt, 
           type = 1, #Labels above node
           extra = 101) #Gives the number of entries in each node

#Make a confusion matrix
cutoff <- 0.5
print("Train confusion matrix")
predicted <- predict(dt, type= "prob")[,1]
predicted
predict.final <- as.factor(ifelse(predicted > cutoff, "0", "1"))
predict.final
confusionMatrix(predict.final, factor(TitanicTrain$Survived))


#Create Predictive Model with a random forest
SurvivedEquation <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
SurvivedFormula <- as.formula(SurvivedEquation)

TitanicModel <- randomForest(formula = SurvivedFormula, 
                             data = TitanicTrain, 
                             ntree = 500, 
                             mtry = 3, 
                             nodesize = 0.01 * nrow(TitanicTrain))

#Predict and submit on Kaggle
Survived <- predict(TitanicModel, newdata = TitanicTest)
PassengerId <- TitanicTest$PassengerId
OutputDF <- as.data.frame(PassengerId)
OutputDF$Survived <- Survived

write.csv(OutputDF, "Titanic_Kaggle_Submission.csv", row.names = FALSE)
