library(tidyverse)
library(lubridate)
library(imputeTS)
library(here)
library(caret)
library(dplyr)
# for parallel computing
library(parallel)
library(doParallel)


#-------------------------------------------------------------------------------

df = read.csv("telecom_users.csv")



#----------------------Handling NA, Feature Engineering-------------------------

str(df)
summary(df)

#All entries are unique customer's data
df$Duplicated = duplicated(df$customerID)
df$Duplicated = as.factor(df$Duplicated)
levels(df$Duplicated)


#Handle NA value
colSums(is.na(df))

df_na = df %>%
  filter(is.na(df$TotalCharges))

#For all obs with NA in TotalCharges, tenure is 0. As a result, those seems to 
#new customer and TotalCharges will be manually populated as the MonthlyCharges 
df_na$tenure

df$TotalCharges[is.na(df$TotalCharges)] = df$MonthlyCharges[is.na(df$TotalCharges)]
colSums(is.na(df))


#Get rid of the first two columns: X, customer id, Duplicated
df = df[-c(1,2,23)]


#Classify No phone service, No internet service into No
df$MultipleLines[df$MultipleLines == "No phone service"] = "No"
df$OnlineSecurity[df$OnlineSecurity == "No internet service"] = "No"
df$OnlineBackup[df$OnlineBackup == "No internet service"] = "No"
df$DeviceProtection[df$DeviceProtection == "No internet service"] = "No"
df$TechSupport[df$TechSupport == "No internet service"] = "No"
df$StreamingTV[df$StreamingTV == "No internet service"] = "No"
df$StreamingMovies[df$StreamingMovies == "No internet service"] = "No"


#Change categorical variables to correct data types
df[sapply(df, is.character)] = lapply(df[sapply(df, is.character)], as.factor)

df$SeniorCitizen = as.factor(plyr::mapvalues(df$SeniorCitizen, from=c("0","1"), to=c("No", "Yes")))


#----------------------EDA, Initial Finding-------------------------------------

#Continuous Variables
hist(df$tenure)
hist(df$MonthlyCharges)

#Categorical Variables over Churn

ggplot(df, aes(x = gender, fill = Churn))+geom_bar()

#From percentage view, senior citizen are more likely to churn than non-senior citizen
ggplot(df, aes(x = SeniorCitizen, fill = Churn))+geom_bar()

#Avg Monthly cost over senior citizen or not
df%>%
  group_by(SeniorCitizen)%>%
  summarise(mean = mean(MonthlyCharges), n = n())

#Partner
ggplot(df, aes(x = Partner, fill = Churn))+geom_bar()

#Avg Monthly cost over Partner or not
df%>%
  group_by(Partner)%>%
  summarise(mean = mean(MonthlyCharges), n = n())

#Dependents
ggplot(df, aes(x = Dependents, fill = Churn))+geom_bar()

#Avg Monthly cost over Dependents or not
df%>%
  group_by(Dependents)%>%
  summarise(mean = mean(MonthlyCharges), n = n())



ggplot(df, aes(x = StreamingTV , fill = Churn))+geom_bar()
ggplot(df, aes(x = TechSupport , fill = Churn))+geom_bar()

#Most customers prefer e-check, and indeed e-check has highest churn rate. 
ggplot(df, aes(x = PaymentMethod , fill = Churn))+geom_bar()+ scale_x_discrete(guide = guide_axis(n.dodge = 3))

ggplot(df, aes(x = StreamingTV, fill = Churn))+geom_bar()

#Month to month wins
ggplot(df, aes(x = Contract, fill = Churn))+geom_bar() + scale_x_discrete(guide = guide_axis(n.dodge = 3))

#Monthly cost over Churn
ggplot(df, aes(x = MonthlyCharges, fill = Churn))+geom_histogram(binwidth = 10)

#Fiber optic 
ggplot(df, aes(x = InternetService , fill = Churn))+geom_bar() + scale_x_discrete(guide = guide_axis(n.dodge = 3))

df%>%
  group_by(InternetService)%>%
  summarise(mean = mean(MonthlyCharges), n = n())


#Multicollinearity 
source("http://www.sthda.com/upload/rquery_cormat.r")
mydata <- df[, c(5,18,19)]
require("corrplot")
rquery.cormat(mydata)

#Drop TotalCharge, which is MonthlyCharge*tenure
df = df[,-19]

#Scale numeric variables
df$tenure = as.numeric(df$tenure)
ind_numeric = sapply(df, is.numeric)
df[ind_numeric] = lapply(df[ind_numeric], scale)

head(df)
str(df)


#----------------------Data Preparation (Train, Validation, Test)---------------

set.seed(123)
split = createDataPartition(df$Churn,p = 0.7,list = FALSE)
train = df[split,]

#test set, which is considered as unseen data, 
#is saved for the best model after models' performance comparison
test =  df[-split,]

#Validation set
split_again = createDataPartition(train$Churn,p = 0.7,list = FALSE)

train1 = train[split_again,]
validation = train[-split_again,]



#Need of oversampling for unbalanced?

ggplot(train1, aes(Churn, fill = Churn))+geom_bar()


#library(ROSE)
#train <- ovun.sample(Churn~., data=train,
#N=nrow(train), p=0.5, 
#seed=123, method="both")$data

#serve as baseline, use F1 score or Confusion Matrix
#Oversampling is saved for later if we have time

#-------------------Use below---------------------

library(unbalanced)
levels(train1$Churn) <- c(0,1)
levels(test$Churn) <- c(0,1)
levels(validation$Churn) <- c(0,1)
input <- train1[,1:18]
response <- train1$Churn
balance <- ubOver(X=input, Y=response) 
new_train <- cbind(balance$X, balance$Y)
colnames(new_train)[19] <- "Churn"

ggplot(new_train, aes(Churn, fill = Churn))+geom_bar()




#----------------------Modeling (Logistic Regression)---------------------------
set.seed(123)
# define training control 10 fold cross validation
train_control <- trainControl(method = "cv", number = 10, allowParallel = FALSE)
# train the model on training set using logistic Regression
model_logit <- caret::train(Churn ~ .,data = new_train,
               trControl = train_control,
               method = "glm",
               family=binomial())
summary(model_logit)

print(model_logit)

#model_logit$resample


#Use the logit model to make prediction 
predict_logit <- predict(model_logit, newdata = validation)
#Generate Confusion Matrix and F1 Score.
result_logit <- confusionMatrix(data = predict_logit, reference = validation$Churn, mode = "prec_recall")
F1_logit <- result_logit$byClass[7]
result_logit
F1_logit



#----------------------Modeling (LDA)-------------------------------------------

set.seed(123)
model_lda <- caret::train(Churn ~ .,data = new_train, method = "lda",
                 trControl=train_control,
                 verbose = TRUE)
model_lda

predict_lda <- predict(model_lda, newdata = validation)

result_lda <- confusionMatrix(data = predict_lda, reference = validation$Churn, mode = "prec_recall")
F1_lda <- result_lda$byClass[7]
result_lda
F1_lda


#----------------------Modeling (SVM Linear)------------------------------------
set.seed(123)

cluster <- makeCluster(detectCores() - 2)
registerDoParallel(cluster)

train_control_parallel <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

model_svm <- caret::train(Churn ~ .,data = new_train, method = "svmLinear",
                 trControl=train_control_parallel,
                 tuneGrid = expand.grid(C = c(0.01,0.1, 1,5,10,100)),
                 verbose = TRUE)

stopCluster(cluster)
#Pick C = 0.01

predict_svm <- predict(model_svm, newdata = validation)

result_svm <- confusionMatrix(data = predict_svm, reference = validation$Churn, mode = "prec_recall")
F1_svm <- result_svm$byClass[7]
result_svm
F1_svm


#----------------------Modeling (SVM Radial)------------------------------------
set.seed(123)

cluster <- makeCluster(detectCores() - 2)
registerDoParallel(cluster)

train_control_parallel <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

model_svm_Rad <- caret::train(Churn ~ .,data = new_train, method = "svmRadial",
                          trControl=train_control_parallel,
                          tuneLength = 5,
                          verbose = TRUE)

stopCluster(cluster)

model_svm_Rad$bestTune

predict_svm_Rad <- predict(model_svm_Rad, newdata = validation)

result_svm_Rad <- confusionMatrix(data = predict_svm_Rad, reference = validation$Churn, mode = "prec_recall")
F1_svm_Rad <- result_svm_Rad$byClass[7]
result_svm_Rad
F1_svm_Rad




#----------------------Modeling (Random Forest)---------------------------------
set.seed(123)

cluster <- makeCluster(detectCores() - 2)
registerDoParallel(cluster)
train_control_parallel <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

model_rf <- caret::train(Churn ~ .,data = new_train, method = "rf",
                trControl=train_control_parallel,
                tuneGrid = expand.grid(.mtry = c(1:18)),
                metric="Kappa",
                verbose = TRUE)

stopCluster(cluster)
model_rf

predict_rf <- predict(model_rf, newdata = validation)
result_rf <- confusionMatrix(data = predict_rf, reference = validation$Churn, mode = "prec_recall")
F1_rf <- result_rf$byClass[7]
result_rf
F1_rf


#---------Use randomForest to tune tree and # of variable split

set.seed(123)
#Variable Importance Plot
library(randomForest)

#Choose mtry = 15 based on above
model_rf1 <- randomForest(Churn ~ .,data = new_train,mtry =15, importance = TRUE)
print(model_rf1)

pred_rf1 <- predict(model_rf1, validation)

result_rf1 = caret::confusionMatrix(pred_rf1, validation$Churn, mode = "prec_recall")

result_rf1$byClass[7]


#Find optimal tree size, 100 seems sufficient
plot(model_rf1,main="Error as ntree increases")

set.seed(123)

#Final Model for RF
model_rf2 <- randomForest(Churn ~ .,data = new_train,mtry =15, ntree = 100, importance = TRUE)
print(model_rf2)


pred_rf2 <- predict(model_rf2, validation)

result_rf2 = caret::confusionMatrix(pred_rf2, validation$Churn, mode = "prec_recall")

result_rf2$byClass[7]


varImpPlot(model_rf2, main="Variable Importance Plots")







#----------------------Model Comparison (Accuracy, F1 Score)--------------------







#----------------------Final Model on test set (rf mtry=15, ntree = 100)---------
pred_rf_test <- predict(model_rf2, test)

result_test = caret::confusionMatrix(pred_rf_test, test$Churn, mode = "prec_recall")

result_test$byClass[7]


#----------------------LR on test set (better performance on FN)----------------

pred_Logic_test <- predict(model_logit, test)

result_Logic_test = caret::confusionMatrix(pred_Logic_test, test$Churn, mode = "prec_recall")

result_Logic_test$byClass[7]





