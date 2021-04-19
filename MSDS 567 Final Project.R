library(tidyverse)
library(lubridate)
library(imputeTS)
library(here)
library(caret)
library(dplyr)


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


#Scale numeric variables
df$tenure = as.numeric(df$tenure)
ind_numeric <- sapply(df, is.numeric)
df[ind_numeric] <- lapply(df[ind_numeric], scale)

head(df)


#----------------------EDA, Initial Finding-------------------------------------




#----------------------Need of oversampling for unbalanced?---------------------

#serve as baseline, use F1 score or Confusion Matrix
#Oversampling is saved for later if we have time


#----------------------Data Preparation (Train, Validation, Test)---------------














