## Competition Assignment 

# Load packages
library(plyr)
library(tidyverse)
library(data.table)
library(fst)
library(ranger)
library(tuneRanger)
library(xgboost)
library(caret)
library(viridis)
library(readr)
install.packages("missRanger")
library(missRanger)
holdout_data <- read_csv("input_data/holdout_data.csv")
View(holdout_data)

training_test_data <- read_csv("input_data/trainig_test_data.csv")
View(trainig_test_data)

## Goal: predict Income 

col_input0 = trainig_test_data

names(col_input0)
# First delete all NAs 
col_input1<- na.omit(col_input0) #only 26000 observations out of 47000

# turn variables into factors 
col_input1 <- as.data.frame(unclass(col_input1), stringsAsFactors = TRUE)

col_input1$state <- as.factor(col_input1$state)
col_input1$class <- as.factor(col_input1$class)
col_input1$marriage <- as.factor(col_input1$marriage)
col_input1$race <- as.factor(col_input1$race) 
col_input1$sex <- as.factor(col_input1$sex) 
col_input1$veteran <- as.factor(col_input1$veteran) 
col_input1$region2 <- as.factor(col_input1$region2) 
col_input1$city <- as.factor(col_input1$city) 
col_input1$ethnicity <- as.factor(col_input1$ethnicity)
col_input1$status_last_week <- as.factor (col_input1$status_last_week)
col_input1$class2 <- as.factor(col_input1$class2)  
col_input1$income_a <- as.factor(col_input1$income_a) 
col_input1$income_b <- as.factor(col_input1$income_b)  
col_input1$income_c <- as.factor(col_input1$income_c) 
col_input1$income_d <- as.factor(col_input1$income_d) 
col_input1$union <- as.factor(col_input1$union)
col_input1$industry <- as.factor(col_input1$industry) 
col_input1$child_info <- as.factor (col_input1$child_info) 
col_input1$family_relation <- as.factor(col_input1$family_relation) 
col_input1$occupation <- as.factor(col_input1$occupation) 
col_input1$family_reference <- as.factor(col_input1$family_reference)  
col_input1$education <- as.factor(col_input1$education) 

# estimate Random Forest 
rf_1 = ranger(data = col_input1, dependent.variable.name = "income", importance = "impurity")
class(rf_1)

# use caret package for Cross Validation 
modelLookup("ranger")
control5 = trainControl(method = "cv", number = 5)
control10 = trainControl(method = "cv", number = 10)

# tuning grids with different numbers of variables sampled as candidates for each split
tuning_grid1 = expand.grid(mtry = seq(20, 30, by = 5), splitrule = "variance", min.node.size = 500)
tuning_grid2 = expand.grid(mtry = seq(20, 30, by = 5), splitrule = "variance", min.node.size = seq(100, 500, by = 50))
tuning_grid3 = expand.grid(mtry = seq(10, 20, by = 2), splitrule = "variance", min.node.size = seq(100, 200, by = 2))
# out of RF from our own analysis
tuning_grid_2b = expand.grid(mtry = seq(20, 30, by = 5), splitrule = "variance", min.node.size = seq(4, 8, by = 2))
head(tuning_grid1)
head(tuning_grid2)
head(tuning_grid3)
# mtry: number of variables randomly sampled as candidates at each spli
# min.node.size: minimum number of observations associated with each leaf node >> sets depth of the tree

rf_caret1 = train(data = col_input1, income ~ ., method = "ranger", trControl = control5, tuneGrid = tuning_grid1, importance = "impurity")
rf_caret2 = train(data = col_input1, income ~ ., method = "ranger", trControl = control10, tuneGrid = tuning_grid1, importance = "impurity")
rf_caret3 = train(data = col_input1, income ~ ., method = "ranger", trControl = control10, tuneGrid = tuning_grid2, importance = "impurity")

rf_caret1 #low R-squared of 0.579
rf_caret2


# Extract final Model 
rf_21 = rf_caret1$finalModel
rf_22 = rf_caret2$finalModel


# Visualize Random Forest Variable importance

# get and order importance scores
importance_rf_21 = data.frame(importance(rf_21)[order(importance(rf_21), decreasing=TRUE)])
importance_rf_22 = data.frame(importance(rf_22)[order(importance(rf_22), decreasing=TRUE)])

# adjust the row and column names of the data set
names(importance_rf_21) = "importance"
importance_rf_21$var_name = rownames(importance_rf_21)
importance_rf_22$var_name = rownames(importance_rf_22)



# select 10 most important covariates
importance_rf_21 = importance_rf_21[1:10,]
importance_rf_22 = importance_rf_21[1:10,]
# plot variable importance scores
ggplot(importance_rf_21, aes(x = reorder(var_name, importance, mean), y = importance)) +
  geom_point() +
  labs(title = "Random Forest variable importance", subtitle = "presented as the mean decrease in the sum of squared \nresiduals when a variable is included in a tree split", x = "", y = "Mean decrease in sum of squared residuals") +
  coord_flip() +
  theme(axis.text.y = element_text(hjust = 0))

ggplot(importance_rf_22, aes(x = reorder(var_name, importance, mean), y = importance)) +
  geom_point() +
  labs(title = "Random Forest variable importance", subtitle = "presented as the mean decrease in the sum of squared \nresiduals when a variable is included in a tree split", x = "", y = "Mean decrease in sum of squared residuals") +
  coord_flip() +
  theme(axis.text.y = element_text(hjust = 0))
summary(col_input0) # workhours 2 and 1 is important but many NAs - maybe impute working hours? 

## Imputation -------------------------------------------------------

# Children many NAs 
# Imputation method: 
# Take average
# do Random forest on imputation
# do linear trend/fit per person 


table(col_input0$child_info, col_input0$children)

#replace if not in primary family and NA then children == 0 
# Child Info: replace with category: Other for 9, 10, 11, 12, 14, 15 

#col_input0$child_info <- recode_factor(col_input0$child_info, 9 = 'Other', 10 = 'Other', 11 = 'Other' , 12 =  'Other', 14 =  'Other', 15 =  'Other')
#recode_factor(col_input0$child_info, 9 = "Other")

# generate new dataframe with col_input for imputed values 
col_input_impute <- col_input0
col_input_impute <- as.data.frame(unclass(col_input_impute), stringsAsFactors = TRUE)
summary(col_input_impute)

#replace NA in children with 0 
col_input_impute$children <- tidyr::replace_na(0)

#replace NA in line with median of line
col_input_impute$line <- tidyr::replace_na(1) 
table(is.na(col_input_impute$line))

# education 
# workhours2
# workhours1
# ethnicity 
# city 
# veteran 

# education
col_input_impute$education <- as.character(col_input_impute$education)
col_input_impute$education[is.na(col_input_impute$education)] <- "Unknown"
col_input_impute$education <- as.factor(col_input_impute$education)
summary(col_input_impute$education)
table(is.na(col_input_impute$education))

# ethnicity replace with last category DontKnow 
summary(col_input_impute$ethnicity)
col_input_impute$ethnicity <- as.factor(col_input_impute$ethnicity)
col_input_impute$ethnicity[is.na(col_input_impute$ethnicity)] <- "DontKnow"
summary(col_input_impute$ethnicity)

# veteran replace with category : Nonveteran because mode
summary(col_input_impute$veteran)
col_input_impute$veteran <- as.factor(col_input_impute$veteran)
col_input_impute$veteran[is.na(col_input_impute$veteran)] <- "Nonveteran"
summary(col_input_impute$veteran)
# city 
summary(col_input_impute$city)
col_input_impute$city[is.na(col_input_impute$city)] <- "Unknown"
col_input_impute$city <- as.character(col_input_impute$city) #transfer to character
col_input_impute$city <- replace(col_input_impute$city, is.na(col_input_impute$city), "Unknown") #replace
col_input_impute$city <- as.factor(col_input_impute$city) #turn to factor again
summary(col_input_impute$city)
# union 
col_input_impute$union <- as.character(col_input_impute$union) #transfer to character
col_input_impute$union <- replace(col_input_impute$union, is.na(col_input_impute$union), "Unknown") #replace
col_input_impute$union <- as.factor(col_input_impute$union) #turn to factor again
summary(col_input_impute$union)
# status_last_week 
col_input_impute$status_last_week <- as.character(col_input_impute$status_last_week) #transfer to character
col_input_impute$status_last_week <- replace(col_input_impute$status_last_week, is.na(col_input_impute$status_last_week), "Unknown") #replace
col_input_impute$status_last_week <- as.factor(col_input_impute$status_last_week) #turn to factor again
summary(col_input_impute$status_last_week)
# class2
col_input_impute$class2 <- as.character(col_input_impute$class2) #transfer to character
col_input_impute$class2 <- replace(col_input_impute$class2, is.na(col_input_impute$class2), "Unknown") #replace
col_input_impute$class2 <- as.factor(col_input_impute$class2) #turn to factor again
summary(col_input_impute$class2)

# Workhours 1 and 2 
table(is.na(col_input_impute$workhours1), is.na(col_input_impute$workhours2))
# 103 individuals for which we do not have any information for both 

table(is.na(col_input_impute$workhours1), col_input_impute$class)
table(is.na(col_input_impute$workhours1), col_input_impute$class)

# Create subset with nas for working hours
df_na <- subset(col_input_impute) |> 
  filter(is.na(workhours1) == TRUE | is.na(workhours2) == TRUE)

table(df_na$status_last_week) # some keeping house, some looking for work, some at school, some unable to work, 
# then some with a job, and some working, some other >> but all else maybe say 0? 

table(col_input_impute$workhours1, col_input_impute$status_last_week)

#only with a job can be replaced by 0 

# just delete with working hours 
col_input_impute_filter <- col_input_impute |> 
  filter(is.na(workhours1) == FALSE & is.na(workhours2) == FALSE)

# Now delete paid by hours
col_input_impute_filter <- col_input_impute_filter |> 
  filter(is.na(paidbyhour) == FALSE)

table(is.na(col_input_impute_filter)) # No NAs anymore

summary(col_input_impute_filter)
col_input_impute_filter <- as.data.frame(unclass(col_input_impute_filter), stringsAsFactors = TRUE)

summary(col_input_impute_filter)

## Run Random Forests with new data ---------------------------------

rf_3 = ranger(data = col_input_impute_filter, dependent.variable.name = "income", importance = "impurity")
rf_caret3 = train(data = col_input_impute_filter, income ~ ., method = "ranger", trControl = control5, tuneGrid = tuning_grid1, importance = "impurity")

rf_31 = rf_caret3$finalModel

#importance
importance_rf_31 = data.frame(importance(rf_31)[order(importance(rf_31), decreasing=TRUE)])

# adjust the row and column names of the data set
names(importance_rf_31) = "importance"
importance_rf_31$var_name = rownames(importance_rf_31)

# select 10 most important covariates
importance_rf_31 = importance_rf_31[1:10,]

# plot variable importance scores
ggplot(importance_rf_31, aes(x = reorder(var_name, importance, mean), y = importance)) +
  geom_point() +
  labs(title = "Random Forest variable importance", subtitle = "presented as the mean decrease in the sum of squared \nresiduals when a variable is included in a tree split", x = "", y = "Mean decrease in sum of squared residuals") +
  coord_flip() +
  theme(axis.text.y = element_text(hjust = 0))

summary(rf_31) #R-squared only 0.55


# Other tuning grid 
rf_4 = ranger(data = col_input_impute_filter, dependent.variable.name = "income", importance = "impurity")
rf_caret4 = train(data = col_input_impute_filter, income ~ ., method = "ranger", trControl = control5, tuneGrid = tuning_grid_2b, importance = "impurity")

rf_41 = rf_caret3$finalModel

#importance
importance_rf_41 = data.frame(importance(rf_31)[order(importance(rf_31), decreasing=TRUE)])

# adjust the row and column names of the data set
names(importance_rf_41) = "importance"
importance_rf_41$var_name = rownames(importance_rf_31)

# select 10 most important covariates
importance_rf_31 = importance_rf_31[1:10,]

# plot variable importance scores
ggplot(importance_rf_41, aes(x = reorder(var_name, importance, mean), y = importance)) +
  geom_point() +
  labs(title = "Random Forest variable importance", subtitle = "presented as the mean decrease in the sum of squared \nresiduals when a variable is included in a tree split", x = "", y = "Mean decrease in sum of squared residuals") +
  coord_flip() +
  theme(axis.text.y = element_text(hjust = 0))

summary(rf_41) #R-squared only 0.55



##RF new 
training_imputed <- missRanger(data = col_input1,
                               formula = . ~ . -income - income_a -income_b -income_c -income_d, 
                               num.trees = 100)
rf_5 = ranger(data = col_input_impute_filter, dependent.variable.name = "income", importance = "impurity")
rf_caret5 = train(data = col_input_impute_filter, income ~ ., method = "ranger", trControl = control5, tuneGrid = tuning_grid1, importance = "impurity")

rf_51 = rf_caret5$finalModel
