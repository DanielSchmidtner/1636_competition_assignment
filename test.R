pacman::p_load(plyr, tidyverse, data.table, fst, ranger, xgboost, caret, viridis, missRanger)

training1 <- read_csv("/Users/raphaelboulat/Desktop/Github/1636_competition_assignment/input_data/trainig_test_data.csv")
testing1 <- read_csv("/Users/raphaelboulat/Desktop/Github/1636_competition_assignment/input_data/holdout_data.csv")
training1 <- as.data.frame(unclass(training1), stringsAsFactors = TRUE) 
testing1 <- as.data.frame(unclass(testing1), stringsAsFactors = TRUE) 


## create numerical variables for the training data

training1$state <- as.numeric(training1$state)
training1$class <- as.numeric(training1$class)
training1$marriage <- as.numeric(training1$marriage)
training1$race <- as.numeric(training1$race)
training1$sex <- as.numeric(training1$sex)
training1$veteran <- as.numeric(training1$veteran)
training1$region2 <- as.numeric(training1$region2)
training1$city <- as.numeric(training1$city)
training1$ethnicity <- as.numeric(training1$ethnicity)
training1$status_last_week <- as.numeric(training1$status_last_week)
training1$class2 <- as.numeric(training1$class2)
training1$paidbyhour <- as.numeric(training1$paidbyhour)
training1$union <- as.numeric(training1$union)
training1$industry <- as.numeric(training1$industry)
training1$children <- as.numeric(training1$children)
training1$child_info <- as.numeric(training1$child_info)
training1$family_relation <- as.numeric(training1$family_relation)
training1$family_reference <- as.numeric(training1$family_reference)
training1$occupation <- as.numeric(training1$occupation)
training1$education <- as.numeric(training1$education)

### impute NA's

training_imputed <- missRanger(data = training1,
                               formula = . ~ . -income - income_a -income_b -income_c -income_d, 
                               num.trees = 100)


## create numerical variables for the testing data

testing1$state <- as.numeric(testing1$state)
testing1$class <- as.numeric(testing1$class)
testing1$marriage <- as.numeric(testing1$marriage)
testing1$race <- as.numeric(testing1$race)
testing1$sex <- as.numeric(testing1$sex)
testing1$veteran <- as.numeric(testing1$veteran)
testing1$region2 <- as.numeric(testing1$region2)
testing1$city <- as.numeric(testing1$city)
testing1$ethnicity <- as.numeric(testing1$ethnicity)
testing1$status_last_week <- as.numeric(testing1$status_last_week)
testing1$class2 <- as.numeric(testing1$class2)
testing1$paidbyhour <- as.numeric(testing1$paidbyhour)
testing1$union <- as.numeric(testing1$union)
testing1$industry <- as.numeric(testing1$industry)
testing1$children <- as.numeric(testing1$children)
testing1$child_info <- as.numeric(testing1$child_info)
testing1$family_relation <- as.numeric(testing1$family_relation)
testing1$family_reference <- as.numeric(testing1$family_reference)
testing1$occupation <- as.numeric(testing1$occupation)
testing1$education <- as.numeric(testing1$education)

### impute NA's testing

testing_imputed <- missRanger(data = testing1,
                               formula = . ~ . - income_a -income_b -income_c -income_d, 
                               num.trees = 100)

control <- trainControl(method = "cv", number=5)
tuning_grid_rf2 = expand.grid(mtry = 4, splitrule = "variance", min.node.size=100)
rf_caret_2 = caret::train(data = training_imputed , income ~ . - income_a -income_b -income_c -income_d, method = "ranger", tuneGrid = tuning_grid_rf2 , trControl = control, importance="impurity")
rf_2 <- rf_caret_2$finalModel
rf_2

control <- trainControl(method = "cv", number=5)
tuning_grid_rf3 = expand.grid(mtry=seq(10,25, by=5), splitrule = "variance", min.node.size=60)
rf_caret_3 = caret::train(data = training_imputed , income ~ . - income_a -income_b -income_c -income_d, method = "ranger", tuneGrid = tuning_grid_rf3 , trControl = control, importance="impurity")
rf_3 <- rf_caret_3$finalModel
rf_3

testing_imputed <- as.data.frame(testing_imputed)

pred_rf_2 <- predict(rf_2, data = testing_imputed)
pred_rf_2
Metrics::rmse(actual = training_imputed$income, predicted = pred_rf_2$predictions) ### not really sure it is correct

pred_rf_3 <- predict(rf_3, data = testing_imputed)
pred_rf_3
Metrics::rmse(actual = training_imputed$income, predicted = pred_rf_3$predictions) 


tuning_grid_xbg = expand.grid(nrounds = 100, max_depth = 3, eta = 0.4, gamma = 0.01, 
                              colsample_bytree = 1, min_child_weight = 1, subsample = 1)
gb_caret = caret::train(data = training_imputed, income ~ . - income_a -income_b -income_c -income_d, method = "xgbTree", trControl = control, tuneGrid = tuning_grid_xbg, verbosity = 0)
gb_1 <- gb_caret$finalModel

testing_imputed <- as.data.frame(testing_imputed)

pred_gb_1 <- predict(gb_1 , newdata = as.matrix(testing_imputed))

