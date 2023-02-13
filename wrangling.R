pacman::p_load(plyr, tidyverse, data.table, fst, ranger, xgboost, caret, viridis, missRanger)

training <- read_csv("/Users/raphaelboulat/Desktop/Github/1636_competition_assignment/input_data/trainig_test_data.csv")
testing <- read_csv("/Users/raphaelboulat/Desktop/Github/1636_competition_assignment/input_data/holdout_data.csv")

# Convert all columns to factor

training <- as.data.frame(unclass(training), stringsAsFactors = TRUE) 
testing <- as.data.frame(unclass(testing), stringsAsFactors = TRUE) 

### filter for weird numbers --> not sure how to treat income_a/.../income_d (they only have 2,3,4 and "No change" --> maybe kick them out?)

training <- training |>
  filter(!race==4) |>
  filter(!race==5) |>
  filter(!family_reference==7) |>
  filter(!family_reference==8)|>
  filter(!family_reference==9)|>
  filter(!family_reference==10)

### imputation 

training_imputed <- missRanger(data = training,
                               formula = . ~ . -income - income_a -income_b -income_c -income_d, 
                               num.trees = 100)

testing_imputed <- missRanger(data = testing,
                              formula = . ~ . - income_a -income_b -income_c -income_d, 
                              num.trees = 100)

### random forest 

control <- trainControl(method = "cv", number=5)
tuning_grid_rf = expand.grid(mtry = 5, splitrule = "variance", min.node.size=500)
rf_caret = caret::train(data = training_imputed , income ~ ., method = "ranger", tuneGrid = tuning_grid_rf , trControl = control, importance="impurity")
rf_1 <- rf_caret$finalModel
rf_1

pred_rf_1 <- predict(rf_caret, testing_imputed)
Metrics::rmse(actual = training_imputed$income, predicted = pred_rf_1)

### boosting 

control = trainControl(method = "cv", number = 5)
tuning_grid_xbg = expand.grid(nrounds = 100, max_depth = 3, eta = 0.4, gamma = 0.01, 
                              colsample_bytree = 1, min_child_weight = 1, subsample = 1)
gb_caret = caret::train(data = training_imputed, income ~ ., method = "xgbTree", trControl = control, tuneGrid = tuning_grid_xbg, verbosity = 0)
gb_1 <- gb_caret$finalModel

pred_gb_1 <- predict(gb_1 , newdata = testing_imputed)
Metrics::rmse(actual = training_imputed$income, predicted = pred_gb_1)

