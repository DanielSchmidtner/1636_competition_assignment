pacman::p_load(plyr, tidyverse, data.table, fst, ranger, xgboost, caret, viridis, missRanger)

training <- read_csv("/Users/raphaelboulat/Desktop/Github/1636_competition_assignment/input_data/trainig_test_data.csv")
testing <- read_csv("/Users/raphaelboulat/Desktop/Github/1636_competition_assignment/input_data/holdout_data.csv")

# Convert all columns to factor

training <- as.data.frame(unclass(training), stringsAsFactors = TRUE) 

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


### random forest 

control <- trainControl(method = "cv", number=5)
tuning_grid = expand.grid(mtry = seq(10, 20, by = 5), splitrule = "variance", min.node.size=500)
rf_caret = caret::train(data = training_imputed , income ~ ., method = "ranger", tuneGrid = tuning_grid, trControl = control, importance="impurity")
