# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(skimr)
library(patchwork)
library(DataExplorer)
library(recipes)
library(dplyr)
library(poissonreg)
library(glmnet)
library(rpart)
library(ranger)
library(stacks)

# Load the data
dataTrain <- vroom("/Users/carsoncollins/Desktop/Stats348/BikeShare/train.csv")
dataTest <- vroom("/Users/carsoncollins/Desktop/Stats348/BikeShare/test.csv")

# Preprocess the data
dataTrain <- dataTrain %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))

# Create recipe for feature engineering
my_recipe <- recipe(count~., data = dataTrain) %>% 
  step_mutate(season=factor(season, labels=c("Spring","Summer","Fall","Winter")),
              holiday=factor(holiday),
              workingday=factor(workingday),
              weather= factor(ifelse(weather==4,3,weather), labels=c("Sunny","Cloudy","Rainy"))) %>% 
  step_date(datetime, features = "dow") %>% 
  step_time(datetime, features="hour") %>% 
  step_rm(datetime) %>% 
  step_mutate(datetime_hour=factor(datetime_hour)) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# Set up models
rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 600) %>%
  set_engine("ranger") %>%
  set_mode("regression")

lm_mod <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

tree_mod <- decision_tree(tree_depth = tune(),
              cost_complexity = tune(),
              min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # What R function to use
  set_mode("regression")

# Create workflows for each base model
rf_workflow <- workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(my_recipe)

lm_workflow <- workflow() %>%
  add_model(lm_mod) %>%
  add_recipe(my_recipe)

tree_workflow <- workflow() %>%
  add_model(tree_mod) %>%
  add_recipe(my_recipe)

# Cross-validation setup
cv_folds <- vfold_cv(dataTrain, v = 6, repeats = 1)

# Tuning grid
rf_grid <- grid_regular(
  mtry(range = c(1, 40)),
  min_n(),
  levels = 6
)

tree_grid <- grid_regular(
  cost_complexity(),
  tree_depth(),
  levels =6
)

# Tune each model
rf_results <- tune_grid(rf_workflow, resamples = cv_folds, grid = rf_grid, metrics = metric_set(rmse))
tree_results <- tune_grid(tree_workflow, resamples = cv_folds, grid = tree_grid, metrics = metric_set(rmse))

# Select best models
best_rf <- select_best(rf_results, metric = "rmse")
best_tree <- select_best(tree_results, metric = "rmse")

# Finalize workflows with the best hyperparameters
final_rf_workflow <- finalize_workflow(rf_workflow, best_rf)
final_tree_workflow <- finalize_workflow(tree_workflow, best_tree)

# Fit models to the entire training set
final_rf <- fit(final_rf_workflow, dataTrain)
final_tree <- fit(final_tree_workflow, dataTrain)
final_lm <- fit(lm_workflow, dataTrain)

# Make predictions with the base models
rf_preds <- predict(final_rf, new_data = dataTrain) %>% rename(rf_pred = .pred)
tree_preds <- predict(final_tree, new_data = dataTrain) %>% rename(tree_pred = .pred)
lm_preds <- predict(final_lm, new_data = dataTrain) %>% rename(lm_pred = .pred)

# Combine base model predictions
stacked_train <- dataTrain %>%
  bind_cols(rf_preds, tree_preds, lm_preds)

# Meta-model (stacking with linear regression)
stack_recipe <- recipe(count ~ rf_pred + tree_pred + lm_pred, data = stacked_train)
meta_mod <- linear_reg() %>% set_engine("lm")
stack_workflow <- workflow() %>%
  add_model(meta_mod) %>%
  add_recipe(stack_recipe)

# Fit the stacking model on the base predictions
final_meta_model <- fit(stack_workflow, stacked_train)

# Make predictions on test data using the base models
rf_test_preds <- predict(final_rf, new_data = dataTest) %>% rename(rf_pred = .pred)
tree_test_preds <- predict(final_tree, new_data = dataTest) %>% rename(tree_pred = .pred)
lm_test_preds <- predict(final_lm, new_data = dataTest) %>% rename(lm_pred = .pred)

# Combine test predictions
stacked_test <- dataTest %>%
  bind_cols(rf_test_preds, tree_test_preds, lm_test_preds)

# Make final predictions using the meta-model
stacked_preds <- predict(final_meta_model, new_data = stacked_test)

# Prepare Kaggle submission
kaggle_submission <- stacked_preds %>%
  bind_cols(dataTest) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = exp(count)) %>%
  mutate(datetime = as.character(format(datetime)))

# Save predictions to CSV
vroom_write(kaggle_submission, file = "/Users/carsoncollins/Desktop/Stats348/BikeShare/StackedModel.csv", delim = ",")
