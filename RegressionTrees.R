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

dataTrain <- vroom("/Users/carsoncollins/Desktop/Stats348/BikeShare/train.csv")
dataTest <- vroom("/Users/carsoncollins/Desktop/Stats348/BikeShare/test.csv")

dataTrain <- dataTrain %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))

my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # What R function to use
  set_mode("regression")

# Create recipe (optional based on your feature engineering needs)
my_recipe <- recipe(count ~ ., data = dataTrain) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_time(datetime, features=c("hour")) %>%
  step_cut(datetime_hour, breaks=c(7, 15, 24)) %>%
  step_mutate(season = factor(season, labels=c("spring","summer","fall","winter")))%>%
  step_mutate(datetime_hour=factor(datetime_hour)) %>%
  step_rm(datetime, atemp) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Create a workflow with the model and the recipe
my_workflow <- workflow() %>%
  add_model(my_mod) %>%
  add_recipe(my_recipe)

# Set up grid of tuning values for hyperparameters
my_grid <- grid_regular(tree_depth(range = c(1, 10)),
                        cost_complexity(range = c(-2, -0.5)),
                        min_n(range = c(2, 10)),
                        levels = 5)

# Set up K-fold cross-validation
set.seed(123)
cv_folds <- vfold_cv(dataTrain, v = 5)

# Tune the model parameters using the grid
tune_results <- tune_grid(
  my_workflow,
  resamples = cv_folds,
  grid = my_grid,
  metrics = metric_set(rmse, rsq)
)

# Find the best tuning parameters based on RMSE
best_params <- select_best(tune_results, "rmse")

# Finalize the workflow with the best parameters
final_workflow <- finalize_workflow(my_workflow, best_params)

# Fit the final model to the full training data
final_fit <- fit(final_workflow, data = dataTrain)

# Make predictions on the test data
predictions <- predict(final_fit, new_data = dataTest)

# You can un-log transform the predictions if needed
predictions <- exp(predictions$.pred)

# View predictions
predictions











