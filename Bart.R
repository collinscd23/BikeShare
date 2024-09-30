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

train <- vroom("/Users/carsoncollins/Desktop/Stats348/BikeShare/train.csv")
test <- vroom("/Users/carsoncollins/Desktop/Stats348/BikeShare/test.csv")
# Preprocess the data
train <- train %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))

# Data Preprocessing Recipe (same as your recipe)
bart_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(season = factor(season, labels = c("Spring", "Summer", "Fall", "Winter")),
              holiday = factor(holiday),
              workingday = factor(workingday),
              weather = factor(ifelse(weather == 4, 3, weather), labels = c("Sunny", "Cloudy", "Rainy"))) %>%
  step_mutate(hour = factor(lubridate::hour(datetime))) %>%  # Extract hour using lubridate
  step_date(datetime, features = "dow") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

recipe1 <- recipe(count~., data = train) %>% 
  step_date(datetime, features = "dow") %>% 
  step_time(datetime, features="hour") %>% 
  step_rm(datetime, holiday, temp) %>% 
  step_mutate(working_hour = workingday * datetime_hour) %>% 
  step_mutate(season=factor(season, labels=c("Spring","Summer","Fall","Winter")),
              workingday=factor(workingday),
              weather= factor(ifelse(weather==4,3,weather), labels=c("Sunny","Cloudy","Rainy"))) %>%
  step_mutate(datetime_hour=factor(datetime_hour),
              datetime_dow = factor(datetime_dow))


bart_model <- parsnip::bart(trees = 5000) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression")%>%
  translate()

bart_workflow <- workflow() %>% 
  add_model(bart_model) %>% 
  add_recipe(recipe1) %>%
  fit(train)

predictions <- predict(bart_workflow, new_data = test)

# View predictions
print(predictions)


kaggle_submission <- predictions %>%
  bind_cols(test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = exp(count)) %>%
  mutate(datetime = as.character(format(datetime)))

# Save predictions to CSV
vroom_write(kaggle_submission, file = "/Users/carsoncollins/Desktop/Stats348/BikeShare/BART.csv", delim = ",")
