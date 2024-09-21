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

dataTrain <- vroom("/Users/carsoncollins/Desktop/Stats348/BikeShare/train.csv")
dataTest <- vroom("/Users/carsoncollins/Desktop/Stats348/BikeShare/test.csv")

dataTrain <- dataTrain %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))


#-----EDA-------------
dplyr::glimpse(dataTrain) 
skimr::skim(dataTrain) 

DataExplorer::plot_intro(dataTrain)
DataExplorer::plot_correlation(dataTrain)
DataExplorer::plot_bar(dataTrain)
DataExplorer::plot_histogram(dataTrain) 
DataExplorer::plot_missing(dataTrain) 

#Plot 1: Scatter plot of temperature vs count
plot1 <- ggplot(data = dataTrain, aes(x=temp, y=count)) +
  geom_point(color = "blue") +
  geom_smooth(se = FALSE, color = "red") +
  ggtitle("Temperature vs. Bike Count")

#Plot 2: Bar plot of weather conditions
plot2 <- ggplot(data = dataTrain, aes(x=weather)) +
  geom_bar(fill = "lightblue") +
  ggtitle("Weather Conditions")

#Plot 3: Histogram of wind speed
plot3 <- ggplot(data = dataTrain, aes(x=windspeed)) +
  geom_histogram(fill = "lightgreen", bins = 20) +
  ggtitle("Wind Speed Distribution")

#Plot 4: Box plot of humidity vs count
plot4 <- ggplot(data = dataTrain, aes(x=factor(humidity), y=count)) +
  geom_boxplot(fill = "orange") +
  ggtitle("Humidity vs. Bike Count")

combined_plot <- (plot1 | plot2) / (plot3 | plot4)

print(combined_plot)

ggsave("4_panel_bikeshare_plot.png", plot = combined_plot, width = 12, height = 8)

#----------------------Linear Model-----------------------------#

my_recipe1 <- recipe(count ~ ., data = dataTrain) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_time(datetime, features=c("hour")) %>%
  step_cut(datetime_hour, breaks=c(7, 15, 24)) %>%
  step_mutate(season = factor(season, labels=c("spring","summer","fall","winter")))%>%
  step_mutate(datetime_hour=factor(datetime_hour)) %>%
  step_rm(datetime, atemp) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

my_linear_model <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression") 

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_linear_model) %>%
  fit(data = dataTrain)

bike_predictions <- predict(bike_workflow, new_data = dataTest)

bike_predictions

kaggle_submission <- bike_predictions %>%
  bind_cols(., dataTest) |>
  select(datetime, .pred) |>
  rename(count=.pred) |>
  mutate(count=exp(count)) |>
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="/Users/carsoncollins/Desktop/Stats348/BikeShare/LinearPredsTESTNOW.csv", delim=",")
#-----------------Penalized Regression------------------------------#

## Penalized regression model
preg_model <- linear_reg(penalty=0.01, mixture=.99) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=dataTrain)
preg_preditions <- predict(preg_wf, new_data=dataTest)
preg_preditions <- exp(preg_preditions)


kaggle_submission <- preg_preditions %>%
  bind_cols(., dataTest) |>
  select(datetime, .pred) |>
  rename(count=.pred) |>
  mutate(count=exp(count)) |>
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./Penalty0.01Mix0.99.csv", delim=",")
#-------------------Tuning model--------------#

my_recipe2 <- recipe(count~., data = dataTrain) %>% 
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



L <- 5
K <- 5

## Penalized regression model
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R
## Set Workflow
preg_wf <- workflow() %>%
add_recipe(my_recipe2) %>%
add_model(preg_model)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = L) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(dataTrain, v = K, repeats=1)
## Run the CV
CV_results <- preg_wf %>%
tune_grid(resamples=folds,
          grid=grid_of_tuning_params,
          metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL

## Plot Results (example)
#collect_metrics(CV_results) %>% # Gathers metrics into DF
  #filter(.metric=="rmse") %>%
#ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
#geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
select_best(metric = "rmse")
## Finalize the Workflow & fit it
final_wf <-
preg_wf %>%
finalize_workflow(bestTune) %>%
fit(data=dataTrain)

## Predict
preg_tune <- predict(final_wf, new_data = dataTest)

kaggle_submission <- preg_tune %>%
  bind_cols(., dataTest) |>
  select(datetime, .pred) |>
  rename(count=.pred) |>
  mutate(count=exp(count)) |>
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="/Users/carsoncollins/Desktop/Stats348/BikeShare/TuningModel.csv", delim=",")

#---------Poisson model-------------#
library(poissonreg)

my_pois_model <- poisson_reg() %>% #Type of model
  set_engine("glm") %>% # GLM = generalized linear model
  set_mode("regression") %>%
fit(formula=count~., data=dataTrain)

## Generate Predictions Using Linear Model
bike_predictions <- predict(my_pois_model,
                            new_data=dataTest) # Use fit to predict
bike_predictions ## Look at the output

kaggle_submission <- bike_predictions %>%
  bind_cols(., dataTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=kaggle_submission, file="./PoissonPreds.csv", delim=",")


