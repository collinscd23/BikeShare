library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(skimr)
library(patchwork)
library(DataExplorer)


dataTrain <- vroom("/Users/carsoncollins/Desktop/Stat348/BikeShare/train.csv")%>%
  select(-casual, -registered)
dataTest <- vroom("/Users/carsoncollins/Desktop/Stat348/BikeShare/test.csv")



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


## Setup and Fit the Linear Regression Model
my_linear_model <- linear_reg() %>% #Type of model
  set_engine("lm") %>% # Engine = What R function to use
  set_mode("regression") %>% # Regression just means quantitative response6
  fit(formula=count ~ ., data=dataTrain)

## Generate Predictions Using Linear Model
bike_predictions <- predict(my_linear_model,
                            new_data=dataTest) # Use fit to predict11
bike_predictions ## Look at the output


kaggle_submission <- bike_predictions %>%
bind_cols(., dataTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")



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


----------test-----------
library(lubridate) # For handling date-time objects

dataTest <- dataTest %>%
  mutate(datetime = as.POSIXct(datetime, format="%Y-%m-%d %H:%M:%S")) %>%
  mutate(hour = hour(datetime), 
         day_of_week = wday(datetime, label = TRUE), 
         month = month(datetime),
         year = year(datetime),
         season = case_when(
           month %in% c(3, 4, 5) ~ "Spring",
           month %in% c(6, 7, 8) ~ "Summer",
           month %in% c(9, 10, 11) ~ "Fall",
           month %in% c(12, 1, 2) ~ "Winter"
         )) %>%
  mutate(log_temp = log(temp + 1),       # Adding 1 to avoid log(0)
         log_windspeed = log(windspeed + 1),
         log_humidity = log(humidity + 1))

bike_predictions <- predict(my_pois_model, new_data = dataTest)

kaggle_submission <- bike_predictions %>%
  bind_cols(., dataTest) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(x = kaggle_submission, file = "./PoissonPreds_with_log_features.csv", delim = ",")




