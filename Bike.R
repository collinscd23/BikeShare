library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(skimr)
library(patchwork)
library(DataExplorer)


dataTrain <- vroom("/Users/carsoncollins/Desktop/Stat348/BikeShare/train.csv")
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
