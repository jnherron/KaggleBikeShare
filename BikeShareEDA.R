library(vroom)
library(ggplot2)
library(patchwork)
library(DataExplorer)


# Upload Data
bike_data <- vroom("train.csv")


# EDA
DataExplorer::plot_intro(bike_data)         # Good Dataset! Note that some discrete columns were recorded as continuous
DataExplorer::plot_correlation(bike_data)   # Correlation w/in weather, temp, and counting variables
DataExplorer::plot_bar(bike_data)           # As Expected...
DataExplorer::plot_missing(bike_data)       # None!!!! :)


plot_temp <- ggplot(data = bike_data, aes(x = temp, y = count)) +
  geom_point() +
  geom_smooth(se=FALSE) +
  labs(title = "Count by Temp",
       x = "Temp (Celcius)",
       y = "Count")

plot_weather <- ggplot(data = bike_data, aes(x = weather)) +
  geom_bar() +
  labs(title = "Days Reported by Weather",
       x = "Ideal(1) to Poor(4) Weather",
       y = "Days Reported")

plot_season <- ggplot(data = bike_data, aes(x = season, y = count)) +
  geom_jitter() +
  labs(title = "Count by Season",
       x = "1=Spr, 2=Sum, 3=Fall, 4=Win",
       y = "Count")

plot_workingday <- ggplot(data = bike_data, aes(x = workingday, y = count)) +
  geom_jitter() +
  labs(title = "Workingday vs. Not",
       x = "1=Workingday",
       y = "Count")


# Display Plots
(plot_temp + plot_weather) / (plot_season + plot_workingday)



