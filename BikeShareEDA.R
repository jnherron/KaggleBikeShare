library(vroom)
library(ggplot2)
library(patchwork)
library(DataExplorer)


# Upload Data
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")


# EDA
DataExplorer::plot_intro(train_data)         # Good Dataset! Note that some discrete columns were recorded as continuous
DataExplorer::plot_correlation(train_data)   # Correlation w/in weather, temp, and counting variables
DataExplorer::plot_bar(train_data)           # As Expected...
DataExplorer::plot_missing(train_data)       # None!!!! :)


plot_temp <- ggplot(data = train_data, aes(x = temp, y = count)) +
  geom_point() +
  geom_smooth(se=FALSE) +
  labs(title = "Count by Temp",
       x = "Temp (Celcius)",
       y = "Count")

plot_weather <- ggplot(data = train_data, aes(x = weather)) +
  geom_bar() +
  labs(title = "Days Reported by Weather",
       x = "Ideal(1) to Poor(4) Weather",
       y = "Days Reported")

plot_season <- ggplot(data = train_data, aes(x = season, y = count)) +
  geom_jitter() +
  labs(title = "Count by Season",
       x = "1=Spr, 2=Sum, 3=Fall, 4=Win",
       y = "Count")

plot_workingday <- ggplot(data = train_data, aes(x = workingday, y = count)) +
  geom_jitter() +
  labs(title = "Workingday vs. Not",
       x = "1=Workingday",
       y = "Count")


# Display Plots
(plot_temp + plot_weather) / (plot_season + plot_workingday)



# IMPORTANT NOTE:
# No data has weather category 4


