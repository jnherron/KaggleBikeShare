library(vroom)
library(dplyr)


# Upload Data
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")

# dyplyr
train_data <- train_data %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))

# recipe
my_recipe <- recipe(count~., data=train_data) %>%
  step_mutate(weather = ifelse(weather == 4,3,weather)) %>%
  step_mutate(weather=factor(weather)) %>%
  step_time(datetime, features=c("hour")) %>%
  step_date(datetime, features=c("dow", "month", "year")) %>%
  step_mutate(datetime_dow = as.numeric(datetime_dow)) %>%
  step_harmonic(datetime_hour, cycle_size = 24, frequency = 1) %>%
  step_harmonic(datetime_dow, cycle_size = 7, frequency = 1) %>%
  step_rm(datetime) %>%
  step_mutate(season=factor(season)) %>%
  step_mutate(workingday=factor(workingday)) %>%
  step_mutate(holiday=factor(holiday)) %>%
  # step_interact(~ datetime_hour:workingday) %>%
  # step_interact(~ holiday:workingday) %>%
  # step_interact(~ weather:workingday) %>%
  # step_interact(~ season:workingday) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Bake
baked_data <- bake(prep(my_recipe), new_data=train_data)


# Linear Regression Using Workflow
lin_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

# Combine into a Workflow and fit
bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(lin_model) %>%
  fit(data=train_data)

# Run all the steps on test data
lin_preds <- exp(predict(bike_workflow, new_data = test_data))


# Format the Predictions for Submission to Kaggle
kag_sub <- lin_preds %>%
  bind_cols(., test_data) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

# Write out the file
vroom_write(x=kag_sub, file="./RecipePreds.csv", delim=",")




prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data=test_data)
head(baked, 5)
