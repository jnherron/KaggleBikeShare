# Linear Regression ---------------------------
lin_mod <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(formula = count ~ season + holiday + workingday + weather + temp + atemp + humidity + windspeed, data=train_data)

# Generate Predictions
preds <- predict(lin_mod, new_data=test_data)
preds

# Format the Predictions for Submission to Kaggle
kag_sub <- preds %>%
  bind_cols(., test_data) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

# Write out the file
vroom_write(x=kag_sub, file="./LinearPreds.csv", delim=",")



## Penalized Regression -----------------------------------

preg_model <- linear_reg(penalty=tune(), mixture=tune()) %>%
  set_engine("glmnet")

preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model)

grid_of_tuning_params <- grid_regular(penalty(), mixture(), levels=10)
folds <- vfold_cv(train_data, v=5, repeats=1)

CV_results <- preg_wf %>%
  tune_grid(resamples=folds, grid=grid_of_tuning_params, metrics=metric_set(rmse,mae))

collect_metrics(CV_results) %>%
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

bestTune <- CV_results %>%
  select_best(metric="rmse")



# penalty=.0000000001, mixture=.111
preg_model <- linear_reg(penalty=.0000000001, mixture=.111) %>%
  set_engine("glmnet")
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=train_data)
lin_preds <- exp(predict(preg_wf, new_data=test_data))

# Format the Predictions for Submission to Kaggle
kag_sub <- lin_preds %>%
  bind_cols(., test_data) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

# Write out the file
vroom_write(x=kag_sub, file="./PenaltyPreds.csv", delim=",")




