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




