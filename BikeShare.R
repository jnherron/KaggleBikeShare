library(tidyverse)
library(tidymodels)
library(glmnet)


## Penalized regression mode ---------------------------------

# penalty=0, mixture=0
# kaggle score: 1.02303
preg_model <- linear_reg(penalty=0, mixture=0) %>%
  set_engine("glmnet")
preg_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(preg_model) %>%
fit(data=train_data)
predict(preg_wf, new_data=test_data)

# penalty=.5, mixture=0
# kaggle score: 1.03200
preg_model <- linear_reg(penalty=.5, mixture=0) %>%
  set_engine("glmnet")
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=train_data)
lin_preds <- exp(predict(preg_wf, new_data=test_data))

# penalty=0, mixture=.5
# kaggle score: 1.02293
preg_model <- linear_reg(penalty=0, mixture=.5) %>%
  set_engine("glmnet")
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=train_data)
lin_preds <- exp(predict(preg_wf, new_data=test_data))

# penalty=.5, mixture=.5
# kaggle score: 1.12224
preg_model <- linear_reg(penalty=.5, mixture=.5) %>%
  set_engine("glmnet")
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=train_data)
lin_preds <- exp(predict(preg_wf, new_data=test_data))

# penalty=.1, mixture=.1
# kaggle score: 1.02230
preg_model <- linear_reg(penalty=.1, mixture=.1) %>%
  set_engine("glmnet")
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=train_data)
lin_preds <- exp(predict(preg_wf, new_data=test_data))




# Finish and format for Kaggle -----------------------------------

# Format the Predictions for Submission to Kaggle
kag_sub <- lin_preds %>%
  bind_cols(., test_data) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

# Write out the file
vroom_write(x=kag_sub, file="./PenaltyPreds.csv", delim=",")

