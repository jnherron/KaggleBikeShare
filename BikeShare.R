library(tidyverse)
library(tidymodels)
library(glmnet)
library(rpart)
library(ranger)
library(bonsai)
library(lightgbm)
library(agua)


#
# Penalized regression mode ---------------------------------

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

# Finish and format for Kaggle

# Format the Predictions for Submission to Kaggle
kag_sub <- lin_preds %>%
  bind_cols(., test_data) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

# Write out the file
vroom_write(x=kag_sub, file="./PenaltyPreds.csv", delim=",")








# Regression Tree ---------------------------------------------------

tree_model <- decision_tree(tree_depth=tune(), cost_complexity=tune(), min_n=tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")
tree_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(tree_model)

grid_of_tuning_params <- grid_regular(tree_depth(), cost_complexity(), min_n(), levels=5)
folds <- vfold_cv(train_data, v=5, repeats=1)

CV_results <- tree_wf %>%
  tune_grid(resamples=folds, grid=grid_of_tuning_params, metrics=metric_set(rmse,mae))
collect_metrics(CV_results) %>%
  filter(.metric=="rmse")
bestTune <- CV_results %>%
  select_best(metric="rmse")

final_wf <- tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

tree_preds <- exp(predict(final_wf, new_data=test_data))

# Format the Predictions for Submission to Kaggle
kag_sub <- tree_preds %>%
  bind_cols(., test_data) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

# Write out the file
vroom_write(x=kag_sub, file="./TreePreds.csv", delim=",")



# Random Forests ---------------------------------------------------------

forest_mod <- rand_forest(mtry = tune(), min_n=tune(), trees=500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

## Create a workflow with model & recipe
forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)

## Set up grid of tuning values
grid_of_tuning_params <- grid_regular(mtry(range=c(1,10)), min_n(), levels=5)

## Set up K-fold CV
folds <- vfold_cv(train_data, v=5, repeats=1)

## Find best tuning parameters
CV_results <- forest_wf %>%
  tune_grid(resamples=folds, grid=grid_of_tuning_params, metrics=metric_set(rmse,mae))
collect_metrics(CV_results) %>%
  filter(.metric=="rmse")
bestTune <- CV_results %>%
  select_best(metric="rmse")

## Finalize workflow and predict
final_wf <- forest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

forest_preds <- exp(predict(final_wf, new_data=test_data))

# Format the Predictions for Submission to Kaggle
kag_sub <- forest_preds %>%
  bind_cols(., test_data) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

# Write out the file
vroom_write(x=kag_sub, file="./ForestPreds.csv", delim=",")




# Boost -----------------------------------------------------------

boost_mod <- boost_tree(tree_depth=tune(), trees=tune(), learn_rate=tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

## Create a workflow with model & recipe
boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_mod)

## Set up grid of tuning values
grid_of_tuning_params <- grid_regular(tree_depth(), trees(), learn_rate(), levels=5)

## Set up K-fold CV
folds <- vfold_cv(train_data, v=5, repeats=1)

## Find best tuning parameters
CV_results <- boost_wf %>%
  tune_grid(resamples=folds, grid=grid_of_tuning_params, metrics=metric_set(rmse,mae))
collect_metrics(CV_results) %>%
  filter(.metric=="rmse")
bestTune <- CV_results %>%
  select_best(metric="rmse")

## Finalize workflow and predict
final_wf <- boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

boost_preds <- exp(predict(final_wf, new_data=test_data))

# Format the Predictions for Submission to Kaggle
kag_sub <- boost_preds %>%
  bind_cols(., test_data) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

# Write out the file
vroom_write(x=kag_sub, file="./BoostPreds.csv", delim=",")



# Bart ------------------------------------------------------------

bart_mod <- parsnip::bart(trees = 100) %>%
  set_engine("dbarts") %>%
  set_mode("regression")

# Create a workflow with model & recipe
bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_mod)

# Set up grid of tuning values
grid_of_tuning_params <- grid_regular(trees(), levels=5)

# Set up K-fold CV
folds <- vfold_cv(train_data, v=5, repeats=1)

# Find best tuning parameters
CV_results <- bart_wf %>%
  tune_grid(resamples=folds, grid=grid_of_tuning_params, metrics=metric_set(rmse,mae))
collect_metrics(CV_results) %>%
  filter(.metric=="rmse")
bestTune <- CV_results %>%
  select_best(metric="rmse")

# Finalize workflow and predict
final_wf <- bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

bart_preds <- exp(predict(final_wf, new_data=test_data))

# Format the Predictions for Submission to Kaggle
kag_sub <- bart_preds %>%
  bind_cols(., test_data) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

# Write out the file
vroom_write(x=kag_sub, file="./BartPreds.csv", delim=",")



# Stacking -------------------------------------------------------

h2o::h2o.init()

auto_mod <- auto_ml() %>%
  set_engine("h2o", max_models=5) %>%
  set_mode("regression")

auto_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(auto_mod) %>%
  fit(data=train_data)

auto_preds <- exp(predict(auto_wf, new_data=test_data))

# Format the Predictions for Submission to Kaggle
kag_sub <- auto_preds %>%
  bind_cols(., test_data) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

# Write out the file
vroom_write(x=kag_sub, file="./AutoPreds.csv", delim=",")












