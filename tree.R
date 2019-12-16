## Predict wins in NCAA tournament with XGBoost

library(dplyr)
library(ggplot2)
library(recipes)
library(rsample)
library(xgboost)
library(coefplot)
library(DiagrammeR)
library(iml)

set.seed(1234)


## import stats for seasons 2015 thru 2019
df <- read.csv("cbb.csv")

head(df, 20)

df_clean <- df %>% filter(!is.na(SEED))
nrow(df_clean)
table(df_clean$POSTSEASON)

## We are going to predict which roound a team advances to on a continuous scale
df_clean$Tourney_Round <- if_else(df_clean$POSTSEASON %in% c("R68", "R64"), 1,
                                  if_else(df_clean$POSTSEASON %in% c("R32"), 2,
                                          if_else(df_clean$POSTSEASON %in% c("S16"), 3,
                                                  if_else(df_clean$POSTSEASON %in% c("E8"), 4,
                                                          if_else(df_clean$POSTSEASON %in% c("F4"), 5,
                                                                  if_else(df_clean$POSTSEASON %in% c("2ND"), 6,
                                                                          if_else(df_clean$POSTSEASON %in% c("Champions"), 7, 0)))))))

ggplot(df_clean, aes(x=Tourney_Round)) + geom_histogram()


## Data prep
## We want to predict wins in tournament, so for training purposes, back out wins that occur while in the tournament
df_clean$W_RegSeason <- df_clean$W - (df_clean$Tourney_Round - 1)

## Create train/test split
the_split <- initial_split(data = df_clean, prop = 0.8, strata='Tourney_Round')

train <- training(the_split)
test <- testing(the_split)


xgb_recipe <- recipe(Tourney_Round ~ ., data=train) %>%
  step_rm(TEAM, G, W, POSTSEASON, YEAR) %>%
  step_zv(all_predictors()) %>% 
  step_other(all_nominal()) %>% 
  step_dummy(all_nominal(), one_hot=TRUE)

xgb_recipe

xgb_prepped <- xgb_recipe %>% prep()
xgb_prepped

train_x <- xgb_prepped %>% bake(all_predictors(), new_data=train, composition='matrix')
train_y <- xgb_prepped %>% bake(all_outcomes(), new_data=train, composition='matrix')

test_x <- xgb_prepped %>% bake(all_predictors(), new_data=test, composition='matrix')
test_y <- xgb_prepped %>% bake(all_outcomes(), new_data=test, composition='matrix')

train_xgb <- xgb.DMatrix(data = train_x, label=train_y)
test_xgb <- xgb.DMatrix(data = test_x, label=test_y)

xgb_model_1 <- xgb.train(
  data = train_xgb,
  objective="reg:tweedie",
  tweedie_variance_power = 2.0,
  #objective="reg:squarederror",
  eta=0.1,
  nrounds=500,
  print_every_n = 5,
  watchlist = list(train=train_xgb, validate=test_xgb),
  eval_metric = 'mae',
  max_depth = 10,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model_1$evaluation_log %>% dygraphs::dygraph()
best_ntree <- which.min(xgb_model_1$evaluation_log$validate)
best_ntree

xgb_preds <- predict(xgb_model_1,test_x, ntreelimit = best_ntree)

## RMSE:
sqrt(mean((test_y - xgb_preds)^2))
## MAE:
mean(abs(test_y - xgb_preds))

importance_model <- xgb.importance(model = xgb_model_1)
importance_model$Feature

###############################
## Analysis with IML package ##
###############################
predictor = Predictor$new(xgb_model_1, data = data.frame(test_x) %>% select(one_of(xgb_model_1$feature_names)),
                          y = data.frame(test_y),
                          predict.fun=function(model, newdata){
                            newData_x = xgb.DMatrix(data.matrix(newdata), missing = NA)
                            results<-predict(model, newData_x)
                            return(results)
                          })

## IML feature importance measure works by shuffling each feature and measuring how much the performance drops
imp = FeatureImp$new(predictor, loss = "mae")
plot(imp)


## Partial Dependency plots
effect = FeatureEffect$new(predictor, method = "pdp", feature = "BARTHAG", grid.size = 100)
effect$plot()
effect$set.feature("WAB")
effect$plot()
effect$set.feature("ADJDE")
effect$plot()
effect$set.feature("ADJOE")
effect$plot()
effect$set.feature("EFG_D")
effect$plot()


## Explain single predictions with Shapley values
shapley.xgb = Shapley$new(predictor, x.interest = data.frame(test_x)[1,])
shapley.xgb$plot()

shapley.xgb$explain(x.interest = data.frame(test_x)[50,])
shapley.xgb$plot()
