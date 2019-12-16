## Predict wins in NCAA tournament

library(dplyr)
library(ggplot2)
library(recipes)
library(rsample)
library(glmnet)
library(coefplot)

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

TR_recipe <- recipe(Tourney_Round ~ . , data = train) %>% 
  step_rm(TEAM, G, W, POSTSEASON, YEAR) %>%
  step_log(Tourney_Round) %>% 
  step_zv(all_predictors()) %>%  
  step_knnimpute(all_predictors()) %>% 
  step_normalize(all_numeric(), -Tourney_Round) %>% 
  step_other(all_nominal()) %>% 
  step_dummy(all_nominal(), one_hot=TRUE)

TR_recipe

TR_prepped <- TR_recipe %>% prep(data=train)
TR_prepped

TR_prepped %>% bake(new_data=train)

train_x <- TR_prepped %>% bake(all_predictors(), new_data=train, composition='matrix')
head(train_x)
train_y <- TR_prepped %>% bake(all_outcomes(), new_data=train, composition='matrix')

TR_model_1 <- glmnet(x=train_x, y=train_y, family='gaussian', standardize = FALSE)

plot(TR_model_1, xvar = 'lambda')

## Show interactive plot
coefpath(TR_model_1)
coefplot(TR_model_1, sort='magnitude')   

TR_model_2 <- cv.glmnet(x=train_x, y=train_y, family='gaussian', standardize=FALSE, nfolds=5)
plot(TR_model_2)
TR_model_2$lambda
coefpath(TR_model_2)
TR_model_2$lambda.min
TR_model_2$lambda.1se

coefplot(TR_model_2, sort='magnitude', lambda='lambda.1se', intercept = FALSE, plot=FALSE)

TR_model_3 <- cv.glmnet(x=train_x, y=train_y, family='gaussian', nfolds=5, standardize=FALSE, alpha=0)
coefpath(TR_model_3)
TR_model_4 <- cv.glmnet(x=train_x, y=train_y, family='gaussian', nfolds=5, standardize=FALSE, alpha=0.7)
coefpath(TR_model_4)
## note:  lasso (alpha = 1.0) better for model interpretation

test_x <- TR_prepped %>%  bake(all_predictors(), new_data=test, composition='matrix')
test_y <- TR_prepped %>%  bake(all_outcomes(), new_data=test, composition='matrix')

TR_preds <- predict(TR_model_4, newx=test_x, s='lambda.1se')
head(exp(TR_preds))

test %>% mutate(Predicted_Round=as.vector(exp(TR_preds))) %>% select(TEAM, Tourney_Round, Predicted_Round)

## RMSE:
sqrt(mean((test_y - TR_preds)^2))
## MAE:
mean(abs(test_y - TR_preds))

