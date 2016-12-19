setwd("/Users/wang/Desktop/kaggle/kobe-bryant-shot-selection")

# loading data
raw <- read.csv("data.csv", stringsAsFactors = TRUE)
raw$playoffs <- as.factor(raw$playoffs)
raw$numeric_shot_made_flag <- raw$shot_made_flag
raw$shot_made_flag <- as.factor(raw$shot_made_flag)
raw$action_type <- as.factor(raw$action_type)
raw$season <- as.character(raw$season)


# derive new feature
raw$time_remaining <- raw$minutes_remaining * 60 + raw$seconds_remaining
raw$lastsec <- 1*(raw$seconds_remaining == 1)
raw$lastmin <- 1*(raw$minutes_remaining == 0)
raw$away <- 1*grepl('@', raw$matchup)
raw$dist <- sqrt(raw$loc_x^2 + raw$loc_y^2)
raw$close <- 1*(raw$dist < 50)

#split season 
raw$meta_season <- substr(raw$season, 1, 4)
raw$meta_season <- as.numeric(raw$meta_season)

raw$team_id <- NULL
raw$team_name <- NULL
raw$matchup <- NULL
raw$lat <- NULL
raw$lon <- NULL
raw$game_event_id <- NULL
raw$game_id <- NULL
raw$season <- NULL

# Descriptive Statistics
# nrow(raw)

for(i in names(raw)){
  # print(i)
  cat(i, ": ", length(unique(raw[[i]])), "\n" )
}

# unique(raw$combined_shot_type)
# unique(raw$game_id)
# head(raw[,c('lat','lon', 'loc_x', 'loc_y')])

# simple plot accuracy in basketball court
library(plyr)
library(dplyr)
library(ggplot2)

# split test and training set; mytest and mytrain
train <- raw[!is.na(raw$shot_made_flag),]
test <- raw[is.na(raw$shot_made_flag),]

num_mytrain <- floor(nrow(train) * 0.75)
sam <- sample(1:dim(train)[1], num_mytrain)

my_train <- train[sam,]
# dim(my_train)

my_train.shot_made_flag <- my_train$numeric_shot_made_flag
my_test <- train[-(sam),]
dim(my_test)

train.shot_made_flag <- train$numeric_shot_made_flag

# https://thedatagame.com.au/2015/09/27/how-to-create-nba-shot-charts-in-r/

library(grid)
library(jpeg)
# half court image
courtImg.URL <- "nba_court.jpg"
court <- rasterGrob(readJPEG(courtImg.URL),
                    width=unit(1,"npc"), height=unit(1,"npc"))

ggplot(train, aes(loc_x, loc_y)) +
  annotation_custom(court, -250, 250, -50, 420) +
  geom_point(aes(shape = shot_made_flag, color = shot_zone_range ))

shots <- ddply(train, .(shot_zone_range), summarize, 
               SHOTS_ATTEMPTED = length(shot_made_flag),
               SHOTS_MADE = sum(as.numeric(as.character(shot_made_flag))),
               MLOC_X = mean(loc_x),
               MLOC_Y = mean(loc_y))

shots[which(shots$shot_zone_range=="Back Court Shot"),"MLOC_X"] = 0
shots[which(shots$shot_zone_range=="Back Court Shot"),"MLOC_Y"] = 300
shots$SHOT_ACCURACY <- (shots$SHOTS_MADE / shots$SHOTS_ATTEMPTED)
shots$SHOT_ACCURACY_LAB <- paste(as.character(round(100 * shots$SHOT_ACCURACY, 1)), "%", sep="")

# plot shot accuracy per zone
ggplot(shots, aes(x=MLOC_X, y=MLOC_Y)) + 
  annotation_custom(court, -250, 250, -52, 418) +
  geom_text(aes(colour = shot_zone_range, label = SHOT_ACCURACY_LAB)) +
  xlim(250, -250) +
  ylim(-52, 418)

# validate mytrain and mytest
my_model <- glm(shot_made_flag ~ loc_x + loc_y + opponent + 
               combined_shot_type + shot_type + shot_zone_area + shot_zone_basic +
               shot_zone_range + shot_distance + minutes_remaining + seconds_remaining +
               meta_season + time_remaining, data = my_train, family=binomial)

summary(my_model)
my_result <- predict(my_model, my_test, type = "response")

#MultiLogLoss on my_train and my_test
MultiLogLoss <- function(act, pred){
  eps <- 1e-15
  pred <- pmin(pmax(pred, eps), 1 - eps)
  sum(act * log(pred) + (1 - act) * log(1 - pred)) * -1/NROW(act)
}

my_test[, "shot_made_flag"]
my_test$shot_made_flag <- as.numeric(my_test$shot_made_flag)
MultiLogLoss(my_test[, "shot_made_flag"], my_result)
# 0.71052

# actual logit submission model and prediction
model <- glm(shot_made_flag ~ loc_x + loc_y + opponent + 
               combined_shot_type + shot_type + shot_zone_area + shot_zone_basic +
               shot_zone_range + shot_distance + minutes_remaining + seconds_remaining +
               meta_season + time_remaining, data = train, family=binomial)

summary(model)
logit_result <- predict(model, test, type = "response")
# rank 966th for logit only

# cat("Saving the submission file\n")
# submission <- data.frame(shot_id=test$shot_id, shot_made_flag=logit_result)
# write.csv(submission, "logit.csv", row.names = F)

# xgboost
cat("Loading libraries...\n")
library(xgboost)
library(data.table)
library(Matrix)

my_test.shot_id <- my_test$shot_id
my_train$shot_id <- NULL
my_test$shot_id <- NULL
str(my_train.shot_made_flag)
my_train$shot_made_flag <- NULL
my_test$shot_made_flag <- NULL

trainM <- model.matrix(~ combined_shot_type +
                          shot_distance +
                          dist +
                          lastsec +
                          close +
                          lastmin +
                          playoffs +
                          away +
                          shot_type + 
                          opponent +
                          action_type +
                          shot_zone_basic, data = train)[,-1]

testM <- model.matrix(~  combined_shot_type +
                         shot_distance +
                         dist +
                         lastsec +
                         close +
                         lastmin +
                         playoffs +
                         away +
                         shot_type + 
                         opponent +
                         action_type +
                         shot_zone_basic, data = test)[,-1]

Y <- as.numeric(train.shot_made_flag)

nrounds = 400
cv.error <- xgb.cv(data = trainM, label = Y, 
                   objective = "binary:logistic",
                   eval_metric = "logloss",
                   eta = 0.03, 
                   max_depth = 8,
                   nrounds = nrounds,
                   gamma = 1,
                   subsample = 0.8,
                   colsample_bytree = 0.5,
                   min_child_weight = 10,
                   nfold = 10)

min(cv.error$test.logloss.mean)
bestRounds <- which.min(cv.error$test.logloss.mean)

#100 rounds, 6 depth, eta = 0.03, gamma = 0.1, logloss = 0.60739
#100 rounds, 6 depth, eta = 0.03, gamma = 2, logloss = 0.607554
#100 rounds, 6 depth, eta = 0.03, gamma = 1, logloss = 0.607376

#400 rounds, 6 depth, eta = 0.03, gamma = 1, logloss = 0.60357
#400 rounds, 6 depth, eta = 0.03, gamma = 0.1, logloss = 0.60411

xgb_model <- xgboost(data = trainM, label = Y, 
                     objective = "binary:logistic",
                     eval_metric = "logloss",
                     eta = 0.03, 
                     max_depth = 6,
                     nrounds = bestRounds,
                     gamma = 0.5,
                     subsample = 0.8,
                     colsample_bytree = 0.5)

preds <- predict(xgb_model, testM)
# 326th ranking

# final_result <- preds
# final_result[final_result>1] <- 1
# 
# cat("Saving the submission file\n")
# submission <- data.frame(shot_id=test.shot_id, shot_made_flag=final_result)
# write.csv(submission, "solution.csv", row.names = F)

# Monte Carlo simulation - averaging all the xgb model result
set.seed(2016)
ensemble_preds <- replicate(200, {
  model_xgb <- xgboost(data = trainM, label = Y, 
                       objective = "binary:logistic",
                       eval_metric = "logloss",
                       eta = 0.03, 
                       max_depth = sample(7:8,1),
                       nrounds = 400,
                       gamma = 0.1,
                       subsample = sample(c(0.8,1), 1),
                       colsample_bytree = sample(c(0.5,0.8), 1))
  predict(model_xgb, testM)})

mean_ensemble_result <- rowMeans(ensemble_preds)

# solutions <- data.frame(shot_id = test.shot_id, shot_made_flag = mean_ensemble_result)
# write.csv(solutions, "solution.csv", row.names = FALSE)
# 368th ranking

# final_result
# head(final_result)
cat("Saving the submission file\n")

#ensemble all result by giving averaage weight

final_result <- (mean_ensemble_result + preds) / 2
final_result[final_result>1] <- 1
submission <- data.frame(shot_id=test.shot_id, shot_made_flag=final_result)
write.csv(submission, "final_solution.csv", row.names = F)
# 326th ranking
