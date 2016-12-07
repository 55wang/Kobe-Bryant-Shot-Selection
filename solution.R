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

my_train <- train[1:num_mytrain,]
my_train.shot_made_flag <- my_train$numeric_shot_made_flag
my_test <- train[-(1:num_mytrain),]

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

my_trainM<-data.matrix(my_train, rownames.force = NA)
my_dtrain <- xgb.DMatrix(data=my_trainM, label=my_train.shot_made_flag, missing = NaN);
my_watchlist <- list(my_trainM=my_dtrain)

set.seed(2016)
param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "logloss",
                eta                 = 0.035,
                max_depth           = 4,
                subsample           = 0.40,
                colsample_bytree    = 0.40
)

my_clf <- xgb.cv(params             = param, 
                data                = my_dtrain, 
                nrounds             = 1500, 
                verbose             = 1,
                watchlist           = my_watchlist,
                maximize            = FALSE,
                nfold               = 3,
                early.stop.round    = 10,
                print.every.n       = 1
);

my_bestRound <- which.min(as.matrix(my_clf)[,3] + as.matrix(my_clf)[,4]);
cat("Best round:", my_bestRound,"\n");
cat("Best result:",as.matrix(my_clf)[my_bestRound,],"\n")

test.shot_id <- test$shot_id
train$shot_id <- NULL
test$shot_id <- NULL
train$shot_made_flag <- NULL
test$shot_made_flag <- NULL
test$numeric_shot_made_flag <- NULL

trainM<-data.matrix(train, rownames.force = NA)
dtrain <- xgb.DMatrix(data=trainM, label=train.shot_made_flag, missing = NaN);
watchlist <- list(trainM=dtrain)
  
set.seed(2016)
bst <- xgboost(     params              = param, 
                    data                = dtrain, 
                    nrounds             = my_bestRound, 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

testM <- data.matrix(test, rownames.force = NA);
preds <- predict(bst, testM, ntreelimit = my_bestRound);

head(preds)
head(as.vector(logit_result))
final_result <- (0.9*preds + 0.1*as.vector(logit_result))
final_result[final_result>1] <- 1
# 1004th ranking =.=!

# intend to iterate through my_train on both xgboost and logistic to find out the right weightage with my_test.
# might even stacking with another logistic regression. 
# would continue later

# final_result
# head(final_result)

cat("Saving the submission file\n")
submission <- data.frame(shot_id=test.shot_id, shot_made_flag=final_result)
write.csv(submission, "solution.csv", row.names = F)

