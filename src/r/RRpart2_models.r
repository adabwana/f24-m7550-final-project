library(here)
library(dplyr)
library(car)
library(readr)
library(caret)
library(doParallel)
library(R.utils)
library(splines)
nCores<-detectCores()
cl<-makeCluster(nCores)
registerDoParallel(cl)


engineered_data <- readr::read_csv(here("data", "LC_engineered.csv"))
part_2_data <- readr::read_csv(here("data", "part_2_data.csv"))
part_2_data <- as.data.frame(part_2_data)
str(part_2_data)
dim(part_2_data)


categorical_factors <- c("Gender", "Semester", "Day_of_Week",
"Course_Level", "Underclassman", "Expected_Graduation_Yr")

# Convert categorical columns to factors
part_2_data[categorical_factors] <- lapply(part_2_data[categorical_factors], as.factor)

# take absolute value of Duration_In_Min
#part_1_data$Duration_In_Min <- abs(part_1_data$Duration_In_Min)
View(part_2_data)

# show structure of data
str(part_2_data)
dim(part_2_data)



partition.2 <- function(data, prop.train){
  selected <- sample(1:nrow(data), round(nrow(data)*prop.train), replace = FALSE) 
  data.train <- data[selected,]
  rest <- setdiff(1:nrow(data), selected)
  data.test <- data[rest,]
  return(list(data.train=data.train, data.test=data.test))
}

RNGkind (sample.kind = "Rounding") 
set.seed(42)
p2 <- partition.2(part_2_data, 0.8) ## 80:20 split
training.data <- p2$data.train
test.data <- p2$data.test

trControl <- trainControl(
  method = "cv",
  number = 10,
  allowParallel = TRUE
)



############## MULTIPLE LINEAR REGRESSION ################
mlr <- lm(Occupancy ~ ., data = training.data)
summary(mlr)

mlr2 <- lm(Occupancy ~ . -Expected_Graduation_Yr -Change_in_GPA -Term_Credit_Hours 
                        -Semester_Week -Gender -Total_Credit_Hours_Earned -Term_GPA 
                        -Cumulative_GPA -Underclassman -Total_Visits,
                        data = training.data)
summary(mlr2)

anova(mlr, mlr2)

# Predictions
mlr_pred <- predict(mlr, newdata = test.data)
mlr2_pred <- predict(mlr2, newdata = test.data)

# RMSE
rmse_mlr <- sqrt(mean((test.data$Occupancy - mlr_pred)^2))
rmse_mlr

rmse_mlr2 <- sqrt(mean((test.data$Occupancy - mlr2_pred)^2))
rmse_mlr2





############ MLR Forward Selection ##############
nVar <- ncol(training.data)-1
mlr_forward <- caret::train(Occupancy ~ ., data=training.data,
             method="leapForward",
             tuneGrid=data.frame(nvmax=1:nVar),
             trControl=trControl
             )

#print(mlr_forward)


mlr_forward_Summary <- summary(mlr_forward$finalModel)
mlr_forward_Summary$outmat # provides the order in which the variables enter the model
nBest <- mlr_forward$bestTune$nvmax # size of the best model
coef(mlr_forward$finalModel,nBest) # coefficient estimates of the best model

predY <- predict(mlr_forward, newdata = test.data) # predictions on the test set
testY <- test.data[,'Occupancy'] # true values on the test set
mlr_forward_RMSE <- sqrt(mean((predY-testY)^2)) # test RMSE (square roof of MSE)
mlr_forward_RMSE



############ MLR Backward Selection ##############
nVar <- ncol(training.data)-1
mlr_backward <- caret::train(Occupancy ~ ., data=training.data,
             method="leapBackward",
             tuneGrid=data.frame(nvmax=1:nVar),
             trControl=trControl
             )

#print(mlr_backward)


mlr_backward_Summary <- summary(mlr_backward$finalModel)
mlr_backward_Summary$outmat # provides the order in which the variables enter the model
nBest <- mlr_backward$bestTune$nvmax # size of the best model
coef(mlr_backward$finalModel,nBest) # coefficient estimates of the best model

predY <- predict(mlr_backward, newdata = test.data) # predictions on the test set
testY <- test.data[,'Occupancy'] # true values on the test set
mlr_backward_RMSE <- sqrt(mean((predY-testY)^2)) # test RMSE (square roof of MSE)
mlr_backward_RMSE




############## RIDGE ################

set.seed(0)
mlr_ridge <- caret::train(Occupancy ~ ., data = training.data,
             method = 'glmnet',
             preProcess = c("center","scale"),
             tuneGrid = expand.grid(alpha = 0, lambda = 10^seq(-4, 4, by = 0.1)),
             trControl=trControl
             )

print(mlr_ridge)

coef(mlr_ridge$finalModel,as.numeric(mlr_ridge$bestTune[2])) # estimated coefficients
predY <- predict(mlr_ridge,newdata = test.data) # predictions on the test set
testY <- test.data[,'Occupancy'] # true values on the test set
mlr_ridge_RMSE <- sqrt(mean((predY-testY)^2)) # test RMSE
mlr_ridge_RMSE



############## LASSO ################

set.seed(0)
mlr_lasso <- caret::train(
  Occupancy ~ .,
  data = training.data,
  method = 'glmnet',
  preProcess = c("center","scale"),
  tuneGrid = expand.grid(alpha = 1, lambda = 10^seq(-4, 4, by = 0.1)),
  trControl=trControl
  ) 

print(mlr_lasso)

coef(mlr_lasso$finalModel,as.numeric(mlr_lasso$bestTune[2])) # estimated coefficients
predY <- predict(mlr_lasso,newdata = test.data) # predictions on the test set
testY <- test.data[,'Occupancy'] # true values on the test set
mlr_lasso_RMSE <- sqrt(mean((predY-testY)^2)) # test RMSE
mlr_lasso_RMSE




############## ELASTIC NET REGRESSION ################
set.seed(0)
tuneGrid_elastic <- expand.grid(
  alpha = seq(0, 1, by = 0.1),
  lambda = 10^seq(-4, 4, by = 0.1)
)

mlr_elastic <- caret::train(
  Occupancy ~ ., 
  data = training.data,
  method = 'glmnet',
  preProcess = c("center", "scale"),
  tuneGrid = tuneGrid_elastic,
  trControl = trControl
) 

print(mlr_elastic)

# Best alpha and lambda
best_alpha <- mlr_elastic$bestTune$alpha
best_lambda <- mlr_elastic$bestTune$lambda
cat("Best alpha:", best_alpha, "\n")
cat("Best lambda:", best_lambda, "\n")

# Extract coefficients for the best lambda
elastic_coefficients <- coef(mlr_elastic$finalModel, best_lambda)
#print(elastic_coefficients)

# Predictions on the test set
predY_elastic <- predict(mlr_elastic, newdata = test.data)

# Calculate RMSE
mlr_elastic_RMSE <- sqrt(mean((predY_elastic - testY)^2))
print(mlr_elastic_RMSE)




############## KNN ################


modelKNN <- train(Occupancy ~ ., 
                  data=training.data,
                  method='knn',
                  preProc=c('center','scale'),
                  tuneGrid=data.frame(k = seq(2, 50, by = 2)), 
                  trControl=trControl,
                  metric='RMSE')

print(modelKNN)
plot(modelKNN)

predY <- predict(modelKNN, newdata = test.data)
testY <- test.data[,'Occupancy']
rmse_knn <- sqrt(mean((predY - testY)^2))
print(rmse_knn)





############## REPORTING RMSE RESULTS ################

# create dataframe with rmse results of each model for comparison
rmse_results <- data.frame(
  model = c("Multiple Linear Regression", "MLR Forward Selection", "Ridge", "Lasso", "Elastic Net", "KNN"),
  rmse = c(rmse_mlr, mlr_forward_RMSE, mlr_ridge_RMSE, mlr_lasso_RMSE, mlr_elastic_RMSE, rmse_knn)
) %>%
  arrange(rmse)
print(rmse_results)
