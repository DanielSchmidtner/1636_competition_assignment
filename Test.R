library(mlbench)
library(caret)

data(BostonHousing)
#exclude one factor column
tr_dat = BostonHousing[1:300,-4]
test_dat = BostonHousing[301:nrow(BostonHousing),-4]

fitControl = trainControl(method = "cv",number = 5)

glmnet.tuneGrid = expand.grid(alpha = seq(from = 0, to = 1, by = 0.2),
                              lambda = seq(from = 0, to = 1, by = 0.2))

glmnet.fit = train(x = tr_dat[,-ncol(tr_dat)], y = tr_dat[,ncol(tr_dat)], 
                   method = "glmnet",etric = "RMSE",trControl = fitControl,tuneGrid = glmnet.tuneGrid)


# Caret prediction:

pred_caret = predict(glmnet.fit,newdata=test_dat)

# We do the manual prediction, so you can get it by do a matrix multiplication %*% between your coefficients and predictor matrix:
  
  predictor = cbind(Intercept=1,as.matrix(test_dat[,-ncol(test_dat)]))
coef_m = as.matrix(coef(glmnet.fit$finalModel,s=glmnet.fit$bestTune$lambda))
pred_manual = predictor %*% coef_m

table(pred_manual == pred_caret)

# TRUE 
# 206 

# You get back exactly the same