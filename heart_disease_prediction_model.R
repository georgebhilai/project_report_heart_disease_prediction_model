library(readr) 
library(dplyr)
library(ggplot2)
library(lmtest)
library(caret)
library(car)
library(MLmetrics)
library(class)
library(GGally)

heart <- read.csv("/Users/georgemathew/Desktop/Desktop Folders/HP BACKUP/C DRIVE/Desktop/Kerala Summons/GEM/edX courses SBI/Data Science HarvardX/choose_your_own_project/Heart Disease_log_regrr_N_KNN/heart.csv")
heart %>% head()

str(heart)

heart <- heart %>% mutate(sex = as.factor(sex), cp = as.factor(cp), fbs = as.factor(fbs), restecg = as.factor(restecg), exang = as.factor(exang), slope = as.factor(slope), ca = as.factor(ca), thal = as.factor(thal), target = as.factor(target))
str(heart)

prop.table(table(heart$target))

anyNA(heart)

colSums(is.na(heart))

a <- ggcorr(heart, label = TRUE, label_size = 2.5, hjust = 1, layout.exp = 2)
plot(a)

RNGkind(sample.kind = "Rounding") 
set.seed(231)
index <- sample(x = nrow(heart), size = nrow(heart)*0.75)
heart_train <- heart[index,]
heart_test <- heart[-index,]
prop.table(table(heart_train$target))
     
summary(heart)

heart_train_x <- heart_train %>% select_if(is.numeric)
heart_test_x <- heart_test %>% select_if(is.numeric)
heart_train_y <- heart_train[,"target"]
heart_test_y <- heart_test[,"target"]
heart_train_xs <- scale(heart_train_x)
heart_test_xs <- scale(heart_test_x, center = attr(heart_train_xs, "scaled:center"), scale = attr(heart_train_xs, "scaled:scale"))

heart_pred_lr <- glm(target ~ ., data = heart_train, family = "binomial")
heart_model_step <- step(object = heart_pred_lr, direction = "backward",trace = F)

summary(heart_model_step)

heart_test$pred_lr <- predict(heart_model_step, heart_test, type = "response")
heart_test$pred_label_lr <- ifelse(heart_test$pred_lr >= 0.5, yes = 1, no = 0)
heart_test %>% select(target, pred_lr, pred_label_lr) %>% rmarkdown::paged_table()
sqrt(nrow(heart_train_xs))

heart_pred_knn <- knn(train =  heart_train_xs, test = heart_test_xs, cl = heart_train_y, k = 27)

confusionMatrix(data = as.factor(heart_test$pred_label_lr), reference = heart_test$target, positive = "1") 

confusionMatrix(data = heart_pred_knn, reference = heart_test_y, positive = '1')

