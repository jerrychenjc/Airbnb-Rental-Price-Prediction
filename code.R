## Introduction
#This project is basically the process of making prediction for the price of Airbnb rentals in New York City. By providing a data set called "analysisData",
#which lists over 25,000 Airbnb rentals information including 90 variables like price, id, accommodates, bed type, etc., participants are required to 
#explore these data, clean and arrange the useful data, construct efficient models, and apply models to the test data which is called "scoringData" to make 
#the prediction for price. The final goal is to generate a data set contains the id number of test data and prediction of price, and the accuracy is 
#assessed by Root Mean Square Error (RMSE).

## Exploring the Data
#Before using the data provided, the action needed is to explore the data in order to remove the unuseful variables from data. To better understand the 
#types and properties of analysisData, I constructed a new data frame which shows type of data, whether it is a factor, number of levels, whether it is a 
#missing value (NA), and the number of NAs for each variables.

#read analysis and scoring data
library(dplyr)
setwd("/Users/Jerry/Desktop/5200/R code/Kaggle_Final/")
getwd()
analysisData = read.csv('analysisData.csv')
scoringData = read.csv('scoringData.csv')

# Examing data ------------------------------------------------------------
colNames = colnames(analysisData)
colDataType = sapply(analysisData, class)
isColFactor = sapply(analysisData, is.factor)
numOfLevels = sapply(analysisData, function(x) length(levels(x)))
isNAincluded = sapply(analysisData, function(x) any(is.na(x)))
numOfNAs = sapply(analysisData, function(x) sum(is.na(x)))

colSummary = data.frame(colDataType, isColFactor, numOfLevels, isNAincluded, numOfNAs)
str(colSummary)

## Preparing the Data for Analysis
#Before using the data from "analysisData", it is better to manually prepare the data. This will basically shrink the data set and increase the efficiency
#for R to run algorithms in in terms of the models that would be used for predicting. By looking at the features of the variables in original data sets, I
#found there are two variables that may be useful for prediction but was not presented in a measurable way, which are "amentities" and "host_verifications".
#Therefore I change they to numeric format by counting the item numbers in each cell. 

library(stringr)
analysisData_changed = analysisData
scoringData_changed = scoringData
analysisData_changed$amenities = str_count(analysisData_changed$amenities, ',') + 1
analysisData_changed$host_verifications = str_count(analysisData_changed$host_verifications, ',') + 1

scoringData_changed$amenities = str_count(scoringData_changed$amenities, ',') + 1
scoringData_changed$host_verifications = str_count(scoringData_changed$host_verifications, ',') + 1

#The initial selection of data is to remove some useless variables from the data First of all, I removed the columns which is factor but only have 1 level,
#because this type of variable will not affect the result of prediction since there is only one category. Next, I removed the factorial variables which has
#more than 600 levels since many regression models can not handle variable which contains too many categories. Then I removed variables that contains more
#than 12,000 missing values, because 12,000 is almost the half number of rows in train data set and they will decrease the accuracy of prediction. 

# Examing data again ------------------------------------------------------------
colNames = colnames(analysisData_changed)
colDataType = sapply(analysisData_changed, class)
isColFactor = sapply(analysisData_changed, is.factor)
numOfLevels = sapply(analysisData_changed, function(x) length(levels(x)))
isNAincluded = sapply(analysisData_changed, function(x) any(is.na(x)))
numOfNAs = sapply(analysisData_changed, function(x) sum(is.na(x)))

colSummary = data.frame(colDataType, isColFactor, numOfLevels, isNAincluded, numOfNAs)

# Remove useless columns 
colSelect = colSummary[which(numOfLevels != 1 & numOfLevels < 600 & numOfNAs < 12000),]
analysisDataSelected = analysisData_changed %>% select(rownames(colSelect))
scoringDataSelected = scoringData_changed %>% select(setdiff(rownames(colSelect), 'price'))

#The next step is to fill the missing values. Since in factorial variables, missing value are automatically regarded as a factor, there is no need to 
#replace them. Now let's move attention to numerical variables. There are three numerical variables contains missing values, which respectively are bed,
#security-deposit and cleaning-fee.I replace the NAs in bed with median of the all the non-NA values in the same column. But for the security-deposit and 
#cleaning-fee, I replace the NAs with 0 since I assume the missing value means these sepecific rentals does not have security-deposit and cleaning-fee.

#replace missing values (NAs)
analysisDataSelected[is.na(analysisDataSelected$beds),"beds"] = median(analysisDataSelected[,"beds"], na.rm = TRUE)
scoringDataSelected[is.na(scoringDataSelected$beds),"beds"] = median(scoringDataSelected[,"beds"], na.rm = TRUE)

analysisDataSelected[is.na(analysisDataSelected$security_deposit),"security_deposit"] = 0
scoringDataSelected[is.na(scoringDataSelected$security_deposit),"security_deposit"] = 0

analysisDataSelected[is.na(analysisDataSelected$cleaning_fee),"cleaning_fee"] = 0
scoringDataSelected[is.na(scoringDataSelected$cleaning_fee),"cleaning_fee"] = 0

## Modeling Techniques
#After the data was ideally cleaned, now it is the most essential part in the whole project, which is fitting data into models. Before using model, 
#we need to determine which variables are going to be used as indicators for the model. Here I used Lasso as a feature selection method to list the reduced
#number of explanatory variable to describe a response variable, which is price. The advantage of Lasso is that it performs better than the other selection,
#performs variable selection to any number of variables, and it is relatively fast. Feature selection also can redce overfitting of model. Following are the
#variables came from Lasso:

##Feature Selection, Lasso
library(glmnet)
x = model.matrix(price~., data = analysisDataSelected)
y = analysisDataSelected$price
set.seed(100)
cv.lasso = cv.glmnet(x,y,alpha=1,nfolds=10)
coef(cv.lasso)

coefs = coef(cv.lasso, s = 'lambda.min') 
#coefs
index = which(coefs != 0)
variables = row.names(coefs)[index]
variables = variables[!(variables %in% '(Intercept)')]; variables

#variables selected from the result of Lasso
variables_list = c("id","host_id","host_response_time","host_response_rate","host_is_superhost","host_neighbourhood","host_listings_count",
                   "host_total_listings_count","host_verifications","host_has_profile_pic","host_identity_verified","street","neighbourhood",
                   "neighbourhood_cleansed","neighbourhood_group_cleansed","city","zipcode","market","smart_location","longitude","is_location_exact",
                   "property_type","room_type","accommodates","bathrooms","bedrooms","beds","bed_type","amenities","security_deposit","cleaning_fee",
                   "guests_included","minimum_nights","calendar_updated","availability_30","availability_90","availability_365","calendar_last_scraped",
                   "number_of_reviews","review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin",
                   "review_scores_communication","review_scores_location","review_scores_value","jurisdiction_names","is_business_travel_ready",
                   "cancellation_policy","require_guest_phone_verification","calculated_host_listings_count","reviews_per_month")
variables_list

#Than I used random forest as a decision tree model to train the data instead of simple linear regression model, since it not only can be used for both 
#classification and regression problems, but also reduces overfitting and reduce variance by using mutiple trees, which would esentially increased the 
#accuracy of results. Since only the categorical variables with less than 53 categories can be applied to random forest model, I mannual removed the 
#variables that don't satisfy the requrement of the model. Finally the model was constructed below:

library(randomForest)
#construct random forest model
forest = randomForest(price ~ host_response_time+host_is_superhost+host_listings_count+host_total_listings_count+host_verifications+host_has_profile_pic+
                        host_identity_verified+neighbourhood_group_cleansed+market+longitude+is_location_exact+property_type+room_type+accommodates+
                        bathrooms+bedrooms+beds+bed_type+amenities+security_deposit+cleaning_fee+guests_included+minimum_nights+availability_30+
                        availability_90+availability_365+calendar_last_scraped+number_of_reviews+review_scores_rating+review_scores_accuracy+
                        review_scores_cleanliness+review_scores_checkin+review_scores_communication+review_scores_location+review_scores_value+
                        jurisdiction_names+is_business_travel_ready+cancellation_policy+require_guest_phone_verification+calculated_host_listings_count+
                        reviews_per_month,
                      data = analysisDataSelected, ntree = 1000)

## Results
#After the random forest model is successfully trained, it is able to use the random forest model to predict the prices based on the test data. But there 
#is still a issue, which is some catagorical variables in test data contains the new factors that the train data does not have. So I needed to set the 
#levels of test data and make them equal to the levels of train data.

#set level of test data 
for (k in colnames(analysisDataSelected)) { 
  if (class(scoringDataSelected[[k]]) == "factor") { 
    levels(scoringDataSelected[[k]]) = levels(analysisDataSelected[[k]]) 
  }
}

#Finally, the prediction was maded.The table constructed below shows the rmse of the train data which is calculated by using the prediction based on the 
#train data set, and the rmse of the test data which is evaluated by Kaggle competition website. The reason why I calculated rmse of the train data is 
#that I always use the rmse of the train as a reference to the rmse provided by Kaggle in each attempt I tried for a new method.

#predict
predForest = predict(forest, newdata = scoringDataSelected)
pred0 = predict(forest)
rmse = sqrt(mean((pred0-analysisDataSelected$price)^2));rmse

# construct submision from predictions
submissionFile = data.frame(id = scoringDataSelected$id, price = predForest)
write.csv(submissionFile, 'sample_submission_Final.csv',row.names = F)

##Discussion
#Looking bakward to the whole process of doing this interesting competition, it is fun to have every attempts that tried to make the prediction as 
#accurate I can. Especially when everytime I tried a model or a new way to select variables, it is always appreciated to see the improvements. 
#Following tables shows the scores of each attempts I got and the method I used for each attempts. I wan to highlited the 3 most significant changes that 
#I have made when I got new improvements. In the beginning, I used linear model, and I got the prediction result with rmse around 65. And I have to say 
#feature selection is the most useful especially lasso, which reduced the rmse of my result based on linear model to around 60. Then I tried the 
#complecated model which is random forest based on the variables selected by Lasso, this method reduced the rmse by 3. Meanwhile, after comparing the 
#results of my experiments of using randomforests, I found picking 1000 as the number of trees gave the best prediction.




