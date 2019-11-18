setwd("/Users/clairessabrown/Desktop/data_sci_fun/well_prediction")
library(tidyverse)
library(scales) # For plot axis formatting
library(rworldmap); library(rworldxtra) # for basic graphing of lattitude and longitudes 
library(stringi) # checking for empty strings
library(caret) # machine learning

# Modeling this based on Emily's assignment https://github.com/MagicMilly/predicting-water-pump-functionality/blob/master/student.ipynb
# Will eventually make this into a Jupyter Notebook


# Data source: https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/
# Goal: Predict which pumps are functional, which need some repairs, and which don't work at all
# More info
#### tsh measures of water pressure
#### gps height altitude of welll
#### payment - what the water costs
wp_feats = read.csv("waterpumpfeatures.csv")
train_target = read.csv("train_targets.csv")
#write.table(train_target, "train_targets.csv", quote = F, sep = ",")

# Time to start exploring the data
# First plot well locations across Tanzania
new_map = getMap(resolution = "high")
plot(new_map, 
     xlim = c(30, 50), 
     ylim = c(-11, 0))
points(wp_feats$longitude, wp_feats$latitude, col = "purple", cex = 0.2)


# Visualize the water pressures of these wells (amount_tsh)
range(wp_feats$amount_tsh) # Very wide range, need to filter for more useful visualization
hist(wp_feats$amount_tsh) 

# Results: Most values are zero, this could be because values weren't recorded
wp_feats$amount_tsh[wp_feats$amount_tsh < 100] %>%
  hist()

# Looking at water quality of wells
# Result: Majority of wells observations record water as soft
table(wp_feats$water_quality)
table(wp_feats$quality_group) # Want to see abandoned mines separate from ones that are still in use

wp_feats %>%
  ggplot(aes(water_quality)) + 
  geom_bar() + 
  theme_bw() + 
  labs(x = "Water Quality Grouping", y = "Number of Observations")

# Now for the machine learning part...
# Combine by ids
all_wp_feats = left_join(train_target, wp_feats, by = "id")

# Let's explore model data
table(all_wp_feats$status_group)

all_wp_feats %>%
  ggplot(aes(status_group, fill = status_group)) +
  geom_bar(color = "black") + 
  scale_fill_manual(values = c('#1eb53a', '#fcd116', '#00a3dd')) +
  scale_y_continuous(labels = scales::comma) + 
  theme_bw() + 
  theme(axis.text.x = element_text(vjust = 1, hjust = 1, angle = 15)) + 
  labs(x = "", y = "Number of Observations", fill = "Labels")

# First check for NAs, NULLs or empty spaces
# Result: There are no NAs, NULLs. However, there are several columns with values that are empty spaces
sum(is.na(all_wp_feats))
sum(is.null(all_wp_feats))
apply(all_wp_feats, 2, function(u){
  stringi::stri_isempty(u) %>%
    sum()
})

# Explore whether to delete: Going to remove id because not helpful
length(table(all_wp_feats$scheme_name)) # Too many labels
table(all_wp_feats$scheme_management) # Going to combine empty spaces, None and Other into Unknown label
length(table(all_wp_feats$installer)) # Too many labels
length(table(all_wp_feats$permit))

# This is taking too long so let's go faster
# Get columns that are factors (for machine learning) and non
categoricals = all_wp_feats[,sapply(all_wp_feats, is.factor)]
others = all_wp_feats[,!sapply(all_wp_feats, is.factor)]

# Clean factor data
# Get list of how many labels are in the factor
uniques = sapply(categoricals, function(e){
  length(table(e))
})

low_categoricals = categoricals[,which(uniques < 10)]

uniques[uniques < 10]

# Need to decide what to do with empty spaces in public_meeting, permit: Changing to "Unknown"
sapply(low_categoricals, table)


low_categoricals$public_meeting_edited = replace(as.character(low_categoricals$public_meeting),
                                                      stri_isempty(low_categoricals$public_meeting),
                                                      "Unknown") %>%
  as.factor()

low_categoricals$permit_edited = replace(as.character(low_categoricals$permit),
                                                 stri_isempty(low_categoricals$permit),
                                                 "Unknown") %>%
  as.factor()

glimpse(low_categoricals) # check 
table(low_categoricals$permit_edited) # check

# Clean numeric data
# Region_code and district_code are discrete values, want to convert to factor
length(table(others$region_code)) # 27 region codes
length(table(others$district_code)) # 20 district codes
others$region_code = as.factor(others$region_code)
others$district_code = as.factor(others$district_code)

# 0s in population may be in locations that are passed through by nomadic tribes
# Convert 0s in construction_year to "Unknown", then convert years to factors
others$construction_year_edited = replace(as.character(others$construction_year),
                                   others$construction_year == 0,
                                   "Unknown") %>%
  as.factor()
  

# Combined dataframes again
all = cbind(low_categoricals, others)

# Dropping feautures that have too many labels or are redundant
fil.all = all[,!names(all) %in% c("id",
                                  "scheme_name", 
                                  "scheme_management",
                                  "installer",
                                  "recorded_by",
                                  "payment_type",
                                  "quality_group",
                                  "quantity_group",
                                  "waterpoint_type_group",
                                  "public_meeting",
                                  "permit",
                                  "num_private",
                                  "construction_year"
                                  )]

# Try new package
library(DataExplorer)

# Create report
create_report(fil.all)

# Supervised machine learning: Going first try using decision trees
# Helpful links: https://machinelearningmastery.com/pre-process-your-dataset-in-r/
# Helpful links: 

library(caret)
library(randomForest)
# First convert outcome (dependent) variable to numeric
fil.all$status_group = fil.all$status_group %>%
  as.numeric()

# Check; result: Numbers look good
table(all$status_group)

# Perform one-hot encoding to convert categorical variables to numeric ones for the algorithm
dummys = dummyVars( ~ ., data = fil.all)

dummys.fil.all = predict(dummys, newdata = fil.all, fullRank = T) %>%
  as.data.frame()

dummys.fil.all$status_group = dummys.fil.all$status_group %>%
  as.factor()

# What are our data's new dimensions
dummys.fil.all %>%
  as.data.frame() %>%
  dim()

# First split data into training (75%) and testing (25%) sets
# argument p is the percentage of data that goes to training
index = createDataPartition(dummys.fil.all$status_group, p = 0.75, list = F)
trainset = dummys.fil.all[index,]
testset = dummys.fil.all[-index,]

# Get help from random forest for feature selection
feat_select_control = rfeControl(functions = rfFuncs,
                                 method = "repeatedcv",
                                 repeats = 3,
                                 verbose = T)

predictors = names(fil.all)[!names(fil.all) %in% "status_group"]

# Started at 11:15pm
# feat_select = rfe(trainset[,2:ncol(trainset)], 
#                   trainset$status_group,
#                   rfeControl = feat_select_control)

# save.image("ML_job.RData") # Took about 15 hours

# Look at results
print(feat_select)

# Get best variables
rf_best_names = names(dummys.fil.all)[grep("quantity", names(dummys.fil.all))]
rf_best_names = c(rf_best_names, "latitude", "longitude", "status_group","gps_height")
rf_best = dummys.fil.all[,rf_best_names]

# Get random sample of 10,000 to work with and then expand to 50,000 later
select_rf_best = rf_best[sample(nrow(rf_best), 10000),]

# Split data on our smaller dataset
index_select_rf_best = createDataPartition(select_rf_best$status_group,
                                           p = 0.75,
                                           list = F)
train_select_rf_best = select_rf_best[index_select_rf_best,]
test_select_rf_best = select_rf_best[-index_select_rf_best,]

# Run random forest model for prediction now
# Accuracy: 67%
rf_mod = train(status_group ~ .,
               data = train_select_rf_best,
               method = "rf") 
rf_mod

# Functional (1) is being confused with Needs Repair (3)
confusionMatrix.train(rf_mod)


rf_mod_predict = predict(rf_mod, newdata = test_select_rf_best)

# Accuracy: 66%
gbm_mod = train(status_group ~ .,
                train_select_rf_best,
               method = "gbm") 

# Same as random forest model
confusionMatrix(gbm_mod)

gbm_mod_predict = predict(gbm_mod, newdata = test_select_rf_best)


# Accuracy: 65%
cart_mod = train(status_group ~ .,
                 data = train_select_rf_best,
                method = "rpart") 

confusionMatrix(cart_mod)

cart_mod_predict = predict(cart_mod, newdata = test_select_rf_best)


# Same results as others
# Accuracy: 0.6583
xgboost_mod = train(status_group ~ .,
                 data = train_select_rf_best,
                 method = "xgbTree") 

xgboost_mod_predict = predict(xgboost_mod, newdata = test_select_rf_best)
cart_mod_predict = predict(cart_mod, newdata = test_select_rf_best)
gbm_mod_predict = predict(gbm_mod, newdata = test_select_rf_best)
rf_mod_predict = predict(rf_mod, newdata = test_select_rf_best)

# Reframe model results in terms of sensitivity and specificity: can the models correctly identify every class when appropriate?
observed_xg = test_select_rf_best$status_group
predicted_xg = xgboost_mod_predict

# Classification class refresher
levels(all$status_group)
# [1] "functional"              "functional needs repair" "non functional" 


# I can see that the model erroneously classifies the second class (non-functional needs repair) as the first class (functional) across all models
confusionMatrix(data = predicted_xg, reference = observed_xg, mode = "prec_recall")
confusionMatrix(data = cart_mod_predict, reference = observed_xg, mode = "prec_recall")
confusionMatrix(data = gbm_mod_predict, reference = observed_xg, mode = "prec_recall")
confusionMatrix(data = rf_mod_predict, reference = observed_xg, mode = "prec_recall") # Every class confused classified pretty incorrrectly


# This is most likely due to the imbalance of it where "functional needs repair" constitutes only 7% of the dataset  
table(all_wp_feats$status_group)[names(table(all_wp_feats$status_group)) %in% "functional needs repair"]/sum(table(all_wp_feats$status_group)) # 0.07267677 

# ...while functional is 54% and non functional is 38%
table(all_wp_feats$status_group)[names(table(all_wp_feats$status_group)) %in% "functional"]/sum(table(all_wp_feats$status_group)) # 0.07267677 
table(all_wp_feats$status_group)[names(table(all_wp_feats$status_group)) %in% "non functional"]/sum(table(all_wp_feats$status_group)) # 0.07267677 

# Get more classification stats
multiClassSummary(data = data.frame("obs" = observed_xg, "pred" = predicted_xg), lev = c("1","2","3"), model = "xgbTree")

