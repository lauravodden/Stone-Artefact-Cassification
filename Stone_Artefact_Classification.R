getwd()
setwd("C:/Users/laura/OneDrive/Desktop/MA5810/Assignment_3")

R.version

library(ISLR, warn.conflicts = F, quietly = T) 
library(caret, warn.conflicts = F, quietly = T) 
library(dplyr, warn.conflicts = F, quietly = T) 
library(cluster, warn.conflicts = F, quietly = T) 
library(factoextra, warn.conflicts = F, quietly = T)
library(ggplot2)
library(naniar)
library(naivebayes)
library(tidyr)
library(plyr)
library(pROC)

#######################################################################################################################
#                                                      LOAD DATA                                                      #                 
#######################################################################################################################

# Read csv file
data <- read.csv("Ban_Rai_Area_3_lithics.csv")


#######################################################################################################################
#                                                    DATA CLEANING                                                    #                                                   
#######################################################################################################################

# View column names as list for removal
names <- names(data)
names <- as.data.frame(names)

# Remove superfluous columns
data <- data[-c(1:5,7:9,15,17,20:22,26,29:71,73:164)]

# Remove invalid data
data <- na.omit(data)

# Fill empty cells with 'Absent'
data$OVERHANG_R <- sub("^$", "Absent", data$OVERHANG_R)
data$INITIATION <- sub("^$", "Absent", data$INITIATION)
data$BULB_OF_PE <- sub("^$", "Absent", data$BULB_OF_PE)
data$TERMINATIO <- sub("^$", "Absent", data$TERMINATIO)


# Remove the two 'Broken Cbl' entries as they equate to 'retouched flake' and there is no distinction in 
# the data between this and 'flake'
data <- data[- grep("Broken Cbl", data$ARTEFACT_C),]


# Set as factor/Standardise abbreviations within catgorical variables
data$OVERHANG_R <- as.factor(data$OVERHANG_R)
levels(data$OVERHANG_R)

data$INITIATION <- as.factor(data$INITIATION)
levels(data$INITIATION)

data$BULB_OF_PE <- as.factor(data$BULB_OF_PE)
levels(data$BULB_OF_PE)

data$TERMINATIO <- as.factor(data$TERMINATIO)
levels(data$TERMINATIO)

data$RAW_MATERI <- as.factor(data$RAW_MATERI)
data$RAW_MATERI <- revalue(data$RAW_MATERI, c("Blk qtzt"="Black Quartzite", "Rd qtzt"="Red Quartzite", "Yell qtzt"="Yellow Quartzite", "hi-q qtzt"="Quartzite", "vfg Qtzt"="Quartzite", "Fe qtzt"="Quartzite", "Chert blk"= "Chert", "Blk chert"="Chert", "Blk Chrt"="Chert", "chert"="Chert", "Ind mudst"="Shale", "Diorite 1"="Diorite", "Chalc"="Chalk", "Limest"="Limestone"))
levels(data$RAW_MATERI)

# Set response variable as factor
data$ARTEFACT_C <- as.factor(data$ARTEFACT_C)
levels(data$ARTEFACT_C)
########################################################################################################################
#                                              NAIVE-BAYES CLASSIFICATION                                              # 
########################################################################################################################

# View proportions of categories within response variables
table(data$ARTEFACT_C) %>% prop.table()



# Preliminary visualisation: Density plot
numeric_data <- data[-c(10:14)]
numeric_data %>%
  reshape2::melt(id.vars = "ARTEFACT_C") %>%
  ggplot(aes(value, colour = ARTEFACT_C))+
  labs(title="Artefact Class density distributions")+
  geom_density(show.legend = TRUE)+
  facet_wrap(~variable, scales = "free")



# Ensure reproducibility
set.seed(111)

# Split data into test and train
split <- createDataPartition(data$ARTEFACT_C, p = 0.8, list = FALSE)
train <- data[split, ]
test <- data[-split, ]

# Train model on train data
nB_model <- naive_bayes(ARTEFACT_C ~ ., data = train, laplace = 1, kernel = TRUE)

# View predictions and accuracy on test data
pred <- predict(nB_model, test, type = "class")
confusionMatrix(pred, test$ARTEFACT_C)

# View predictions and accuracy on train data
pred <- predict(nB_model, train, type = "class")
confusionMatrix(pred, train$ARTEFACT_C)


########################################################################################################################
#                                                 LOGISTIC REGRESSION                                                  # 
########################################################################################################################

# Isolate dataframe numeric variables
log_data <- data[-c(10:14)]

# Investigate collinearity
cor_tab <- cor(log_data[1:9])

# Revalue Core and Flake to binary factor where Flake = 1, Core = 0
log_data$ARTEFACT_C <- revalue(log_data$ARTEFACT_C, c("Flake"="1", "Core"="0")) 

# Set numeric predictors
names(log_data)
numeric_predictors <- c("MASS__G_", "PROX_WIDTH", "MED_WIDTH_", "DIST_WIDTH", "PROX_THICK", "MED_THICKN", "DIST_THICK", "PLAT_WIDTH", "PLAT_THICK")

set.seed(111)

#split into training (80%) and test (20%)
split <- createDataPartition(log_data$ARTEFACT_C, p = 0.8, list = F)
train <- log_data[split, c(numeric_predictors, "ARTEFACT_C")]
test <- log_data[-split, c(numeric_predictors, "ARTEFACT_C")]

# print number of observations in test vs. train
c(nrow(train), nrow(test))

# Proportions of core vs flake in train data
table(train$ARTEFACT_C) %>% prop.table()

# Proportions of core vs flake in test data
table(test$ARTEFACT_C) %>% prop.table() 

# Train the model on the train data
log_model <- glm(
  ARTEFACT_C ~ MASS__G_ + PROX_WIDTH + MED_WIDTH_ + DIST_WIDTH
  + PROX_THICK + MED_THICKN + DIST_THICK +
    PLAT_WIDTH + PLAT_THICK,
  family = "binomial",
  data = train
)
log_model
summary(log_model)

# Accuracy of train data
lodds <- predict(log_model, type = "link")
preds_lodds <- ifelse(lodds > 0, "1", "0") 
# Accuracy
confusionMatrix(as.factor(preds_lodds), train$ARTEFACT_C)

# Make predictions on test data
test_lodds <- predict(log_model, newdata = test, type = "link")
test_preds_lodds <- ifelse(test_lodds > 0, "1", "0") 
# Accuracy
confusionMatrix(as.factor(test_preds_lodds), test$ARTEFACT_C) 


# Order variables in order of their impact on classification
summary(log_model)

mod_fit <- glm(ARTEFACT_C ~ MASS__G_ + PROX_WIDTH + MED_WIDTH_ + DIST_WIDTH
               + PROX_THICK + MED_THICKN + DIST_THICK +
                 PLAT_WIDTH + PLAT_THICK, data=train, family=binomial(link = 'logit'))

imp <- as.data.frame(varImp(mod_fit))
imp <- data.frame(overall = imp$Overall,
                  names   = rownames(imp))
importance <- imp[order(imp$overall,decreasing = T),]
importance

# Plot ROC curve
test_prob = predict(mod_fit, newdata = test, type = "response")
test_roc = roc(test$ARTEFACT_C ~ test_prob, plot = TRUE, print.auc = TRUE)
########################################################################################################################
#                                           K-MEANS CLUSTER ANALYSIS                                                   # 
########################################################################################################################

# Create cluster dataframe
clust_data <- data[-c(10:14)]

# Scale data
dat_scaled <- scale(clust_data[1:9])

set.seed(111) 

# Train model
kmeans_res <- kmeans(dat_scaled, centers = 2, nstart = 25) 
str(kmeans_res)
fviz_cluster(kmeans_res, data = dat_scaled)

# Accuracy
kmeans_res

# Plot model
fviz_cluster(kmeans_res, 
             data = dat_scaled,
             geom = "point", 
             shape = 19,
             alpha = 0)+ 
  geom_point(aes(colour = as.factor(kmeans_res$cluster),
                 shape = clust_data$ARTEFACT_C))+ 
  ggtitle("Comparing Clusters and Artefact Class") 


# Perform kmeans & calculate ss
total_sum_squares <- function(k){ 
  kmeans(dat_scaled, centers = k, nstart = 25)$tot.withinss
}
# Define a sequence of values for k
all_ks <- seq(1,20,1)
choose_k <- sapply(seq_along(all_ks), function(i){ #apply to all values
  total_sum_squares(all_ks[i])
})
choose_k_plot <- data.frame(k = all_ks, # dataframe for plotting
                            within_cluster_variation = choose_k)
head(choose_k_plot)
ggplot(choose_k_plot, aes(x = k, # plot
                          y = within_cluster_variation))+
  geom_point()+
  geom_line()+
  xlab("Number of Clusters (K)")+
  ylab("Within Cluster Variation")


# Evaluate performance
SWC <- function(clusterLabels, dataPoints){
  require(cluster)
  sil <- silhouette(x = clusterLabels, dist = dist(dataPoints))
  return(mean(sil[,3]))
}
set.seed(0)
Silhouette <- rep(0, 10)
for (k in 2:10){
  km.out <- kmeans(x = dat_scaled, centers = k, nstart = 10)
  Silhouette[k] <- SWC(clusterLabels = km.out$cluster, dataPoints = dat_scaled)
}
plot(2:10, Silhouette[2:10], xlab="k", ylab="Silhouette Width Criterion (SWC)", type = "b")



