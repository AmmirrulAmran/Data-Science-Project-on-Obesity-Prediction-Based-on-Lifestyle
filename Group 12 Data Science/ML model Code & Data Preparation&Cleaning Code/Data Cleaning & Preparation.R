# =============================================================================
# DATA PREPARATION & EDA PIPELINE
# =============================================================================

library(dplyr)
library(readr)
library(caret)
library(ggplot2)
library(tidyr)

setwd("C:/Users/ammir/Desktop/DataScience UTP/DataScience Project")

# Helper function for mode imputation
get_mode <- function(v) {
  uniqq <- unique(v[!is.na(v)])
  uniqq[which.max(tabulate(match(v, uniqq)))]
}

df <- read_csv("ObesityDataSet_raw_and_data_sinthetic.csv", show_col_types = FALSE)
cat(sprintf("\n[LOAD] Original dataset: %d rows × %d columns\n", nrow(df), ncol(df)))

# =============================================================================
# PHASE 1: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
cat("\n--- PHASE 1: DATA AUDIT & EDA ---\n")

# Missing Values & Duplicates Check
cat(sprintf("-> Missing Values: %d\n", sum(is.na(df))))
cat(sprintf("-> Duplicates: %d exact duplicate rows found.\n", nrow(df) - nrow(distinct(df))))

# Structural Errors Check
df_temp <- df %>% mutate(Calculated_BMI = Weight / (Height ^ 2))
dirty_rows <- df_temp %>% filter(Age <= 0 | Height <= 0 | Weight <= 0 | Calculated_BMI >= 80)
cat(sprintf("-> Structural Errors: %d physically impossible records found.\n", nrow(dirty_rows)))

# Generate EDA Visualizations
p1 <- ggplot(df, aes(x=NObeyesdad, fill=NObeyesdad)) +
  geom_bar() + theme_minimal() + theme(axis.text.x=element_text(angle=45, hjust=1), legend.position="none") +
  labs(title="Distribution of Obesity Levels (Raw Data)")
print(p1)

p2 <- df %>% select(NObeyesdad, Age, FCVC, NCP, CH2O, FAF, TUE) %>%
  pivot_longer(cols = -NObeyesdad, names_to = "Feature", values_to = "Value") %>%
  ggplot(aes(x = NObeyesdad, y = Value, fill = NObeyesdad)) +
  geom_boxplot(outlier.colour="red", outlier.size=1.5, alpha=0.7) + facet_wrap(~Feature, scales="free_y") +
  theme_minimal() + theme(axis.text.x=element_blank(), legend.position="bottom") +
  labs(title="Outlier Detection Across All Numeric Features")
print(p2)

p3 <- ggplot(df_temp, aes(x=Height, y=Weight)) +
  geom_point(aes(color = Calculated_BMI >= 80 | Height <= 0), alpha=0.6, size=2) +
  scale_color_manual(values=c("FALSE"="#2c3e50", "TRUE"="#e74c3c"), name="Dirty Data?") +
  theme_minimal() + labs(title="Detecting Structural Anomalies: Height vs. Weight")
print(p3)

# =============================================================================
# PHASE 2: DATA CLEANING & ENGINEERING
# =============================================================================
cat("\n--- PHASE 2: DATA CLEANING & ENGINEERING ---\n")

# 1. Clean Duplicates and Impossible Values
df <- distinct(df)
df$BMI <- df$Weight / (df$Height ^ 2)
df <- df %>% filter(Age > 0 & Height > 0 & Weight > 0 & BMI < 80) %>% select(-BMI)

# 2. Feature Encoding & Engineering
obesity_levels <- c('Insufficient_Weight','Normal_Weight','Overweight_Level_I',
                    'Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III')
ordinal_map <- c('no' = 0, 'Sometimes' = 1, 'Frequently' = 2, 'Always' = 3)

df <- df %>%
  mutate(
    NObeyesdad_enc = as.integer(factor(NObeyesdad, levels = obesity_levels)) - 1,
    Gender_enc = as.integer(as.factor(Gender)) - 1,
    family_history_enc = as.integer(as.factor(family_history_with_overweight)) - 1,
    FAVC_enc = as.integer(as.factor(FAVC)) - 1,
    SMOKE_enc = as.integer(as.factor(SMOKE)) - 1,
    SCC_enc = as.integer(as.factor(SCC)) - 1,
    CAEC_enc = ordinal_map[CAEC],
    CALC_enc = ordinal_map[CALC],
    transport_active = ifelse(MTRANS %in% c('Walking','Bike'), 1, 0),
    transport_passive = ifelse(MTRANS %in% c('Automobile','Motorbike','Public_Transportation'), 1, 0)
  )

# 3. Create Feature Matrix (Ensuring Height and Weight are included for scaling)
feature_cols <- c('Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
                  'Gender_enc','family_history_enc','FAVC_enc','SMOKE_enc',
                  'SCC_enc','CAEC_enc','CALC_enc','transport_active','transport_passive')

X <- df %>% select(all_of(feature_cols))
y <- df %>% select(NObeyesdad_enc) %>% pull()

# 4. Stratified Train/Test Split
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex,]
X_test  <- X[-trainIndex,]
y_train <- y[trainIndex]
y_test  <- y[-trainIndex]

# =============================================================================
# PHASE 3: TRANSFORMATION & FEATURE SELECTION
# =============================================================================
cat("\n--- PHASE 3: TRANSFORMATION & SELECTION ---\n")

# 5. Missing Value Imputation
for (col in names(X_train)) {
  if(sum(is.na(X_train[[col]])) > 0){
    val <- ifelse(is.numeric(X_train[[col]]), median(X_train[[col]], na.rm=TRUE), get_mode(X_train[[col]]))
    X_train[[col]][is.na(X_train[[col]])] <- val
    X_test[[col]][is.na(X_test[[col]])] <- val
  }
}

# 6. Winsorizing Outliers (Capping extreme values using IQR)
num_cols <- c('Age','FCVC','NCP','CH2O','FAF','TUE')
for (col in num_cols) {
  Q1 <- quantile(X_train[[col]], 0.25, na.rm=TRUE)
  Q3 <- quantile(X_train[[col]], 0.75, na.rm=TRUE)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  
  X_train[[col]] <- pmin(pmax(X_train[[col]], lower), upper)
  X_test[[col]]  <- pmin(pmax(X_test[[col]], lower), upper)
}

# 7. Feature Scaling (Standardizing all 8 continuous variables to Mean=0)
scale_cols <- c('Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE')
scaler <- preProcess(X_train[,scale_cols], method=c("center","scale"))

X_train[,scale_cols] <- predict(scaler, X_train[,scale_cols])
X_test[,scale_cols]  <- predict(scaler, X_test[,scale_cols])

cat("\n[PROOF] 4.5 Feature Scaling Results (Standardized to Mean=0):\n")
print(head(X_train[, scale_cols], 4))

# 8. Feature Selection 
# Check for Highly Correlated Features
corr_matrix <- cor(X_train)
high_corr <- findCorrelation(corr_matrix, cutoff=0.85, names=TRUE)
if(length(high_corr) > 0){
  X_train <- X_train[, !(names(X_train) %in% high_corr)]
  X_test  <- X_test[, !(names(X_test) %in% high_corr)]
  cat(sprintf("\n[SELECT] Dropped redundant correlated feature: %s\n", paste(high_corr, collapse=", ")))
}

# Drop Height and Weight to prevent Target Leakage
X_train <- X_train %>% select(-Height, -Weight)
X_test  <- X_test %>% select(-Height, -Weight)
cat("[SELECT] Dropped Height and Weight to prevent Target Leakage.\n")

# =============================================================================
# PHASE 4: FINAL AUDIT & EXPORT
# =============================================================================
cat("\n--- PHASE 4: FINAL AUDIT & EXPORT ---\n")

# 9. Target Class Imbalance Resolution
imbalance_ratio <- max(table(y_train)) / min(table(y_train))
if(imbalance_ratio > 1.5){
  balanced_data <- upSample(x = X_train, y = as.factor(y_train), yname = "NObeyesdad_enc")
  X_train <- balanced_data[,!(names(balanced_data) %in% "NObeyesdad_enc")]
  y_train <- as.integer(as.character(balanced_data$NObeyesdad_enc))
} else {
  cat(sprintf("[AUDIT] Target variable balanced (Ratio: %.2f). No resampling needed.\n", imbalance_ratio))
}

# 10. Compile and Save Final DataFrames
train_df <- cbind(X_train, NObeyesdad_enc = y_train)
test_df  <- cbind(X_test,  NObeyesdad_enc = y_test)

setwd("C:/Users/ammir/Desktop/DataScience UTP/DataScience Project/models")
write_csv(train_df, "train.csv")
write_csv(test_df, "test.csv")

cat(sprintf("\n✅ Pipeline Complete. Saved Train (%d cols) and Test (%d cols) successfully.\n", ncol(train_df), ncol(test_df)))