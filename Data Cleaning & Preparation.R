# =============================================================================
# TEB2043: DATA SCIENCE - GROUP 12
# Project: Analyzing Lifestyle and Dietary Factors Influencing Obesity Levels
# Section 4.2: Data Preparation & EDA (Fully Instrumented with Visual Proofs)
# =============================================================================

# =============================================================================
# 1 — LIBRARIES
# =============================================================================
library(dplyr)
library(readr)
library(caret)
library(ggplot2)
library(corrplot)
library(tidyr)

# =============================================================================
# 2 — WORKING DIRECTORY
# =============================================================================
setwd("C:/Users/ammir/Desktop/DataScience UTP/DataScience Project")

# =============================================================================
# 3 — HELPER FUNCTIONS
# =============================================================================
get_mode <- function(v) {
  uniqq <- unique(v[!is.na(v)])
  uniqq[which.max(tabulate(match(v, uniqq)))]
}

# =============================================================================
# 4 — LOAD DATA
# =============================================================================
df <- read_csv("ObesityDataSet_raw_and_data_sinthetic.csv", show_col_types = FALSE)

cat(sprintf("\n[LOAD] Original dataset: %d rows × %d columns\n", nrow(df), ncol(df)))

# =============================================================================
# 5 — EXPLORATORY DATA ANALYSIS (PROVING THE DIRTINESS)
# =============================================================================
cat("\n--- PHASE 1: DATA AUDIT & EDA (EXPOSING DIRTY DATA) ---\n")

# Audit A: Missing Values Breakdown
total_na <- sum(is.na(df))
cat(sprintf("-> Missing Values Check: Found %d total missing values.\n", total_na))

if(total_na > 0) {
  cat("   Breakdown of missing values by column:\n")
  na_cols <- colSums(is.na(df))
  print(na_cols[na_cols > 0])
}

# Audit B: Categorical Skewness (MTRANS)
cat("\n-> Transportation (MTRANS) Distribution (Justifies Feature Engineering):\n")
print(table(df$MTRANS))

# Audit C: Duplicate Rows (Moved up to report BEFORE cleaning)
n_dupes <- nrow(df) - nrow(distinct(df))
cat(sprintf("\n-> Duplicates Check: Found %d exact duplicate rows in the dataset.\n", n_dupes))

# Audit D: Structural Errors & Impossible Values
df_temp <- df %>% mutate(Calculated_BMI = Weight / (Height ^ 2))
dirty_rows <- df_temp %>% filter(Age <= 0 | Height <= 0 | Weight <= 0 | Calculated_BMI >= 80)
cat(sprintf("\n-> Structural Errors Check: Found %d physically impossible records.\n", nrow(dirty_rows)))

# PRINT THE DIRTY DATA FOR YOUR REPORT SCREENSHOTS
if(nrow(dirty_rows) > 0) {
  cat("   [PROOF] Here is a sample of the impossible records requiring removal:\n")
  print(head(dirty_rows %>% select(Gender, Age, Height, Weight, Calculated_BMI), 10))
}

cat("\n--- Generating EDA Visualizations for Report ---\n")

# Plot 1: Target distribution
p1 <- ggplot(df, aes(x=NObeyesdad, fill=NObeyesdad)) +
  geom_bar() +
  theme_minimal() +
  theme(axis.text.x=element_text(angle=45, hjust=1), legend.position="none") +
  labs(title="Distribution of Obesity Levels (Raw Data)")
print(p1)

# Plot 2: Boxplots for ALL Numeric Features
p2 <- df %>%
  select(NObeyesdad, Age, FCVC, NCP, CH2O, FAF, TUE) %>%
  pivot_longer(cols = -NObeyesdad, names_to = "Feature", values_to = "Value") %>%
  ggplot(aes(x = NObeyesdad, y = Value, fill = NObeyesdad)) +
  geom_boxplot(outlier.colour="red", outlier.size=1.5, alpha=0.7) +
  facet_wrap(~Feature, scales="free_y") +
  theme_minimal() +
  theme(axis.text.x=element_blank(), legend.position="bottom") +
  labs(title="Outlier Detection Across All Numeric Features",
       subtitle="Red dots indicate extreme values requiring Winsorizing",
       fill="Obesity Level")
print(p2)

# Plot 3: Structural Error Detection (Height vs Weight)
p3 <- ggplot(df_temp, aes(x=Height, y=Weight)) +
  geom_point(aes(color = Calculated_BMI >= 80 | Height <= 0), alpha=0.6, size=2) +
  scale_color_manual(values=c("FALSE"="#2c3e50", "TRUE"="#e74c3c"), name="Dirty Data?") +
  theme_minimal() +
  labs(title="Detecting Structural Anomalies: Height vs. Weight",
       subtitle="Red points indicate physically impossible records requiring removal")
print(p3)

# Plot 4: BMI Histogram (Showing the absurd extremes)
p4 <- ggplot(df_temp, aes(x=Calculated_BMI)) +
  geom_histogram(bins=60, fill="#3498db", color="black", alpha=0.7) +
  geom_vline(xintercept=80, color="red", linetype="dashed", linewidth=1.2) +
  annotate("text", x=95, y=500, label="Impossible BMI Limit (>80)", color="red", angle=90, vjust=-0.5) +
  theme_minimal() +
  labs(title="Distribution of Calculated BMI (Raw Data)",
       subtitle="Notice the extreme structural errors pushing far past the red threshold line",
       x="Calculated BMI", y="Count")
print(p4)

# =============================================================================
# 6 — REMOVE DUPLICATES & STRUCTURAL ERRORS (WITH PROOF)
# =============================================================================
cat("\n--- PHASE 2: DATA CLEANING (THE FIX) ---\n")

# =============================================================================
# 6 — REMOVE DUPLICATES & STRUCTURAL ERRORS (WITH PROOF)
# =============================================================================
cat("\n--- PHASE 2: DATA CLEANING (THE FIX) ---\n")

# PROOF 1: Duplicates
rows_before_dupes <- nrow(df)

# Find ALL duplicate rows
duplicate_rows <- df[duplicated(df), ]

if(nrow(duplicate_rows) > 0) {
  # Print the exact number of duplicates found
  cat(sprintf("\n   [PROOF] Found exactly %d duplicate rows. Here are ALL of them:\n", nrow(duplicate_rows)))
  
  # Print ALL rows by using 'n = Inf' so R doesn't hide any of them
  print(duplicate_rows %>% select(Gender, Age, Height, Weight, NObeyesdad), n = Inf)
} else {
  cat("\n   [PROOF] No duplicate rows found.\n")
}

# Now we actually remove them
df <- distinct(df)
dupes_removed <- rows_before_dupes - nrow(df)
cat(sprintf("\n[FIX] Duplicates: Started with %d rows. Removed %d duplicates. Remaining: %d rows.\n", 
            rows_before_dupes, dupes_removed, nrow(df)))

# PROOF 2: Structural Errors (Bad Values)
df$BMI <- df$Weight / (df$Height ^ 2)
rows_before_bad <- nrow(df)

df <- df %>% filter(Age > 0 & Height > 0 & Weight > 0 & BMI < 80)

bad_removed <- rows_before_bad - nrow(df)
cat(sprintf("[FIX] Bad Values: Removed %d physically impossible records (e.g., BMI >= 80). Remaining: %d rows.\n", 
            bad_removed, nrow(df)))
# =============================================================================
# 7 — ENCODING & FEATURE ENGINEERING
# =============================================================================
cat("\n--- PHASE 2: ENCODING & FEATURE ENGINEERING ---\n")

obesity_levels <- c(
  'Insufficient_Weight','Normal_Weight','Overweight_Level_I',
  'Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'
)

ordinal_map <- c('no' = 0,'Sometimes' = 1,'Frequently' = 2,'Always' = 3)

# THIS IS WHERE THE DATA CHANGES HAPPEN:
df <- df %>%
  mutate(
    # 1. Target Encoding
    NObeyesdad_enc = as.integer(factor(NObeyesdad, levels = obesity_levels)) - 1,
    
    # 2. Binary Encoding
    Gender_enc = as.integer(as.factor(Gender)) - 1,
    family_history_enc = as.integer(as.factor(family_history_with_overweight)) - 1,
    FAVC_enc = as.integer(as.factor(FAVC)) - 1,
    SMOKE_enc = as.integer(as.factor(SMOKE)) - 1,
    SCC_enc = as.integer(as.factor(SCC)) - 1,
    
    # 3. Ordinal Encoding
    CAEC_enc = ordinal_map[CAEC],
    CALC_enc = ordinal_map[CALC],
    
    # 4. Feature Engineering
    transport_active = ifelse(MTRANS %in% c('Walking','Bike'),1,0),
    transport_passive = ifelse(MTRANS %in% c('Automobile','Motorbike','Public_Transportation'),1,0)
  )

# ==========================================
# PROOF FOR REPORT SECTION 4.2
# ==========================================
cat("\n[PROOF] 4.2 Feature Engineering and Encoding Results:\n")

cat("\n1. Target Encoding (NObeyesdad -> NObeyesdad_enc):\n")
print(head(df %>% select(NObeyesdad, NObeyesdad_enc), 4))

cat("\n2. Binary Encoding (ALL 5 Features - Gender, Family History, FAVC, SMOKE, SCC):\n")
print(head(df %>% select(
  Gender_enc, 
  family_history_enc, 
  FAVC_enc, 
  SMOKE_enc, 
  SCC_enc
), 4))

cat("\n3. Ordinal Encoding (ALL 2 Features - CAEC & CALC):\n")
print(head(df %>% select(CAEC_enc, CALC_enc), 2))

cat("\n4. Feature Engineering (MTRANS -> Active/Passive):\n")
print(head(df %>% select(transport_active,transport_passive), 4))

# =============================================================================
# 8 — FEATURE MATRIX (Excluding Target Leakers like BMI, Weight, Height)
# =============================================================================
feature_cols <- c(
  'Age','FCVC','NCP','CH2O','FAF','TUE',
  'Gender_enc','family_history_enc','FAVC_enc','SMOKE_enc',
  'SCC_enc','CAEC_enc','CALC_enc','transport_active','transport_passive'
)

X <- df %>% select(all_of(feature_cols))
y <- df %>% select(NObeyesdad_enc) %>% pull()

# =============================================================================
# 9 — TRAIN / TEST SPLIT (Stratified)
# =============================================================================
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)

X_train <- X[trainIndex,]
X_test  <- X[-trainIndex,]
y_train <- y[trainIndex]
y_test  <- y[-trainIndex]

cat(sprintf("\n[SPLIT] Step 9: Train rows: %d | Test rows: %d\n", nrow(X_train), nrow(X_test)))

# =============================================================================
# 10 — IMPUTATION (WITH PROOF)
# =============================================================================
cat("\n[TRANSFORM] Step 10: Handling Missing Values...\n")

# Count missing values before
na_before <- sum(is.na(X_train))
cat(sprintf("   -> BEFORE Imputation: Found %d missing values in training data.\n", na_before))

for (col in names(X_train)) {
  if(sum(is.na(X_train[[col]])) > 0){
    if(is.numeric(X_train[[col]])){
      val <- median(X_train[[col]], na.rm=TRUE)
    } else{
      val <- get_mode(X_train[[col]])
    }
    X_train[[col]][is.na(X_train[[col]])] <- val
    X_test[[col]][is.na(X_test[[col]])] <- val
  }
}

# Count missing values after
na_after <- sum(is.na(X_train))
cat(sprintf("   -> AFTER Imputation: %d missing values remain. (Successfully imputed %d values using Median/Mode)\n", 
            na_after, na_before - na_after))

# =============================================================================
# 11 — WINSORIZING OUTLIERS (WITH VISUAL PROOF)
# =============================================================================
num_cols <- c('Age','FCVC','NCP','CH2O','FAF','TUE')
cat("\n[TRANSFORM] Step 11: Winsorizing Outliers...\n")

# SAVE 'BEFORE' DATA FOR THE PLOT
df_before_winsor <- X_train %>% select(Age) %>% mutate(State = "1. BEFORE Winsorizing")

total_outliers_fixed <- 0

for (col in num_cols) {
  Q1 <- quantile(X_train[[col]], 0.25, na.rm=TRUE)
  Q3 <- quantile(X_train[[col]], 0.75, na.rm=TRUE)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  
  outliers_capped <- sum(X_train[[col]] < lower | X_train[[col]] > upper, na.rm=TRUE)
  total_outliers_fixed <- total_outliers_fixed + outliers_capped
  
  X_train[[col]] <- pmin(pmax(X_train[[col]], lower), upper)
  X_test[[col]]  <- pmin(pmax(X_test[[col]], lower), upper)
}

cat(sprintf("[FIX] Successfully capped a total of %d outlier data points across all numeric columns.\n", total_outliers_fixed))

# SAVE 'AFTER' DATA & PLOT IT
df_after_winsor <- X_train %>% select(Age) %>% mutate(State = "2. AFTER Winsorizing")
plot_data_winsor <- bind_rows(df_before_winsor, df_after_winsor)

p_outliers <- ggplot(plot_data_winsor, aes(x = State, y = Age, fill=State)) +
  geom_boxplot(outlier.colour="red", outlier.size=2, alpha=0.7) +
  theme_minimal() +
  labs(title="Proof of Outlier Handling (Feature: Age)",
       subtitle="Notice how the red extreme dots are safely capped into the whiskers in the 'After' plot",
       y="Age", x="") +
  theme(legend.position="none")
print(p_outliers)

# =============================================================================
# 12 — FEATURE SCALING (WITH VISUAL PROOF)
# =============================================================================
cat("\n[TRANSFORM] Step 12: Standardization...\n")

# SAVE 'BEFORE' DATA FOR THE PLOT
df_before_scale <- X_train %>% select(Age) %>% mutate(State = "1. BEFORE Scaling (Original Range)")

scale_cols <- c('Age','FCVC','NCP','CH2O','FAF','TUE')
scaler <- preProcess(X_train[,scale_cols], method=c("center","scale"))

X_train[,scale_cols] <- predict(scaler, X_train[,scale_cols])
X_test[,scale_cols]  <- predict(scaler, X_test[,scale_cols])

# SAVE 'AFTER' DATA & PLOT IT
df_after_scale <- X_train %>% select(Age) %>% mutate(State = "2. AFTER Standardization (Mean=0, SD=1)")
plot_data_scale <- bind_rows(df_before_scale, df_after_scale)

p_scale <- ggplot(plot_data_scale, aes(x = Age, fill=State)) +
  geom_density(alpha=0.6) +
  facet_wrap(~State, scales="free") +
  theme_minimal() +
  labs(title="Proof of Standardization (Feature: Age)",
       subtitle="The shape of the data remains identical, but the X-axis is shifted to Mean = 0",
       x="Value", y="Density") +
  theme(legend.position="none")
print(p_scale)

cat("[FIX] StandardScaler applied successfully. Numeric features are now centered at 0.\n")
cat("\n[PROOF] 4.5 Feature Scaling Results (Standardized to Mean=0):\n")
# Printing a sample of all 8 features to prove they are all now small decimals

# PRINT FIRST FEW ROWS OF SCALED FEATURES
cat("\n[OUTPUT] Sample of standardized features:\n")
print(head(X_train[, scale_cols]))
# =============================================================================
# 13 — FEATURE SELECTION
# =============================================================================
cat("\n--- PHASE 3: FEATURE SELECTION ---\n")
corr_matrix_viz <- cor(X_train, use="complete.obs")
# corrplot(corr_matrix_viz, method="color", type="upper", tl.col="black", tl.srt=45) # Uncomment to plot

corr_matrix <- cor(X_train)
high_corr <- findCorrelation(corr_matrix, cutoff=0.85, names=TRUE)

if(length(high_corr) > 0){
  cat(sprintf("[SELECT] Dropped highly correlated features (>0.85): %s\n", paste(high_corr, collapse=", ")))
  X_train <- X_train[, !(names(X_train) %in% high_corr)]
  X_test  <- X_test[, !(names(X_test) %in% high_corr)]
} else {
  cat("[SELECT] No highly correlated features found to drop.\n")
}

p_values <- sapply(names(X_train), function(col){
  fit <- aov(X_train[[col]] ~ as.factor(y_train))
  summary(fit)[[1]][["Pr(>F)"]][1]
})

remove_cols <- names(p_values[p_values > 0.05])

if(length(remove_cols) > 0){
  cat(sprintf("[SELECT] Dropped statistically insignificant features (p > 0.05): %s\n", paste(remove_cols, collapse=", ")))
  X_train <- X_train[, !(names(X_train) %in% remove_cols)]
  X_test  <- X_test[, !(names(X_test) %in% remove_cols)]
} else {
  cat("[SELECT] All remaining features are statistically significant (ANOVA F-Test passed).\n")
}



# =============================================================================
# 14 — TARGET CLASS IMBALANCE AUDIT
# =============================================================================
class_dist <- table(y_train)
imbalance_ratio <- max(class_dist) / min(class_dist)
cat(sprintf("\n[AUDIT] Target Imbalance Ratio: %.2f\n", imbalance_ratio))

if(imbalance_ratio > 1.5){
  cat("[AUDIT] Class imbalance detected. Applying upSample...\n")
  balanced_data <- upSample(x = X_train, y = as.factor(y_train), yname = "NObeyesdad_enc")
  X_train <- balanced_data[,!(names(balanced_data) %in% "NObeyesdad_enc")]
  y_train <- as.integer(as.character(balanced_data$NObeyesdad_enc))
} else {
  cat("[AUDIT] Target variable is well-balanced. No target resampling required.\n")
}

# =============================================================================
# 15 — COMPILE FINAL DATAFRAMES
# =============================================================================
train_df <- cbind(X_train, NObeyesdad_enc = y_train)
test_df  <- cbind(X_test,  NObeyesdad_enc = y_test)

# =============================================================================
# 16 — FINAL FEATURE IMBALANCE REPORT
# =============================================================================
cat("\n====================================================\n")
cat("             FINAL FEATURE IMBALANCE REPORT                 \n")
cat("====================================================\n\n")

imbalance_threshold <- 90

for (col_name in names(train_df)) {
  unique_vals <- length(unique(train_df[[col_name]]))
  
  if (unique_vals < 10) {
    props <- prop.table(table(train_df[[col_name]])) * 100
    max_prop <- max(props)
    
    cat(sprintf("Feature: %s\n", col_name))
    print(round(props, 2)) 
    
    if (max_prop >= imbalance_threshold) {
      cat(sprintf("⚠️ WARNING: Highly imbalanced! Dominant category is %.2f%%\n", max_prop))
    }
    cat("----------------------------------------------------\n")
  }
}

# =============================================================================
# 17 — SAVE TRAIN / TEST DATA
# =============================================================================
setwd("C:/Users/ammir/Desktop/DataScience UTP/DataScience Project/models")

write_csv(train_df, "train.csv")
write_csv(test_df, "test.csv")

cat(sprintf("\n✅ Pipeline Complete. Saved Train (%d cols) and Test (%d cols) successfully.\n", ncol(train_df), ncol(test_df)))