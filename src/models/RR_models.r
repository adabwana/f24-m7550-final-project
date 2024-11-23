library(dplyr)
library(car)


engineered_data <- readr::read_csv(here("data", "LC_engineered.csv"))

################################################################################
############################ PART 1 ANALYSIS ###################################
################################################################################

# Cleaning

# Drop specific columns
part_1_data <- engineered_data %>%
  select(-Student_IDs, -Course_Name, -Course_Number, -Check_Out_Time, 
  -Check_In_Date, -Check_In_Time, -Major, -Week_of_Month, -Session_Length_Category, -Occupancy, -Course_Type)  

# List of categorical columns
categorical_factors <- c("Gender", "Is_Weekend", "Semester", "Degree_Type", 
"Class_Standing", "Expected_Graduation", "Day_of_Week", "Month", "Course_Level", 
"Course_Code_by_Thousands", "Time_Period", "GPA_Category", 
"Credit_Load_Category", "Class_Standing_Self_Reported", "Class_Standing_BGSU")

# Convert categorical columns to factors
part_1_data[categorical_factors] <- lapply(part_1_data[categorical_factors], as.factor)


# show structure of data
str(part_1_data)
dim(part_1_data)




# Analysis

# Split data into training and testing
partition.2 <- function(data, prop.train){
  # select a random sample of size = prop.train % of total records
  selected <- sample(1:nrow(data), round(nrow(data)*prop.train), replace = FALSE) 
  # create training data which has prop.train % of total records
  data.train <- data[selected,]
  # create validation data
  rest <- setdiff(1:nrow(data), selected)
  data.test <- data[rest,]
  return(list(data.train=data.train, data.test=data.test))
}

RNGkind (sample.kind = "Rounding") 
set.seed(0)
p2 <- partition.2(data, 0.8) ## 80:20 split
training.data <- p2$data.train
test.data <- p2$data.test



############## MULTIPLE LINEAR REGRESSION ################
mlr <- lm(Duration_In_Min ~ ., data = part_1_data)
summary(mlr)

vif(mlr)



############## RIDGE ################