library(here)
library(dplyr)
library(car)


engineered_data <- readr::read_csv(here("data", "LC_engineered.csv"))
part_1_data <- readr::read_csv(here("data", "part_1_data.csv"))

str(part_1_data)

################################################################################
############################ PART 1 ANALYSIS ###################################
################################################################################

# Cleaning

part_1_data <- engineered_data %>%

  mutate(
      # Extract year from Expected_Graduation
      Expected_Graduation_Yr = substr(as.character(Expected_Graduation),
        nchar(as.character(Expected_Graduation)) - 3, 
        nchar(as.character(Expected_Graduation))),
    
      # Underclassman Indicator
      Underclassman = if_else(
        Class_Standing %in% c("Freshman", "Sophomore"), 1, 0)) %>%

  # Drop columns
  select(
    -Student_IDs, -Course_Name, -Course_Number, -Check_Out_Time,
    -Check_In_Date, -Check_In_Time, -Major, -Week_of_Month,
    -Session_Length_Category, -Occupancy, -Course_Type,
    -Is_Weekend, -Time_Period, -Class_Standing_Self_Reported,
    -Class_Standing_BGSU, -Credit_Load_Category, -GPA_Category, 
    -Class_Standing, -Month, -Course_Code_by_Thousands, -Expected_Graduation,
    -Degree_Type)


# List of categorical columns
categorical_factors <- c("Gender", "Semester", "Day_of_Week",
"Course_Level", "Underclassman", "Expected_Graduation_Yr")

# Convert categorical columns to factors
part_1_data[categorical_factors] <- lapply(part_1_data[categorical_factors], as.factor)


# show structure of data
str(part_1_data)
dim(part_1_data)




########## MODELING ##########


# Split data into training and testing
partition.2 <- function(data, prop.train){
  selected <- sample(1:nrow(data), round(nrow(data)*prop.train), replace = FALSE) 
  data.train <- data[selected,]
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

#vif(mlr)



############## RIDGE ################