# Load required libraries
library(here)
library(readr)
library(lubridate)
library(dplyr)

# -----------------------------------------------------------------------------
# READ DATA
# -----------------------------------------------------------------------------
# here() starting path is root of the project
data_raw <- readr::read_csv(here("data", "LC_train.csv"))

LC_train_adj <- data_raw
LCtrain_adj <- LCtrain_adj[LC_train_adj$Duration_In_Min > 0, ]
LCtrain_adj <- LCtrain_adj[LC_train_adj$Duration_In_Min <= 300, ]

# -----------------------------------------------------------------------------
# ENGINEER FEATURES
# -----------------------------------------------------------------------------
lc_engineered <- LC_train_adj %>%
  # Convert dates and times to appropriate formats
  mutate(
    Check_In_Date = mdy(Check_In_Date),
    Check_In_Time = hms::as_hms(Check_In_Time),
    Check_Out_Time = hms::as_hms(Check_Out_Time)
  ) %>%
  # Sort in ascending order
  arrange(Check_In_Date, Check_In_Time) %>%
  # Group by each date
  group_by(Check_In_Date) %>%
  mutate(
    # Existing features
    Cum_Arrivals = row_number(), # - 1, # MINUS ONE TO START AT 0 OCCUPANCY AS 1st PERSON ARRIVES
    Cum_Departures = sapply(seq_along(Check_In_Time), function(i) {
      sum(!is.na(Check_Out_Time[1:i]) & 
            Check_Out_Time[1:i] <= Check_In_Time[i])
    }),
    Occupancy = Cum_Arrivals - Cum_Departures,
    
    # New temporal features
    Day_of_Week = wday(Check_In_Date, label = TRUE),
    Is_Weekend = if_else(Day_of_Week %in% c("Sat", "Sun"), TRUE, FALSE),
    Week_of_Month = ceiling(day(Check_In_Date) / 7),
    Month = month(Check_In_Date, label = TRUE),
    Hour_of_Day = hour(Check_In_Time),
    Time_Period = case_when(
      Hour_of_Day < 6 ~ "Late Night",
      Hour_of_Day < 12 ~ "Morning",
      Hour_of_Day < 17 ~ "Afternoon",
      Hour_of_Day < 22 ~ "Evening",
      TRUE ~ "Late Night"
    ),
    
    # Course-related features
    # Course_Code_by_Thousands = as.factor(Course_Code_by_Thousands),
    Course_Level = case_when(
      Course_Code_by_Thousands < "1500" ~ "Introductory",
      Course_Code_by_Thousands < "2500" ~ "Intermediate",
      Course_Code_by_Thousands < "4500" ~ "Advanced",
      Course_Code_by_Thousands >= "5000" ~ "Graduate",
      TRUE ~ "Other"
    ),
    
    # Student performance indicators
    # Is_Good_Standing = Cumulative_GPA >= 2.0,
    GPA_Category = case_when(
      Cumulative_GPA >= 3.5 ~ "Excellent",
      Cumulative_GPA >= 3.0 ~ "Good",
      Cumulative_GPA >= 2.0 ~ "Satisfactory",
      TRUE ~ "Needs Improvement"
    ),
    
    # Credit load features
    Credit_Load_Category = case_when(
      Term_Credit_Hours <= 6 ~ "Part Time",
      Term_Credit_Hours <= 12 ~ "Half Time",
      Term_Credit_Hours <= 18 ~ "Full Time",
      TRUE ~ "Overload"
    ),
    
    # Renaming column and values for Class_Standing
    Class_Standing_Self_Reported = case_when(
      Class_Standing == "Freshman" ~ "First Year",
      Class_Standing == "Sophomore" ~ "Second Year",
      Class_Standing == "Junior" ~ "Third Year",
      Class_Standing == "Senior" ~ "Fourth Year",
      TRUE ~ Class_Standing
    ),
    
    # Class_standing by BGSU's definition
    # https://www.bgsu.edu/academic-advising/student-resources/academic-standing.html
    Class_Standing_BGSU = case_when(
      Total_Credit_Hours_Earned < 30 ~ "Freshman",
      Total_Credit_Hours_Earned < 60 ~ "Sophomore",
      Total_Credit_Hours_Earned < 90 ~ "Junior",
      Total_Credit_Hours_Earned <= 120 ~ "Senior",
      TRUE ~ "Extended"
    ),
  ) %>%
  ungroup() %>%
  # Remove intermediate columns
  select(-c(Cum_Arrivals, Cum_Departures)) # Check_Out_Time, Class_Standing

# -----------------------------------------------------------------------------
# SAVE ENGINEERED DATA
# -----------------------------------------------------------------------------
readr::write_csv(lc_engineered, here("data", "LC_engineered.csv"))

# -----------------------------------------------------------------------------
# VIEW ENGINEERED DATA
# -----------------------------------------------------------------------------

#View(lc_engineered)

# -----------------------------------------------------------------------------
# READ TEST DATA
# -----------------------------------------------------------------------------
# here() starting path is root of the project
data_raw2 <- readr::read_csv(here("data", "LC_test.csv"))

# -----------------------------------------------------------------------------
# ENGINEER FEATURES
# -----------------------------------------------------------------------------
lc_engineered2 <- data_raw2 %>%
  # Convert dates and times to appropriate formats
  mutate(
    Check_In_Date = mdy(Check_In_Date),
    Check_In_Time = hms::as_hms(Check_In_Time),
  ) %>%
  # Sort in ascending order
  arrange(Check_In_Date, Check_In_Time) %>%
  # Group by each date
  group_by(Check_In_Date) %>%
  mutate(
    
    # New temporal features
    Day_of_Week = wday(Check_In_Date, label = TRUE),
    Is_Weekend = if_else(Day_of_Week %in% c("Sat", "Sun"), TRUE, FALSE),
    Week_of_Month = ceiling(day(Check_In_Date) / 7),
    Month = month(Check_In_Date, label = TRUE),
    Hour_of_Day = hour(Check_In_Time),
    Time_Period = case_when(
      Hour_of_Day < 6 ~ "Late Night",
      Hour_of_Day < 12 ~ "Morning",
      Hour_of_Day < 17 ~ "Afternoon",
      Hour_of_Day < 22 ~ "Evening",
      TRUE ~ "Late Night"
    ),
    
    # Course-related features
    # Course_Code_by_Thousands = as.factor(Course_Code_by_Thousands),
    Course_Level = case_when(
      Course_Code_by_Thousands < "1500" ~ "Introductory",
      Course_Code_by_Thousands < "2500" ~ "Intermediate",
      Course_Code_by_Thousands < "4500" ~ "Advanced",
      Course_Code_by_Thousands >= "5000" ~ "Graduate",
      TRUE ~ "Other"
    ),
    
    # Student performance indicators
    # Is_Good_Standing = Cumulative_GPA >= 2.0,
    GPA_Category = case_when(
      Cumulative_GPA >= 3.5 ~ "Excellent",
      Cumulative_GPA >= 3.0 ~ "Good",
      Cumulative_GPA >= 2.0 ~ "Satisfactory",
      TRUE ~ "Needs Improvement"
    ),
    
    # Credit load features
    Credit_Load_Category = case_when(
      Term_Credit_Hours <= 6 ~ "Part Time",
      Term_Credit_Hours <= 12 ~ "Half Time",
      Term_Credit_Hours <= 18 ~ "Full Time",
      TRUE ~ "Overload"
    ),
    
    # Renaming column and values for Class_Standing
    Class_Standing_Self_Reported = case_when(
      Class_Standing == "Freshman" ~ "First Year",
      Class_Standing == "Sophomore" ~ "Second Year",
      Class_Standing == "Junior" ~ "Third Year",
      Class_Standing == "Senior" ~ "Fourth Year",
      TRUE ~ Class_Standing
    ),
    
    # Class_standing by BGSU's definition
    # https://www.bgsu.edu/academic-advising/student-resources/academic-standing.html
    Class_Standing_BGSU = case_when(
      Total_Credit_Hours_Earned < 30 ~ "Freshman",
      Total_Credit_Hours_Earned < 60 ~ "Sophomore",
      Total_Credit_Hours_Earned < 90 ~ "Junior",
      Total_Credit_Hours_Earned <= 120 ~ "Senior",
      TRUE ~ "Extended"
    ),
  ) %>%
  ungroup()

# -----------------------------------------------------------------------------
# SAVE ENGINEERED DATA
# -----------------------------------------------------------------------------
readr::write_csv(lc_engineered2, here("data", "LC_test_engineered.csv"))

# -----------------------------------------------------------------------------
# VIEW ENGINEERED DATA
# -----------------------------------------------------------------------------

#View(lc_engineered2)




# -----------------------------------------------------------------------------
# DATA CLEANING FOR PART 1
# -----------------------------------------------------------------------------

part_1_data <- lc_engineered %>%
  filter(Duration_In_Min > 0) %>%
  
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
    -Student_IDs, -Course_Name, -Course_Number, -Check_In_Date,
    -Check_In_Time, -Check_Out_Time, -Week_of_Month, -Month,
    -Occupancy, -Cumulative_GPA, -Course_Code_by_Thousands, 
    -Time_Period, -Class_Standing_Self_Reported, -Is_Weekend,
    -Credit_Load_Category, -GPA_Category, -Class_Standing,
    -Expected_Graduation, -Degree_Type, -Class_Standing_BGSU)


# List of categorical columns
categorical_factors <- c("Gender", "Semester", "Day_of_Week",
                         "Course_Level", "Course_Type",
                         "Underclassman", "Expected_Graduation_Yr")

# Convert categorical columns to factors
part_1_data[categorical_factors] <- lapply(part_1_data[categorical_factors], as.factor)

#View(part_1_data)




part_2_data <- lc_engineered2 %>%
  
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
    -Student_IDs, -Course_Name, -Course_Number, -Check_In_Date,
    -Check_In_Time, -Week_of_Month, -Cumulative_GPA, -Course_Code_by_Thousands, 
    -Time_Period, -Class_Standing_Self_Reported, -Is_Weekend,
    -Credit_Load_Category, -GPA_Category, -Class_Standing, -Month,
    -Expected_Graduation, -Degree_Type, -Class_Standing_BGSU)


# List of categorical columns
categorical_factors <- c("Gender", "Semester", "Day_of_Week",
                         "Course_Level", "Course_Type", 
                         "Underclassman", "Expected_Graduation_Yr")

# Convert categorical columns to factors
part_2_data[categorical_factors] <- lapply(part_2_data[categorical_factors], as.factor)





#Additional adjustments
traindata <- part_1_data

Major_Indicated <- ifelse(traindata$Major %in% c("No Response"), 0, 1)
Major_Indicated <- as.factor(Major_Indicated)
traindata <- cbind(traindata, Major_Indicated)
traindata <- traindata %>% select(-Major)

MATH <- ifelse(traindata$Course_Type %in% c("MATH"), 1, 0)
MATH <- as.factor(MATH)
STAT <- ifelse(traindata$Course_Type %in% c("STAT"), 1, 0)
STAT <- as.factor(STAT)
traindata <- cbind(traindata, MATH, STAT)
traindata <- traindata %>% select(-Course_Type)

Exams_Approaching <- ifelse(traindata$Semester_Week %in% c(6,7,8,14,15,16,17), 1, 0)
Exams_Approaching <- as.factor(Exams_Approaching)
traindata <- cbind(traindata, Exams_Approaching)
traindata <- traindata %>% select(-Semester_Week)

Afternoon <- ifelse(traindata$Hour_of_Day %in% c(12,13,14,15,16), 1, 0)
Afternoon <- as.factor(Afternoon)
Evening <- ifelse(traindata$Hour_of_Day %in% c(17,18,19,20,21), 1, 0)
Evening <- as.factor(Evening)
traindata <- cbind(traindata, Afternoon, Evening)
traindata <- traindata %>% select(-Hour_of_Day)

traindata$Day_of_Week <- factor(traindata$Day_of_Week, ordered = FALSE)

  
# show structure of data
#View(traindata)
str(traindata)
dim(traindata)

readr::write_csv(traindata, here("data", "traindata.csv"))



testdata <- part_2_data

Major_Indicated <- ifelse(testdata$Major %in% c("No Response"), 0, 1)
Major_Indicated <- as.factor(Major_Indicated)
testdata <- cbind(testdata, Major_Indicated)
testdata <- testdata %>% select(-Major)

MATH <- ifelse(testdata$Course_Type %in% c("MATH"), 1, 0)
MATH <- as.factor(MATH)
STAT <- ifelse(testdata$Course_Type %in% c("STAT"), 1, 0)
STAT <- as.factor(STAT)
testdata <- cbind(testdata, MATH, STAT)
testdata <- testdata %>% select(-Course_Type)

Exams_Approaching <- ifelse(testdata$Semester_Week %in% c(6,7,8,14,15,16,17), 1, 0)
Exams_Approaching <- as.factor(Exams_Approaching)
testdata <- cbind(testdata, Exams_Approaching)
testdata <- testdata %>% select(-Semester_Week)

Afternoon <- ifelse(testdata$Hour_of_Day %in% c(12,13,14,15,16), 1, 0)
Afternoon <- as.factor(Afternoon)
Evening <- ifelse(testdata$Hour_of_Day %in% c(17,18,19,20,21), 1, 0)
Evening <- as.factor(Evening)
testdata <- cbind(testdata, Afternoon, Evening)
testdata <- testdata %>% select(-Hour_of_Day)

testdata$Day_of_Week <- factor(testdata$Day_of_Week, ordered = FALSE)


# show structure of data
#View(testdata)
str(testdata)
dim(testdata)

readr::write_csv(testdata, here("data", "testdata.csv"))
