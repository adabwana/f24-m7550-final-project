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

# -----------------------------------------------------------------------------
# ENGINEER FEATURES
# -----------------------------------------------------------------------------
lc_engineered <- data_raw %>%
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
      Course_Code_by_Thousands == "1000" ~ "Introductory",
      Course_Code_by_Thousands == "2000" ~ "Intermediate",
      Course_Code_by_Thousands >= "3000" ~ "Advanced",
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

    # Study session features
    # TODO: NO `CHECK_OUT_TIME` IN TEST SET ... HASH OUT LATER
    Duration_In_Min = abs(difftime(Check_Out_Time, Check_In_Time, units = "mins")),
    Session_Length_Category = case_when(
      Duration_In_Min <= 30 ~ "Short",
      Duration_In_Min <= 90 ~ "Medium",
      Duration_In_Min <= 180 ~ "Long",
      TRUE ~ "Extended"
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
View(lc_engineered)




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

  # Convert Check_In_Time to datetime and round to the nearest minute
  mutate(
    Check_In_Timestamp = as_datetime(paste(Check_In_Date, Check_In_Time)),
    Check_In_Timestamp = floor_date(Check_In_Timestamp, unit = "minute")
  ) %>%

  # Count the number of students for each rounded Check_In_Timestamp
  add_count(Check_In_Timestamp, name = "Group_Size") %>%

  # Create binary Group_Check_In feature
  mutate(Group_Check_In = if_else(Group_Size > 1, 1, 0)) %>%

      add_count(Student_IDs, name = "Total_Visits")  %>%

  # Drop columns
  select(
    -Student_IDs, -Course_Name, -Course_Number, -Check_Out_Time,
    -Check_In_Date, -Check_In_Time, -Major, -Week_of_Month,
    -Session_Length_Category, -Occupancy, -Course_Type,
    -Is_Weekend, -Time_Period, -Class_Standing_Self_Reported,
    -Class_Standing_BGSU, -Credit_Load_Category, -GPA_Category, 
    -Class_Standing, -Month, -Course_Code_by_Thousands, -Expected_Graduation,
    -Degree_Type, -Check_In_Timestamp)


# List of categorical columns
categorical_factors <- c("Gender", "Semester", "Day_of_Week",
"Course_Level", "Underclassman", "Expected_Graduation_Yr", "Group_Check_In")

# Convert categorical columns to factors
part_1_data[categorical_factors] <- lapply(part_1_data[categorical_factors], as.factor)

# take absolute value of Duration_In_Min
#part_1_data$Duration_In_Min <- abs(part_1_data$Duration_In_Min)
View(part_1_data)

# show structure of data
str(part_1_data)
dim(part_1_data)

readr::write_csv(part_1_data, here("data", "part_1_data.csv"))

summary(part_1_data$Group_Check_In)




# -----------------------------------------------------------------------------
# DATA CLEANING FOR PART 2
# -----------------------------------------------------------------------------

part_2_data <- lc_engineered %>%
  filter(Duration_In_Min > 0) %>%

  mutate(
      # Extract year from Expected_Graduation
      Expected_Graduation_Yr = substr(as.character(Expected_Graduation),
        nchar(as.character(Expected_Graduation)) - 3, 
        nchar(as.character(Expected_Graduation))),
    
      # Underclassman Indicator
      Underclassman = if_else(
        Class_Standing %in% c("Freshman", "Sophomore"), 1, 0)) %>%

      add_count(Student_IDs, name = "Total_Visits")  %>%

  # Drop columns
  select(
    -Student_IDs, -Course_Name, -Course_Number, -Check_Out_Time,
    -Check_In_Date, -Check_In_Time, -Major, -Week_of_Month,
    -Session_Length_Category, -Course_Type,
    -Is_Weekend, -Time_Period, -Class_Standing_Self_Reported,
    -Class_Standing_BGSU, -Credit_Load_Category, -GPA_Category, 
    -Class_Standing, -Month, -Course_Code_by_Thousands, -Expected_Graduation,
    -Degree_Type,
    -Duration_In_Min)


# List of categorical columns
categorical_factors <- c("Gender", "Semester", "Day_of_Week",
"Course_Level", "Underclassman", "Expected_Graduation_Yr")

# Convert categorical columns to factors
part_2_data[categorical_factors] <- lapply(part_2_data[categorical_factors], as.factor)

# take absolute value of Duration_In_Min
#part_1_data$Duration_In_Min <- abs(part_1_data$Duration_In_Min)
View(part_2_data)

# show structure of data
str(part_2_data)
dim(part_2_data)

readr::write_csv(part_2_data, here("data", "part_2_data.csv"))





