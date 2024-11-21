# Load required libraries
library(here)
library(readr)
library(lubridate)
library(dplyr)

# -----------------------------------------------------------------------------
# READ IN DATA
# -----------------------------------------------------------------------------
# here() starting path is root of the project
data_training <- readr::read_csv(here("data", "LC_train.csv"))


# -----------------------------------------------------------------------------
# ENGINEER FEATURES
# -----------------------------------------------------------------------------
lc_engineered <- data_training %>%
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
    Cum_Arrivals = row_number() - 1, 
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
    Duration_In_Min = difftime(Check_Out_Time, Check_In_Time, units = "mins"),
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

View(lc_engineered)

# Save the engineered data
readr::write_csv(lc_engineered, here("data", "LC_engineered.csv"))
