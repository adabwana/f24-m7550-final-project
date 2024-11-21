# Load required libraries
library(here)
library(readr)
library(lubridate)
library(tidyverse)
library(skimr)  
library(DataExplorer)


# -----------------------------------------------------------------------------
# RAW DATA EXPLORATION
# -----------------------------------------------------------------------------

# Read the data
# here() starting path is root of the project
data_raw <- readr::read_csv(here("data", "LC_train.csv"))

lc_data <- data_raw %>%
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
    # Cumulative check-ins
    Cum_Arrivals = row_number() - 1, 
    # Cumulative check-outs
    Cum_Departures = sapply(seq_along(Check_In_Time), function(i) {
      sum(!is.na(Check_Out_Time[1:i]) & 
          Check_Out_Time[1:i] <= Check_In_Time[i])
    }),
    # Current occupancy
    Occupancy = Cum_Arrivals - Cum_Departures,
    # Course_Code_by_Thousands = as.factor(Course_Code_by_Thousands)
  ) %>%
  ungroup() #%>%
  # Remove intermediate columns
  # select(-c(Check_Out_Time, Cum_Arrivals, Cum_Departures))  

# Basic overview of the data
glimpse(lc_data)

# Get comprehensive summary statistics
skim(lc_data)
DataExplorer::plot_intro(lc_data)

# Check for missing values
missing_values <- colSums(is.na(lc_data))
print("Missing values by column:")
print(missing_values[missing_values > 0])

# Basic visualizations
# Plot distribution of numeric columns
DataExplorer::plot_histogram(lc_data)
DataExplorer::plot_bar(lc_data)
DataExplorer::plot_boxplot(lc_data, by = "Class_Standing")

# Correlation analysis of numeric columns
DataExplorer::plot_correlation(lc_data)
DataExplorer::plot_prcomp(lc_data, variance_cap = 0.9, nrow = 2L, ncol = 2L)

# -----------------------------------------------------------------------------
# ENGINEERED DATA EXPLORATION (AFTER RUNNING FEATURE ENGINEERING SCRIPT)
# -----------------------------------------------------------------------------

# Read the engineered data
engineered_data <- readr::read_csv(here("data", "LC_engineered.csv"))

View(engineered_data)

# Basic overview of the data
glimpse(engineered_data)

# Get comprehensive summary statistics
skim(engineered_data)
DataExplorer::plot_intro(engineered_data) 

