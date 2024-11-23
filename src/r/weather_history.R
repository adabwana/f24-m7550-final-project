library(here)
library(readr)
library(dplyr)
library(lubridate)
library(openmeteo)
library(lunar)

# Read the data
# here() starting path is root of the project
data_training <- readr::read_csv(here("data", "LC_train.csv"))
data_testing <- readr::read_csv(here("data", "LC_test.csv"))

data_full <- bind_rows(data_training, data_testing)

# -----------------------------------------------------------------------------
# MOON PHASES
# -----------------------------------------------------------------------------
lc_moon <- data_training %>%
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
    # Moon phases using radian measures (0 to 2π)
    Moon_Phase = lunar::lunar.phase(Check_In_Date),
    Moon_4Phases = case_when(
      Moon_Phase <= pi/4 | Moon_Phase > 7*pi/4 ~ "New",
      Moon_Phase > pi/4 & Moon_Phase <= 3*pi/4 ~ "Waxing",
      Moon_Phase > 3*pi/4 & Moon_Phase <= 5*pi/4 ~ "Full",
      Moon_Phase > 5*pi/4 & Moon_Phase <= 7*pi/4 ~ "Waning"
    ),
    Moon_8Phases = case_when(
      Moon_Phase <= pi/8 | Moon_Phase > 15*pi/8 ~ "New",
      Moon_Phase > pi/8 & Moon_Phase <= 3*pi/8 ~ "Waxing crescent",
      Moon_Phase > 3*pi/8 & Moon_Phase <= 5*pi/8 ~ "First quarter",
      Moon_Phase > 5*pi/8 & Moon_Phase <= 7*pi/8 ~ "Waxing gibbous",
      Moon_Phase > 7*pi/8 & Moon_Phase <= 9*pi/8 ~ "Full",
      Moon_Phase > 9*pi/8 & Moon_Phase <= 11*pi/8 ~ "Waning gibbous",
      Moon_Phase > 11*pi/8 & Moon_Phase <= 13*pi/8 ~ "Last quarter",
      Moon_Phase > 13*pi/8 & Moon_Phase <= 15*pi/8 ~ "Waning crescent"
    ),
  )

View(lc_moon)

# -----------------------------------------------------------------------------
# WEATHER HISTORY
# -----------------------------------------------------------------------------
# Get the start and end dates
start_date <- min(mdy(data_full$Check_In_Date))
end_date <- max(mdy(data_full$Check_In_Date))

# Bowling Green coordinates
bowling_green_coords <- c(
  latitude = 41.374775,
  longitude = -83.651321
)

# OpenMeteo metrics, with higher spreads
response_units <- list(
  temperature_unit = "fahrenheit",
  windspeed_unit = "kmh", #mph
  precipitation_unit = "mm"
)

# View the hourly weather variables available
weather_variables() %>%
  .[["hourly_history_vars"]]

# Select the hourly weather variables to get
hourly_vars <- c("cloudcover", "temperature_2m", "windspeed_10m",
  "precipitation", "rain", "snowfall")

# Get the weather history
(weather_history <- weather_history(bowling_green_coords,
  start = start_date,
  end = end_date,
  response_units = response_units,
  hourly = hourly_vars
))

# View(weather_history)

# Save the weather history
# readr::write_csv(weather_history, here("data", "weather_history.csv"))
