library(here)
library(readr)
library(dplyr)
library(lubridate)
library(openmeteo)

# Read the data
# here() starting path is root of the project
data_training <- readr::read_csv(here("data", "LC_train.csv"))
data_testing <- readr::read_csv(here("data", "LC_test.csv"))

data_full <- bind_rows(data_training, data_testing)

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
readr::write_csv(weather_history, here("data", "weather_history.csv"))