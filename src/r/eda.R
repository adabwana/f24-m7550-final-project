# Load required libraries
library(here)
library(readr)
library(lubridate)
library(dplyr)
library(skimr)  
library(DataExplorer)

library(fitdistrplus)
library(ggplot2)
library(gridExtra)

theme_set(theme_bw())

# -----------------------------------------------------------------------------
# READ RAW DATA
# -----------------------------------------------------------------------------
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
    Cum_Arrivals = row_number(), # - 1,  MINUS ONE TO START AT 0 OCCUPANCY AS 1st PERSON ARRIVES
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
data_eng <- readr::read_csv(here("data", "LC_engineered.csv"))

View(data_eng)

# Basic overview of the data
glimpse(data_eng)

# Get comprehensive summary statistics
skim(data_eng)
DataExplorer::plot_intro(data_eng) 

# =================================================================================
# PART A: DURATION IN MINUTES DISTRIBUTION ANALYSIS
# =================================================================================

descdist(data_eng$Duration_In_Min, boot = 1000)

fitW <- fitdist(data_eng$Duration_In_Min, "weibull")
fitE <- fitdist(data_eng$Duration_In_Min, "exp", lower = c(0))
fitG <- fitdist(data_eng$Duration_In_Min, "gamma", 
                start = list(scale = 1, shape = 1), lower = c(0, 0))
fitN <- fitdist(data_eng$Duration_In_Min, "norm")
fitLn <- fitdist(data_eng$Duration_In_Min, "lnorm")

dc <- denscomp(list(fitW, fitE, fitG, fitN, fitLn), plotstyle = "ggplot", breaks = 30,
         legendtext = c("Weibull", "Exp", "Gamma", "Normal", "Log Normal")) +
     scale_y_continuous(labels = scales::label_number(scale = 1e3, suffix = "(1/K)", big.mark = ",")) + 
  theme(legend.position = "none")

cc <- cdfcomp(list(fitW, fitE, fitG, fitN, fitLn), plotstyle = "ggplot",
         legendtext = c("Weibull", "Exp", "Gamma", "Normal", "Log Normal"))

qqc <- qqcomp(list(fitW, fitE, fitG, fitN, fitLn), plotstyle = "ggplot",
         legendtext = c("Weibull", "Exp", "Gamma", "Normal", "Log Normal")) + 
  theme(legend.position = "none")

ppc <- ppcomp(list(fitW, fitE, fitG, fitN, fitLn), plotstyle = "ggplot",
         legendtext = c("Weibull", "Exp", "Gamma", "Normal", "Log Normal"))

# Create directories recursively if they don't exist
dir.create(here("presentation", "images", "eda"), recursive = TRUE, showWarnings = FALSE)

# For Duration Analysis
grid_title <- grid::textGrob("Duration Distribution Analysis", gp = grid::gpar(fontsize = 14))
g1 <- gridExtra::grid.arrange(
  dc, cc, qqc, ppc, 
  ncol = 2, nrow = 2, 
  widths = c(1.5, 2),
  top = grid_title
)

# Save Duration Distribution plot
ggsave(here("presentation", "images", "eda", "duration_distribution.jpg"), g1, width = 12, height = 8, dpi = 300)

# =================================================================================
# PART B: OCCUPANCY DISTRIBUTION ANALYSIS
# =================================================================================

descdist(data_eng$Occupancy, boot = 1000)

fitW <- fitdist(data_eng$Occupancy, "weibull")
fitE <- fitdist(data_eng$Occupancy, "exp", lower = c(0))
fitG <- fitdist(data_eng$Occupancy, "gamma", 
                start = list(scale = 1, shape = 1), lower = c(0, 0))
fitN <- fitdist(data_eng$Occupancy, "norm")
fitLn <- fitdist(data_eng$Occupancy, "lnorm")

dc <- denscomp(list(fitW, fitE, fitG, fitN, fitLn), plotstyle = "ggplot", breaks = 21,
         legendtext = c("Weibull", "Exp", "Gamma", "Normal", "Log Normal")) +
     scale_y_continuous(labels = scales::label_number(scale = 1e3, suffix = "(1/K)", big.mark = ",")) + 
  theme(legend.position = "none")

cc <- cdfcomp(list(fitW, fitE, fitG, fitN, fitLn), plotstyle = "ggplot",
         legendtext = c("Weibull", "Exp", "Gamma", "Normal", "Log Normal"))

qqc <- qqcomp(list(fitW, fitE, fitG, fitN, fitLn), plotstyle = "ggplot",
         legendtext = c("Weibull", "Exp", "Gamma", "Normal", "Log Normal")) + 
  theme(legend.position = "none")

ppc <- ppcomp(list(fitW, fitE, fitG, fitN, fitLn), plotstyle = "ggplot",
         legendtext = c("Weibull", "Exp", "Gamma", "Normal", "Log Normal"))

# For Occupancy Analysis
grid_title <- grid::textGrob("Occupancy Distribution Analysis", gp = grid::gpar(fontsize = 14))
g2 <- gridExtra::grid.arrange(
  dc, cc, qqc, ppc, 
  ncol = 2, nrow = 2, 
  widths = c(1.5, 2),
  top = grid_title
)

# Save Occupancy Distribution plot
ggsave(here("presentation", "images", "eda", "occupancy_distribution.jpg"), g2, width = 12, height = 8, dpi = 300)
