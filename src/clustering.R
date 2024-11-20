library(here)
library(readr)
library(lubridate)
library(tidyverse)
library(skimr)  
library(DataExplorer)

library(GGally)
library(dendextend)
library(factoextra) #fviz_pca_ind
library(ggforce)
library(mclust)
library(MixSim)

# Read the engineered data
data <- readr::read_csv(here("data", "LC_engineered.csv"))

# Check the data
skimr::skim(data)
