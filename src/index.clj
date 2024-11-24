^:kindly/hide-code
(ns index
  (:require
   [scicloj.kindly.v4.api :as kindly]
   [scicloj.kindly.v4.kind :as kind])
  (:import [java.time LocalDate]
           [java.time.format DateTimeFormatter]))

^:kindly/hide-code
(def md (comp kindly/hide-code kind/md))

(let [formatter (DateTimeFormatter/ofPattern "M/d/yy")
      current-date (str (.format (LocalDate/now) formatter))]
  (md (str "

### Emma Naiyue Liang, Ryan Rankin, Jaryt Salvo, & Jason Turk
**Date:** **" current-date "**

**Fall 2024 | Math 7550 Statistical Learning I**

*************

This project focuses on analyzing and predicting student visit patterns at the BGSU Learning Commons (LC) center using various statistical learning methods. The project addresses two main prediction tasks using data from Fall 2016 - Spring 2017 academic years.

### Part A: Predicting Visit Duration (50 Points)

This section focuses on predicting the 'Duration_In_Min' variable using various statistical learning techniques. The main steps include:

1. **Data Preprocessing:** 
   - Loading and cleaning the LC visit dataset
   - Handling class standing data adjustments
   - Feature engineering and transformations
   
2. **Model Implementation:** Implementing various prediction models:
   - Linear Regression (Simple and Multiple)
   - Polynomial Regression
   - Ridge Regression (L2 regularization)
   - Lasso Regression (L1 regularization)
   - K-fold Cross Validation for Model Selection

3. **Model Evaluation:**
   - Cross-validation for parameter tuning
   - Comparison of model performances
   - Selection of best performing model

### Part B: Predicting LC Occupancy (50 Points)

This section focuses on predicting the number of students present at check-in moments (occupancy). The main steps include:

1. **Data Processing:**
   - Computing occupancy counts for training data
   - Feature engineering for temporal patterns
   - Data preprocessing and scaling

2. **Model Implementation:**
   - Poisson Regression
   - Linear Regression
   - Ridge Regression (L2 regularization)
   - Lasso Regression (L1 regularization)
   - K-fold Cross Validation for Model Selection

3. **Model Selection:**
   - Cross-validation for model tuning
   - Integer-value prediction adjustments
   - Final model selection and justification

### Technologies and Libraries Used:

- R as the primary programming language
- caret for model training and evaluation
- lubridate for temporal data processing
- ggplot2 for data visualization
- tidyverse for data manipulation

This project demonstrates the application of various statistical learning techniques to real-world time-series prediction problems. It showcases both regression and count-based prediction tasks, with careful consideration of data preprocessing, model selection, and evaluation metrics.

The implementation is contained in separate R scripts for each prediction task, with comprehensive documentation of the modeling process and results analysis.")))
