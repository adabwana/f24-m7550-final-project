# Feature Engineering

The complete feature engineering implementation can be found in our [source code](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/r/feature_engineering.R).

## Temporal Feature Engineering

Our feature engineering process began with **_temporal data extraction_** using the 'lubridate' package. The timestamp data provided several readily constructible features:

- Day of week
- Weekend indicator (Sunday-specific)
- Check-in month
- Check-in hour

Analysis of visit patterns revealed a **_non-linear relationship_** between check-in hour and visit duration. This observation prompted the creation of a more nuanced 'Time Category' variable with distinct periods:

- Morning (6am - 12pm)
- Afternoon (12pm - 5pm)
- Evening (5pm - 11pm)
- Late Night (11pm - 6am)

The 'Expected Graduation' variable presented a **_dimensionality challenge_** due to its categorical semester format. We addressed this by converting it to a numeric **_'Months Until Graduation'_** metric, effectively reducing complexity while maintaining predictive potential.

## Course-Related Features

The 'Course Code by Thousands' variable exhibited significant ambiguity when treated categorically. Our analysis indicated the need for a more structured approach, leading to the following **_classification systems_**:

1. Course Level Categories:
   - Special (≤ 100)
   - Lower Classmen (≤ 3000)
   - Upper Classmen (≤ 4000)
   - Graduate (> 4000)

2. GPA Categories:
   - Excellent (≥ 3.5)
   - Good (≥ 3.0)
   - Satisfactory (≥ 2.0)
   - Needs Improvement (< 2.0)

3. Credit Load Categories:
   - Part Time (≤ 6)
   - Half Time (≤ 12)
   - Full Time (≤ 18)
   - Overload (> 18)

## Student Classification Features

The dataset exhibited an **_unexpected concentration_** of 'Senior' classifications in our initial analysis. Further investigation revealed this stemmed from students accumulating excess credits for senior status without fulfilling graduation requirements. To address this imbalance while preserving useful information, we implemented a **_dual classification approach_**.

1. Class Standing (Self-Reported):
   - First Year (Freshman)
   - Second Year (Sophomore)
   - Third Year (Junior)
   - Fourth Year (Senior)
   - Graduate
   - Other

2. BGSU Standing (Credit-Based):
   - Freshman (< 30 credits)
   - Sophomore (< 60 credits)
   - Junior (< 90 credits)
   - Senior (≤ 120 credits)
   - Extended (> 120 credits)

The original Class Standing variable, while potentially containing valuable self-reported insights, required recoding. We preserved this information as **_'Class Standing Self Reported'_** with progression labels from "First Year" through "Fourth Year", along with "Graduate" and "Other" designations. Complementing this, we developed a more **_objective BGSU Standing metric_** based on credit hours. This dual approach preserves potentially valuable self-reported information while introducing a more objective credit-based metric.

## Course Name and Type Features

The Course Name variable presented **_immediate challenges_** for model fitting in its raw form. While various approaches existed for handling this **_high-cardinality variable_**, we opted for a **_flexible keyword-based system_**. This approach identifies key terms within course names - for instance, classifying courses containing 'Culture', 'Language', or 'Ethics' under 'Humanities'. Though this resulted in 14 distinct categories, it provides flexibility for subsequent modeling decisions through **_regularization or variable selection_**.

- Introductory
- Intermediate
- Advanced
- Business
- Laboratory
- Mathematics
- Computer Science
- Natural Sciences
- Social Sciences
- Humanities
- Education

Similarly, the Course Type variable required **_substantial level reduction_**. We consolidated the original categories into **_natural academic groupings_** such as business courses, education courses, and STEM courses. For visits lacking course specifications, we designated a "No Response" category rather than discarding these observations.

- **_STEM Core_**
- **_Engineering & Technology_**
- **_Business_**
- **_Social Sciences_**
- **_Health Sciences_**
- **_Humanities_**

For visits without a specified course association, we introduced a "**_No Response_**" category to maintain data completeness.

## Major Categories

The Major variable demanded a similar **_keyword-based reduction strategy_** as Course Name. Through analysis of major descriptions, we identified **_recurring terms_** that allowed for logical grouping. For example, the 'Mathematics' category encompasses mathematics, statistics, and actuarial science majors. Our final categorization includes:

- Mathematics (including statistics and actuarial science)
- Business
- Computing & Technology
- Natural Sciences
- Health Sciences
- Social Sciences
- Education
- Arts & Humanities
- Pre-Professional
- General Studies

We maintained an 'Other' category for majors that defied clear classification. The data structure also revealed an opportunity to identify students pursuing **_multiple degrees_** - we created this indicator by detecting comma-separated entries in the Major field.

## Visit Pattern Features

Student ID analysis enabled the construction of several **_usage metrics_**. Beyond simple visit counts, we examined **_temporal patterns_** at multiple scales:

- **_Total visits per student_**
- **_Visits per semester_**
- **_Average weekly visits_**
- **_Week volume categories_**

Examination of visit frequency throughout the semester revealed **_clear patterns_**. Weeks 1-3, 9, 14, and 17 consistently showed lower activity levels, while the remaining weeks demonstrated higher traffic. This distinction proved valuable, as visit volume may influence individual visit duration. We encoded this insight through a **_binary 'Volume' indicator_** for each week.

## Course Load and Performance Features

For each student-semester combination, we developed metrics to capture **_academic context_**. We tracked the number of unique courses and examined the distribution of course levels based on the 'Course Code by Thousands' variable. Particular attention was paid to **_upper-division coursework_**, creating a specific metric for the proportion of 4000-level courses. Additionally, we implemented a **_GPA trend indicator_** that focuses on directional changes rather than absolute values, recognizing that the direction of GPA movement might be more informative than the magnitude.

- Count of unique courses
- Distribution of course levels
- **_Proportion of upper-division courses_** (4000-level)
- **_GPA trend indicator_** (focusing on directional changes rather than absolute values)

## Data Quality and Group Dynamics

Our preprocessing included **_essential validation steps_**. We verified **_Duration_in_Min calculations_** through comparison of check-in and check-out times, ensuring no negative values existed in the data. The **_Occupancy variable_** received similar scrutiny during its construction.

- Duration verification through check-in/check-out time comparison
- Accurate occupancy calculations
- Identification of group visits through timestamp clustering

A final analytical step involved identifying **_group study patterns_**. By examining clusters of check-in times, we detected multiple students arriving within the same minute - a strong indicator of group visits. This observation led to three complementary features:

- A **_binary group visit indicator_**
- **_Exact group size_**
- **_Size-based categories_** (individual: $= 1$, small: $< 3$, medium: $3-6$, large: $> 6$)

While some simultaneous check-ins might be coincidental, this classification captures potential **_social patterns_** in Learning Commons usage, particularly among friend groups.