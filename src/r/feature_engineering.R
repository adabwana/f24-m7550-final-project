# Load required libraries
library(here)
library(readr)
library(dplyr)
library(lubridate)
library(hms)
library(purrr)

# -----------------------------------------------------------------------------
# TEMPORAL FEATURES
# -----------------------------------------------------------------------------
prepare_dates <- function(df) {
  df %>% mutate(
    Check_In_Date = mdy(Check_In_Date),
    Check_In_Time = hms::as_hms(Check_In_Time),
    Check_Out_Time = hms::as_hms(Check_Out_Time)
  )
}

add_temporal_features <- function(df) {
  df %>% mutate(
    Check_In_Day = wday(Check_In_Date, label = TRUE),
    Is_Weekend = Check_In_Day %in% c("Sat", "Sun"),
    Check_In_Week = ceiling(day(Check_In_Date) / 7),
    Check_In_Month = month(Check_In_Date, label = TRUE),
    Check_In_Hour = hour(Check_In_Time)
  )
}

add_time_category <- function(df) {
  df %>% mutate(
    Time_Category = case_when(
      Check_In_Hour < 6 ~ "Late Night",
      Check_In_Hour < 12 ~ "Morning",
      Check_In_Hour < 17 ~ "Afternoon",
      Check_In_Hour < 22 ~ "Evening",
      TRUE ~ "Late Night"
    )
  )
}

convert_semester_to_date <- function(semester_str) {
  # Extract year and semester
  parts <- strsplit(semester_str, " ")[[1]]
  year <- parts[length(parts)]
  semester <- parts[1]
  
  # Map semesters to months
  month <- case_when(
    semester == "Fall" ~ "08",
    semester == "Spring" ~ "01",
    semester == "Summer" ~ "06",
    semester == "Winter" ~ "12",
    TRUE ~ NA_character_
  )
  
  # Combine into date
  paste0(month, "/", "01", "/", year)
}

add_date_features <- function(df) {
  df %>%
    mutate(
      # Convert semester to date
      Semester_Date = mdy(purrr::map(Semester, convert_semester_to_date)),
      # Convert expected graduation to date
      Expected_Graduation_Date = mdy(purrr::map(Expected_Graduation, convert_semester_to_date)),
    )
}

add_graduation_features <- function(df) {
  df %>% mutate(
    # Calculate months until graduation
    Months_Until_Graduation = as.numeric(
      difftime(Expected_Graduation_Date, Semester_Date, units = "days") / 30.44 # average days per month
    )
  )
}

# -----------------------------------------------------------------------------
# ACADEMIC & COURSE FEATURES
# -----------------------------------------------------------------------------
add_course_features <- function(df) {
  df %>% mutate(
    Course_Level = case_when(
      Course_Code_by_Thousands <= 100 ~ "Special",
      Course_Code_by_Thousands <= 3000 ~ "Lower Classmen",
      Course_Code_by_Thousands <= 4000 ~ "Upper Classmen",
      TRUE ~ "Graduate"
    )
  )
}

add_gpa_category <- function(df) {
  df %>% mutate(
    GPA_Category = case_when(
      Cumulative_GPA >= 3.5 ~ "Excellent",
      Cumulative_GPA >= 3.0 ~ "Good",
      Cumulative_GPA >= 2.0 ~ "Satisfactory",
      TRUE ~ "Needs Improvement"
    )
  )
}

add_credit_load_category <- function(df) {
  df %>% mutate(
    # Credit load features
    Credit_Load_Category = case_when(
      Term_Credit_Hours <= 6 ~ "Part Time",
      Term_Credit_Hours <= 12 ~ "Half Time",
      Term_Credit_Hours <= 18 ~ "Full Time",
      TRUE ~ "Overload"
    ),
  )
}

add_class_standing_category <- function(df) {
  df %>% mutate(
    # Renaming column and values for Class_Standing
    Class_Standing_Self_Reported = case_when(
      Class_Standing == "Freshman" ~ "First Year",
      Class_Standing == "Sophomore" ~ "Second Year",
      Class_Standing == "Junior" ~ "Third Year",
      Class_Standing == "Senior" ~ "Fourth Year",
      TRUE ~ Class_Standing
    ),
  )
}

add_class_standing_bgsu <- function(df) {
  df %>% mutate(
    # Class_standing by BGSU's definition
    # https://www.bgsu.edu/academic-advising/student-resources/academic-standing.html
    Class_Standing_BGSU = case_when(
      Total_Credit_Hours_Earned < 30 ~ "Freshman",
      Total_Credit_Hours_Earned < 60 ~ "Sophomore",
      Total_Credit_Hours_Earned < 90 ~ "Junior",
      Total_Credit_Hours_Earned <= 120 ~ "Senior",
      TRUE ~ "Extended"
    ),
  )
}

# -----------------------------------------------------------------------------
# COURSE NAME CATEGORIZATION
# -----------------------------------------------------------------------------
add_course_name_category <- function(df) {
  df %>% mutate(
    Course_Name_Category = case_when(
      # Introductory level courses
      grepl("Algebra|Basic|Elementary|Intro|Introduction|Fundamental|General|Principles|Orientation", 
            Course_Name, ignore.case = TRUE) ~ "Introductory",
      
      # Intermediate level courses
      grepl("Intermediate|II$|II |2|Applied", 
            Course_Name, ignore.case = TRUE) ~ "Intermediate",
      
      # Advanced level courses
      grepl("Advanced|III|3|Analysis|Senior|Graduate|Dissertation|Research|Capstone", 
            Course_Name, ignore.case = TRUE) ~ "Advanced",
      
      # Business related courses
      grepl("Business|Finance|Accounting|Economics|Marketing|Management", 
            Course_Name, ignore.case = TRUE) ~ "Business",
      
      # Laboratory/Practical courses
      grepl("Laboratory|Lab", Course_Name, ignore.case = TRUE) ~ "Laboratory",
      
      # Seminar/Workshop courses
      grepl("Seminar|Workshop", Course_Name, ignore.case = TRUE) ~ "Seminar",
      
      # Independent/Special courses
      grepl("Independent|Special", Course_Name, ignore.case = TRUE) ~ "Independent Study",
      
      # Mathematics and Statistics
      grepl("Mathematics|Calculus|Statistics|Probability|Geometry|Discrete", 
            Course_Name, ignore.case = TRUE) ~ "Mathematics",
      
      # Computer Science
      grepl("Computer|Programming|Data|Software|Network|Database|Algorithm", 
            Course_Name, ignore.case = TRUE) ~ "Computer Science",
      
      # Natural Sciences
      grepl("Physics|Chemistry|Biology|Astronomy|Earth|Environment|Science", 
            Course_Name, ignore.case = TRUE) ~ "Natural Sciences",
      
      # Social Sciences
      grepl("Psychology|Sociology|Anthropology|Social|Cultural|Society", 
            Course_Name, ignore.case = TRUE) ~ "Social Sciences",
      
      # Humanities
      grepl("History|Philosophy|Ethics|Literature|Culture|Language|Art", 
            Course_Name, ignore.case = TRUE) ~ "Humanities",
      
      # Education/Teaching
      grepl("Education|Teaching|Learning|Childhood|Teacher|Curriculum", 
            Course_Name, ignore.case = TRUE) ~ "Education",
      
      # Default case
      TRUE ~ "Other"
    )
  )
}

# -----------------------------------------------------------------------------
# COURSE TYPE CATEGORIZATION
# -----------------------------------------------------------------------------
add_course_type_category <- function(df) {
  df %>% mutate(
    Course_Type_Category = case_when(
      # STEM Fields
      Course_Type %in% c("MATH", "STAT", "CS", "ASTR","PHYS", "BIOL", "CHEM", "GEOL", "ECET") ~ "STEM Core",
      
      # Engineering and Technology
      Course_Type %in% c("ENGT", "CONS", "ARCH", "MIS", "TECH") ~ "Engineering & Technology",
      
      # Business and Economics
      Course_Type %in% c("FIN", "ACCT", "ECON", "BA", "MGMT", "MKT", "MBA", "BIZX", "LEGS", "OR") ~ "Business",
      
      # Social Sciences
      Course_Type %in% c("SOC", "PSYC", "POLS", "CRJU", "HDFS", "SOWK", "GERO") ~ "Social Sciences",
      
      # Natural and Health Sciences
      Course_Type %in% c("NURS", "MLS", "EXSC", "FN", "AHTH", "DHS") ~ "Health Sciences",
      
      # Humanities and Languages
      Course_Type %in% c("HIST", "PHIL", "ENG", "GSW", "FREN", "GERM", "SPAN", "LAT", "RUSN", "ITAL", "CLCV") ~ "Humanities",
      
      # Arts and Performance
      Course_Type %in% c("ART", "ID", "MUCT", "MUS", "THFM", "POPC") ~ "Arts",
      
      # Education and Teaching
      Course_Type %in% c("EDTL", "EDFI", "EDIS", "EIEC") ~ "Education",
      
      # Environmental Studies
      Course_Type %in% c("ENVS", "GEOG", "SEES") ~ "Environmental Studies",
      
      # Special Programs
      Course_Type %in% c("HNRS", "UNIV", "ORGD", "RESC") ~ "Special Programs",
      
      # Physical Education
      Course_Type %in% c("PEG", "SM", "HMSL") ~ "Physical Education",
      
      # Cultural Studies
      Course_Type %in% c("ETHN", "COMM", "CDIS") ~ "Cultural & Communication Studies",
      
      # No Response/Unknown
      Course_Type %in% c("No Response", NA) ~ "No Response",
      
      # Default case
      TRUE ~ "Other"
    )
  )
}

# -----------------------------------------------------------------------------
# MAJOR CATEGORIZATION
# -----------------------------------------------------------------------------
add_major_category <- function(df) {
  df %>% mutate(
    Major_Category = case_when(
      # Business and Management
      grepl("MBA|BSBA|Business|Marketing|Finance|Account|Economics|Management|Supply Chain|Analytics", 
            Major, ignore.case = TRUE) ~ "Business",
      
      # Computer Science and Technology
      grepl("Computer|Software|Data|Information Systems|Technology|Engineering|Electronics", 
            Major, ignore.case = TRUE) ~ "Computing & Technology",
      
      # Natural Sciences
      grepl("Biology|Chemistry|Physics|Science|Environmental|Geology|Forensic|Neuroscience", 
            Major, ignore.case = TRUE) ~ "Natural Sciences",
      
      # Health Sciences
      grepl("Nursing|Health|Medical|Nutrition|Dietetics|Physical Therapy|Physician|Laboratory", 
            Major, ignore.case = TRUE) ~ "Health Sciences",
      
      # Social Sciences
      grepl("Psychology|Sociology|Criminal Justice|Political|Economics|Social Work|Anthropology", 
            Major, ignore.case = TRUE) ~ "Social Sciences",
      
      # Education
      grepl("Education|Teaching|Early Childhood|BSED|Intervention Specialist", 
            Major, ignore.case = TRUE) ~ "Education",
      
      # Arts and Humanities
      grepl("Art|Music|Philosophy|History|English|Language|Communication|Media|Journalism|Film|Theatre", 
            Major, ignore.case = TRUE) ~ "Arts & Humanities",
      
      # Mathematics and Statistics
      grepl("Math|Statistics|Actuarial", 
            Major, ignore.case = TRUE) ~ "Mathematics",
      
      # Pre-Professional Programs
      grepl("Pre-|PRELAW|PREMED|PREVET", 
            Major, ignore.case = TRUE) ~ "Pre-Professional",
      
      # Undecided/General Studies
      grepl("Undecided|Liberal Studies|General|Deciding|UND|Individual|BLS", 
            Major, ignore.case = TRUE) ~ "General Studies",
      
      # Special Programs
      grepl("Minor|Certificate|GCERT|Non-Degree", 
            Major, ignore.case = TRUE) ~ "Special Programs",
      
      # No Response/Unknown
      grepl("No Response|NA", Major, ignore.case = TRUE) ~ "No Response",
      
      # Default case
      TRUE ~ "Other"
    ),
    
    # Add a flag for double majors
    Has_Multiple_Majors = grepl(",", Major)
  )
}

# -----------------------------------------------------------------------------
# VISITS FEATURES
# -----------------------------------------------------------------------------
add_visit_features <- function(df) {
  df %>%
    group_by(Student_IDs) %>%
    mutate(
      # Count visits per student
      Total_Visits = n(),
      # Count visits per student per semester
      Semester_Visits = n_distinct(Check_In_Date),
      # Average visits per week
      Avg_Weekly_Visits = Semester_Visits / max(Semester_Week)
    ) %>%
    ungroup()
}

add_week_volume_category <- function(df) {
  df %>%
    mutate(
      Week_Volume = case_when(
        Semester_Week %in% c(4:8, 10:13, 15:16) ~ "High Volume",
        Semester_Week %in% c(1:3, 9, 14, 17) ~ "Low Volume",
        TRUE ~ "Other"
      )
    )
}

# -----------------------------------------------------------------------------
# COURSE LOAD FEATURES
# -----------------------------------------------------------------------------
add_course_load_features <- function(df) {
  df %>%
    group_by(Student_IDs, Semester) %>%
    mutate(
      # Number of unique courses
      Unique_Courses = n_distinct(Course_Number),
      # Mix of course levels
      Course_Level_Mix = n_distinct(Course_Code_by_Thousands),
      # Proportion of advanced courses
      Advanced_Course_Ratio = mean(Course_Level == "Upper Classmen", na.rm = TRUE)
    ) %>%
    ungroup()
}

add_gpa_trend <- function(df) {
  df %>% mutate(
    # Calculate GPA trend (1 for positive, -1 for negative, 0 for no change)
    GPA_Trend = sign(Change_in_GPA),
    # Add categorical version
    GPA_Trend_Category = case_when(
      Change_in_GPA > 0 ~ "Improving",
      Change_in_GPA < 0 ~ "Declining",
      TRUE ~ "Stable"
    )
  )
}

# -----------------------------------------------------------------------------
# RESPONSE/ESQUE FEATURES
# -----------------------------------------------------------------------------
ensure_duration <- function(df) {
  # Calculate duration in minutes
  df %>%
    mutate(
      Duration_In_Min = as.numeric(difftime(
        Check_Out_Time,
        Check_In_Time,
        units = "mins"
      )),
      # Filter out negative durations
      Duration_In_Min = if_else(Duration_In_Min < 0, NA_real_, Duration_In_Min),
    ) %>%
    filter(!is.na(Duration_In_Min))
}

add_session_length_category <- function(df) {
  df %>% mutate(
    # Add session length categories
    Session_Length_Category = case_when(
      Duration_In_Min <= 30 ~ "Short",
      Duration_In_Min <= 90 ~ "Medium",
      Duration_In_Min <= 180 ~ "Long",
      Duration_In_Min > 180 ~ "Extended",
      TRUE ~ NA_character_
    )
  )
}

calculate_occupancy <- function(df) {
  df %>%
    arrange(Check_In_Date, Check_In_Time) %>%
    group_by(Check_In_Date) %>%
    mutate(
      Cum_Arrivals = row_number(),
      Cum_Departures = sapply(seq_along(Check_In_Time), function(i) {
        sum(!is.na(Check_Out_Time[1:i]) & 
            Check_Out_Time[1:i] <= Check_In_Time[i])
      }),
      Occupancy = Cum_Arrivals - Cum_Departures
    ) %>%
    select(-c(Cum_Arrivals, Cum_Departures))
}

# -----------------------------------------------------------------------------
# PIPELINE
# -----------------------------------------------------------------------------
engineer_features <- function(df) {
  df %>%
    prepare_dates() %>%
    add_date_features() %>%
    add_temporal_features() %>%
    add_time_category() %>%
    add_course_features() %>%
    add_course_name_category() %>%
    add_course_type_category() %>%
    add_major_category() %>%
    add_gpa_category() %>%
    add_credit_load_category() %>%
    add_class_standing_category() %>%
    add_class_standing_bgsu() %>%
    ensure_duration() %>%
    add_session_length_category() %>%
    add_visit_features() %>%
    add_week_volume_category() %>%
    add_graduation_features() %>%
    add_course_load_features() %>%
    add_gpa_trend() %>%
    calculate_occupancy() %>%
    ungroup()
}

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
data_raw <- readr::read_csv(here("data", "raw", "LC_train.csv"))
lc_engineered <- engineer_features(data_raw)
readr::write_csv(lc_engineered, here("data", "processed", "train_engineered.csv"))

# lc_engineered %>%
#   filter(Course_Type_Category == "Other") %>%
#   select(Course_Type) %>%
#   distinct() %>%
#   print(n = 150)

# -----------------------------------------------------------------------------
# VIEW ENGINEERED DATA
# -----------------------------------------------------------------------------
# View(lc_engineered)
