# 🚗 Waze Device Type Hypothesis Testing

## Project Overview

This project explores whether there is a statistically significant difference in the average number of drives between Waze users who use **iPhone** devices and those who use **Android** devices.

The project is based on the **Waze user churn case study** from *Course 3: The Power of Statistics*. In the case study, Waze leadership requested a statistical analysis to determine whether device type is related to user driving activity. The specific task was to conduct a two-sample hypothesis test comparing the mean number of drives between iPhone and Android users.

---

## Business Problem

Waze leadership wants to better understand user behavior as part of a broader user churn analysis.

One question raised by leadership is whether users on different device types behave differently in terms of driving activity.

The key business question is:

> Do drivers who open the Waze application using an iPhone have the same average number of drives as drivers who use Android devices?

Understanding this can help Waze decide whether device type should be considered when analyzing user engagement, churn risk, marketing strategies, or app experience improvements.

---

## Research Question

The research question for this project is:

> Is there a statistically significant difference in the average number of drives between iPhone users and Android users?

---

## Dataset

The project uses the file:

```text
waze_dataset.csv

The dataset contains user-level Waze activity data. For this specific analysis, the two main variables are:

Column	Description
device	The type of device used by the Waze user, either iPhone or Android
drives	The number of drives completed by the user

The device column is categorical, so it was converted into a numeric variable called device_type for analysis:

Device	Encoded Value
iPhone	1
Android	2
Tools and Technologies Used

This project was completed using:

Python
Pandas
NumPy
SciPy
Matplotlib
Streamlit
GitHub
Streamlit Community Cloud
Project Methodology

The project followed a structured data analysis process:

1. Data Loading

The Waze dataset was loaded into Python using Pandas.

2. Data Exploration

Basic exploratory data analysis was conducted to understand:

Dataset structure
Number of rows and columns
Missing values
Device type distribution
Average number of drives by device type
3. Feature Preparation

The device column was mapped into a new numeric column called device_type.

map_dictionary = {
    "iPhone": 1,
    "Android": 2
}

df["device_type"] = df["device"].map(map_dictionary)
4. Descriptive Statistics

The average number of drives was calculated for each device group.

This helped compare the driving activity of iPhone and Android users before conducting the formal hypothesis test.

5. Hypothesis Testing

A two-sample t-test was conducted to determine whether the difference in average drives between iPhone and Android users was statistically significant.

Welch’s t-test was used because it does not assume equal population variances.

stats.ttest_ind(a=iphone, b=android, equal_var=False)
Hypotheses
Null Hypothesis

There is no difference in the average number of drives between iPhone users and Android users.

Alternative Hypothesis

There is a difference in the average number of drives between iPhone users and Android users.

The significance level used was:

α = 0.05
Key Finding

The hypothesis test produced a p-value greater than the 5% significance level.

Therefore, the analysis failed to reject the null hypothesis.

This means there is not enough statistical evidence to conclude that iPhone users and Android users have different average numbers of drives.

Business Insight

The key business insight is that device type does not appear to be a major factor explaining differences in average driving activity.

Although iPhone and Android users may show slightly different average drive counts in the sample, the difference is not statistically significant.

This suggests that Waze leadership should avoid making major device-specific business decisions based only on this variable.

Business Recommendation

Since device type does not appear to significantly explain differences in driving activity, Waze should explore other variables that may better explain user behavior and churn risk.

Recommended next steps include analyzing:

Number of app sessions
Number of driving days
Total kilometers driven
User tenure
Churn status
Activity level
Platform-specific app experience
Regional or behavioral patterns

These variables may provide stronger insights into user engagement and retention.

Streamlit Dashboard

This project includes an interactive Streamlit dashboard that presents:

Project overview
Dataset overview
Device type distribution
Average drives comparison
Hypothesis test results
Business insights
Executive summary

For Windows PowerShell:

.venv\Scripts\Activate.ps1
5. Install dependencies
pip install -r requirements.txt
6. Run the Streamlit app
streamlit run app.py
Project Structure
waze-device-hypothesis-testing/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── waze_dataset.csv
│
└── notebook/
    └── waze_hypothesis_testing.ipynb
Requirements
The project requires the following Python libraries:

streamlit
pandas
numpy
scipy
matplotlib
Limitations

This analysis only compares the average number of drives by device type.

It does not explain why differences may or may not exist. Other variables may have stronger relationships with user activity and churn.

Also, this test does not prove causation. It only tests whether there is a statistically significant difference in average drives between two device groups.

Executive Summary

This project analyzed whether there is a statistically significant difference in average driving activity between Waze users on iPhone and Android devices.

A two-sample t-test was conducted using the number of drives as the outcome variable and device type as the comparison group.

The test result showed that the p-value was greater than the 5% significance level. Therefore, the analysis failed to reject the null hypothesis.

This means there is not enough evidence to conclude that iPhone and Android users have different average numbers of drives.

From a business perspective, device type does not appear to be a strong differentiator of driving activity. Waze leadership should focus further analysis on behavioral and engagement-related variables, such as sessions, driving days, kilometers driven, and churn status.

Author

Linda
Data Analyst | Business Analyst | Aspiring Data Scientist