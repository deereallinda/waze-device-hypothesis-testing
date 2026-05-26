# app.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Waze Device Type Hypothesis Test",
    page_icon="🚗",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #4b5563;
        margin-top: 0px;
    }

    .section-header {
        font-size: 28px;
        font-weight: 700;
        color: #111827;
        margin-top: 35px;
        margin-bottom: 15px;
    }

    .insight-box {
        background-color: #f9fafb;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #2563eb;
        margin-top: 15px;
        margin-bottom: 15px;
        color: #111827;
    }

    .success-box {
        background-color: #ecfdf5;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #10b981;
        margin-top: 15px;
        margin-bottom: 15px;
        color: #064e3b;
    }

    .warning-box {
        background-color: #fffbeb;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #f59e0b;
        margin-top: 15px;
        margin-bottom: 15px;
        color: #78350f;
    }

    .danger-box {
        background-color: #fef2f2;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #ef4444;
        margin-top: 15px;
        margin-bottom: 15px;
        color: #7f1d1d;
    }

    .small-text {
        font-size: 14px;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def load_data():
    """
    Loads the Waze dataset.

    The function checks multiple possible file locations so the app works
    locally and on Streamlit Cloud.
    """

    possible_paths = [
        Path("waze_dataset.csv"),
        Path("data/waze_dataset.csv"),
        Path("./waze_dataset.csv"),
        Path("./data/waze_dataset.csv")
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)

    return None


df = load_data()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<p class="main-title">🚗 Waze Device Type Hypothesis Test</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Testing whether iPhone and Android users have different average numbers of drives.</p>',
    unsafe_allow_html=True
)

st.markdown("---")


# ---------------------------------------------------------
# DATA FILE CHECK
# ---------------------------------------------------------

if df is None:
    st.error(
        "Dataset not found. Please make sure `waze_dataset.csv` is placed "
        "inside the project root folder or inside a `data/` folder."
    )

    st.info(
        """
        Recommended project structure:

        ```text
        waze-device-hypothesis-testing/
        │
        ├── app.py
        ├── requirements.txt
        ├── data/
        │   └── waze_dataset.csv
        ```
        """
    )

    st.stop()


# ---------------------------------------------------------
# BASIC DATA PREPARATION
# ---------------------------------------------------------

required_columns = ["device", "drives"]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(
        f"The dataset is missing the following required column(s): {missing_columns}"
    )
    st.stop()


# Create numeric device type column if it does not already exist
device_map = {
    "iPhone": 1,
    "Android": 2
}

df["device_type"] = df["device"].map(device_map)

# Keep only iPhone and Android rows for the main hypothesis test
test_df = df[df["device"].isin(["iPhone", "Android"])].copy()

iphone = test_df[test_df["device"] == "iPhone"]["drives"].dropna()
android = test_df[test_df["device"] == "Android"]["drives"].dropna()

iphone_mean = iphone.mean()
android_mean = android.mean()
difference = iphone_mean - android_mean

alpha = 0.05

t_stat, p_value = stats.ttest_ind(
    a=iphone,
    b=android,
    equal_var=False
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🚗 Project Navigation")

page = st.sidebar.radio(
    "Go to section:",
    [
        "Project Overview",
        "Dataset Overview",
        "Exploratory Analysis",
        "Hypothesis Testing",
        "Business Insights",
        "Executive Summary"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Case Study:** Waze User Churn Project  
    **Focus:** Device type vs number of drives  
    **Method:** Two-sample t-test  
    **Groups:** iPhone users and Android users
    """
)


# ---------------------------------------------------------
# PAGE 1: PROJECT OVERVIEW
# ---------------------------------------------------------

if page == "Project Overview":

    st.markdown(
        '<p class="section-header">📌 Project Overview</p>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This project analyzes Waze user data to determine whether there is a
        statistically significant difference in the average number of drives between
        users who access Waze using **iPhone** devices and users who access Waze using
        **Android** devices.

        The analysis is based on a business request from leadership as part of a
        broader user churn project.
        """
    )

    st.markdown(
        """
        <div class="insight-box">
        <strong>Research Question:</strong><br>
        Do drivers who open the Waze application using an iPhone have the same number
        of drives on average as drivers who use Android devices?
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Why This Analysis Matters")

    st.write(
        """
        Waze leadership wants to better understand user behavior. If driving activity
        differs by device type, this could suggest that iPhone and Android users engage
        with the app differently.

        That insight could influence:

        - Product strategy
        - User experience design
        - Marketing decisions
        - Churn analysis
        - Platform-specific feature testing
        """
    )

    st.markdown("### Statistical Method Used")

    st.write(
        """
        This project uses a **two-sample t-test** to compare the average number of
        drives between two independent groups:

        - **Group 1:** iPhone users
        - **Group 2:** Android users

        The test helps determine whether the observed difference in averages is likely
        to be a real difference or simply the result of random variation.
        """
    )

    st.markdown(
        """
        <div class="warning-box">
        <strong>Important note:</strong><br>
        A hypothesis test can tell us whether a difference is statistically significant,
        but it does not automatically explain why that difference exists.
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# PAGE 2: DATASET OVERVIEW
# ---------------------------------------------------------

elif page == "Dataset Overview":

    st.markdown(
        '<p class="section-header">📊 Dataset Overview</p>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This section provides a quick overview of the Waze dataset used for the analysis.
        The key variables for this project are:
        
        - **device:** whether the user is on iPhone or Android
        - **drives:** number of drives completed by the user
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Number of Rows",
            value=f"{df.shape[0]:,}",
            help="Total number of records in the dataset."
        )

    with col2:
        st.metric(
            label="Number of Columns",
            value=f"{df.shape[1]:,}",
            help="Total number of variables in the dataset."
        )

    with col3:
        st.metric(
            label="Device Types",
            value=f"{df['device'].nunique()}",
            help="Number of unique device categories in the dataset."
        )

    st.markdown("### Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Column Summary")

    column_summary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Missing %": (df.isna().mean().values * 100).round(2)
    })

    st.dataframe(column_summary, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
        <strong>What are we looking at?</strong><br>
        This table helps us understand the structure and quality of the dataset.
        Missing values, incorrect data types, or unexpected columns can affect the
        reliability of the analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Descriptive Statistics")

    st.dataframe(df.describe(include="all"), use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
        <strong>What does this mean?</strong><br>
        Descriptive statistics give us a quick summary of the dataset. They show
        values such as averages, minimums, maximums, and variation in the numerical
        variables.
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# PAGE 3: EXPLORATORY ANALYSIS
# ---------------------------------------------------------

elif page == "Exploratory Analysis":

    st.markdown(
        '<p class="section-header">🔎 Exploratory Analysis</p>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This section explores how driving activity differs across device types before
        performing the formal hypothesis test.
        """
    )

    st.markdown("### Device Type Distribution")

    device_counts = (
        test_df["device"]
        .value_counts()
        .reset_index()
    )

    device_counts.columns = ["Device", "Number of Users"]

    st.dataframe(device_counts, use_container_width=True)

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.bar(device_counts["Device"], device_counts["Number of Users"])
    ax1.set_title("Number of Users by Device Type")
    ax1.set_xlabel("Device Type")
    ax1.set_ylabel("Number of Users")

    st.pyplot(fig1)

    st.markdown(
        """
        <div class="insight-box">
        <strong>What are we looking at?</strong><br>
        This chart shows how many users in the dataset are using iPhone compared to
        Android. This helps us understand the size of each comparison group before
        running the hypothesis test.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Average Number of Drives by Device")

    device_summary = (
        test_df
        .groupby("device")["drives"]
        .agg(
            users="count",
            average_drives="mean",
            median_drives="median",
            standard_deviation="std",
            minimum_drives="min",
            maximum_drives="max"
        )
        .reset_index()
    )

    device_summary["average_drives"] = device_summary["average_drives"].round(2)
    device_summary["median_drives"] = device_summary["median_drives"].round(2)
    device_summary["standard_deviation"] = device_summary["standard_deviation"].round(2)

    st.dataframe(device_summary, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.bar(device_summary["device"], device_summary["average_drives"])
    ax2.set_title("Average Number of Drives by Device Type")
    ax2.set_xlabel("Device Type")
    ax2.set_ylabel("Average Number of Drives")

    st.pyplot(fig2)

    st.markdown(
        """
        <div class="insight-box">
        <strong>What does this mean?</strong><br>
        This chart compares the average number of drives between iPhone and Android
        users. A visible difference may appear, but we still need a hypothesis test
        to know whether the difference is statistically significant.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### iPhone vs Android Key Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="iPhone Average Drives",
            value=f"{iphone_mean:,.2f}",
            help="Average number of drives for users on iPhone."
        )

    with col2:
        st.metric(
            label="Android Average Drives",
            value=f"{android_mean:,.2f}",
            help="Average number of drives for users on Android."
        )

    with col3:
        st.metric(
            label="Difference",
            value=f"{difference:,.2f}",
            help="Difference between average iPhone drives and average Android drives."
        )

    st.markdown("### Distribution of Drives")

    max_drives = int(test_df["drives"].quantile(0.95))

    selected_max = st.slider(
        "Select maximum number of drives to display:",
        min_value=1,
        max_value=max_drives,
        value=max_drives,
        step=1
    )

    visual_df = test_df[
        (test_df["drives"] >= 0) &
        (test_df["drives"] <= selected_max)
    ]

    iphone_visual = visual_df[visual_df["device"] == "iPhone"]["drives"]
    android_visual = visual_df[visual_df["device"] == "Android"]["drives"]

    fig3, ax3 = plt.subplots(figsize=(8, 5))

    ax3.hist(iphone_visual, bins=40, alpha=0.6, label="iPhone")
    ax3.hist(android_visual, bins=40, alpha=0.6, label="Android")

    ax3.set_title(f"Distribution of Drives by Device Type: Up to {selected_max} Drives")
    ax3.set_xlabel("Number of Drives")
    ax3.set_ylabel("Number of Users")
    ax3.legend()

    st.pyplot(fig3)

    st.markdown(
        """
        <div class="insight-box">
        <strong>How to read this chart:</strong><br>
        This histogram shows how the number of drives is distributed for iPhone and
        Android users. Most users are concentrated within a certain drive range, while
        some users may have much higher drive counts.
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# PAGE 4: HYPOTHESIS TESTING
# ---------------------------------------------------------

elif page == "Hypothesis Testing":

    st.markdown(
        '<p class="section-header">🧪 Hypothesis Testing</p>',
        unsafe_allow_html=True
    )

    st.write(
        """
        To test whether iPhone and Android users have different average numbers of
        drives, we conduct a **two-sample t-test**.
        """
    )

    st.markdown("### Research Question")

    st.markdown(
        """
        <div class="insight-box">
        Do drivers who open the Waze application using an iPhone have the same number
        of drives on average as drivers who use Android devices?
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Hypotheses")

    st.write(
        """
        **Null Hypothesis H₀:**  
        There is no difference in the average number of drives between iPhone users
        and Android users.

        **Alternative Hypothesis H₁:**  
        There is a difference in the average number of drives between iPhone users
        and Android users.
        """
    )

    st.markdown("### Test Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Significance Level",
            value=f"{alpha:.0%}",
            help="The threshold used to decide whether the result is statistically significant."
        )

    with col2:
        st.metric(
            label="T-statistic",
            value=f"{t_stat:,.4f}",
            help="Measures the difference between group means relative to variation in the data."
        )

    with col3:
        st.metric(
            label="P-value",
            value=f"{p_value:.4f}",
            help="Probability of observing a difference this large if the null hypothesis were true."
        )

    st.markdown("### Result")

    if p_value < alpha:
        st.markdown(
            """
            <div class="success-box">
            <strong>Decision:</strong> Reject the null hypothesis.<br><br>
            The p-value is smaller than the 5% significance level. This means there is
            a statistically significant difference in average number of drives between
            iPhone and Android users.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="warning-box">
            <strong>Decision:</strong> Fail to reject the null hypothesis.<br><br>
            The p-value is greater than the 5% significance level. This means there is
            not enough evidence to conclude that iPhone and Android users have different
            average numbers of drives.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Interpretation")

    st.write(
        f"""
        The average number of drives for iPhone users is **{iphone_mean:,.2f}**.

        The average number of drives for Android users is **{android_mean:,.2f}**.

        The observed difference is **{difference:,.2f}** drives.

        The p-value is **{p_value:.4f}**.
        """
    )

    if p_value >= alpha:
        st.markdown(
            """
            <div class="insight-box">
            <strong>Plain English explanation:</strong><br>
            Even if the averages are slightly different, the test result suggests that
            the difference is not statistically significant. In this dataset, device
            type does not appear to strongly explain the number of drives.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="insight-box">
            <strong>Plain English explanation:</strong><br>
            The test result suggests that average driving activity differs between
            iPhone and Android users.
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# PAGE 5: BUSINESS INSIGHTS
# ---------------------------------------------------------

elif page == "Business Insights":

    st.markdown(
        '<p class="section-header">💼 Business Insights</p>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This section translates the statistical result into practical business meaning
        for Waze leadership.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="iPhone Average Drives",
            value=f"{iphone_mean:,.2f}",
            help="Average number of drives for iPhone users."
        )

    with col2:
        st.metric(
            label="Android Average Drives",
            value=f"{android_mean:,.2f}",
            help="Average number of drives for Android users."
        )

    st.markdown("### Key Business Insight")

    if p_value >= alpha:
        st.markdown(
            """
            <div class="success-box">
            The key business insight is that iPhone and Android users appear to have
            similar average driving activity. Device type does not seem to be a major
            factor explaining differences in the number of drives.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="success-box">
            The analysis suggests that device type is associated with a statistically
            significant difference in average driving activity.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Business Recommendation")

    if p_value >= alpha:
        st.write(
            """
            Since there is not enough evidence to conclude that iPhone and Android users
            have different average numbers of drives, Waze should avoid making major
            device-specific decisions based only on this variable.

            Instead, leadership should explore other factors that may better explain
            user engagement and churn risk.
            """
        )
    else:
        st.write(
            """
            Since the test suggests a statistically significant difference by device
            type, Waze may want to investigate whether platform-specific user experience,
            app performance, or marketing factors are influencing driving behavior.
            """
        )

    st.markdown("### Recommended Next Analysis")

    st.write(
        """
        Waze should consider analyzing additional variables, such as:

        - Number of app sessions
        - Number of driving days
        - Total kilometers driven
        - User tenure
        - Churn status
        - Activity level
        - Platform-specific app experience
        - Regional or demographic patterns
        """
    )

    st.markdown(
        """
        <div class="insight-box">
        <strong>Why this matters:</strong><br>
        Device type alone may not explain user behavior. A stronger churn analysis
        should investigate broader engagement patterns and user activity metrics.
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# PAGE 6: EXECUTIVE SUMMARY
# ---------------------------------------------------------

elif page == "Executive Summary":

    st.markdown(
        '<p class="section-header">📝 Executive Summary</p>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This project analyzed whether there is a statistically significant difference
        in the average number of drives between Waze users on iPhone and Android devices.
        """
    )

    st.markdown("### Summary of Results")

    summary_df = pd.DataFrame({
        "Metric": [
            "iPhone average drives",
            "Android average drives",
            "Difference",
            "T-statistic",
            "P-value",
            "Significance level"
        ],
        "Value": [
            round(iphone_mean, 2),
            round(android_mean, 2),
            round(difference, 2),
            round(t_stat, 4),
            round(p_value, 4),
            alpha
        ]
    })

    st.dataframe(summary_df, use_container_width=True)

    st.markdown("### Final Conclusion")

    if p_value >= alpha:
        st.markdown(
            """
            <div class="success-box">
            The two-sample t-test produced a p-value greater than the 5% significance
            level. Therefore, we fail to reject the null hypothesis.
            <br><br>
            This means there is not enough evidence to conclude that iPhone users and
            Android users have different average numbers of drives.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="success-box">
            The two-sample t-test produced a p-value smaller than the 5% significance
            level. Therefore, we reject the null hypothesis.
            <br><br>
            This means there is evidence of a statistically significant difference in
            average number of drives between iPhone and Android users.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Stakeholder Message")

    if p_value >= alpha:
        st.write(
            """
            From a business perspective, device type does not appear to be a strong
            differentiator of driving activity. Waze leadership should focus further
            analysis on behavioral and engagement-related variables, such as sessions,
            driving days, kilometers driven, and churn status.
            """
        )
    else:
        st.write(
            """
            From a business perspective, the result suggests that device type may be
            related to driving activity. Waze leadership should investigate whether
            app experience, platform performance, or user behavior differs between
            iPhone and Android users.
            """
        )

    st.markdown("### Limitation")

    st.markdown(
        """
        <div class="warning-box">
        <strong>Important limitation:</strong><br>
        This test only compares average drives by device type. It does not explain
        why differences may or may not exist. Other variables may have stronger
        relationships with user activity and churn.
        </div>
        """,
        unsafe_allow_html=True
    )