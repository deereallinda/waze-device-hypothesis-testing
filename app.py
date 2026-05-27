# app.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Waze Hypothesis Testing Dashboard",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
        color: #f8fafc;
    }

    .subtitle {
        font-size: 18px;
        color: #cbd5e1;
        margin-top: 0px;
        margin-bottom: 20px;
    }

    .section-header {
        font-size: 30px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 15px;
        color: #f8fafc;
    }

    .sub-header {
        font-size: 22px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #f8fafc;
    }

    .hero-box {
        background: linear-gradient(135deg, #1e3a8a, #0f172a);
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 25px;
        border: 1px solid #334155;
    }

    .hero-box h1 {
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .hero-box p {
        color: #dbeafe;
        font-size: 17px;
        line-height: 1.6;
    }

    .insight-box {
        background-color: #e0f2fe;
        color: #0f172a;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #0284c7;
        margin-top: 15px;
        margin-bottom: 15px;
        line-height: 1.6;
    }

    .success-box {
        background-color: #dcfce7;
        color: #052e16;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #16a34a;
        margin-top: 15px;
        margin-bottom: 15px;
        line-height: 1.6;
    }

    .warning-box {
        background-color: #fef3c7;
        color: #451a03;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #f59e0b;
        margin-top: 15px;
        margin-bottom: 15px;
        line-height: 1.6;
    }

    .danger-box {
        background-color: #fee2e2;
        color: #450a0a;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #dc2626;
        margin-top: 15px;
        margin-bottom: 15px;
        line-height: 1.6;
    }

    .neutral-box {
        background-color: #f8fafc;
        color: #0f172a;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #64748b;
        margin-top: 15px;
        margin-bottom: 15px;
        line-height: 1.6;
    }

    .small-note {
        font-size: 14px;
        color: #94a3b8;
    }

    div[data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #334155;
    }

    div[data-testid="stMetric"] label {
        color: #cbd5e1;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    hr {
        border-color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_data():
    """
    Loads the Waze dataset from common local or Streamlit Cloud paths.
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


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero-box">
        <h1>🚗 Waze Device Type Hypothesis Testing Dashboard</h1>
        <p>
        An educational, business-focused dashboard that tests whether iPhone and Android users
        have different average numbers of drives.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATA CHECK
# =========================================================

if df is None:
    st.error(
        "Dataset not found. Please place `waze_dataset.csv` inside the project root folder "
        "or inside a `data/` folder."
    )

    st.info(
        """
        Your folder should look like this:

        ```text
        waze-device-hypothesis-testing/
        │
        ├── app.py
        ├── requirements.txt
        ├── README.md
        └── data/
            └── waze_dataset.csv
        ```
        """
    )

    st.stop()


required_columns = ["device", "drives"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"The dataset is missing these required columns: {missing_columns}")
    st.stop()


# =========================================================
# DATA PREPARATION
# =========================================================

df = df.copy()

device_map = {
    "iPhone": 1,
    "Android": 2
}

df["device_type"] = df["device"].map(device_map)

test_df = df[df["device"].isin(["iPhone", "Android"])].copy()

iphone = test_df[test_df["device"] == "iPhone"]["drives"].dropna()
android = test_df[test_df["device"] == "Android"]["drives"].dropna()

iphone_mean = iphone.mean()
android_mean = android.mean()
difference = iphone_mean - android_mean

iphone_median = iphone.median()
android_median = android.median()

iphone_std = iphone.std()
android_std = android.std()

alpha = 0.05

t_stat, p_value = stats.ttest_ind(
    a=iphone,
    b=android,
    equal_var=False
)


def cohen_d(group1, group2):
    """
    Calculates Cohen's d effect size.
    """
    n1 = len(group1)
    n2 = len(group2)

    s1 = np.var(group1, ddof=1)
    s2 = np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(group1) - np.mean(group2)) / pooled_std


effect_size = cohen_d(iphone, android)


def interpret_p_value(p_value, alpha):
    if p_value < alpha:
        return "Reject the null hypothesis"
    return "Fail to reject the null hypothesis"


def interpret_effect_size(d):
    if pd.isna(d):
        return "Not available"
    abs_d = abs(d)

    if abs_d < 0.2:
        return "Very small effect"
    elif abs_d < 0.5:
        return "Small effect"
    elif abs_d < 0.8:
        return "Medium effect"
    else:
        return "Large effect"


decision = interpret_p_value(p_value, alpha)
effect_interpretation = interpret_effect_size(effect_size)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🚗 Waze Project Guide")

st.sidebar.markdown(
    """
    This dashboard is designed like a guided case study.

    Follow the pages from top to bottom:
    """
)

page = st.sidebar.radio(
    "Choose a section:",
    [
        "1. Start Here",
        "2. Understand the Data",
        "3. Compare Devices",
        "4. Learn the Hypothesis Test",
        "5. Results and Interpretation",
        "6. Stakeholder Summary",
        "7. Glossary"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Project Snapshot")

st.sidebar.info(
    f"""
    **Dataset rows:** {df.shape[0]:,}  
    **Dataset columns:** {df.shape[1]:,}  
    **Test:** Two-sample t-test  
    **Groups:** iPhone vs Android  
    **Significance level:** {alpha:.0%}
    """
)

st.sidebar.markdown("### Final Decision")

if p_value < alpha:
    st.sidebar.success("Reject H₀")
else:
    st.sidebar.warning("Fail to reject H₀")


# =========================================================
# PAGE 1: START HERE
# =========================================================

if page == "1. Start Here":

    st.markdown('<p class="section-header">📌 1. Start Here</p>', unsafe_allow_html=True)

    st.write(
        """
        This project is a statistical analysis case study based on Waze user data.
        The goal is to help leadership understand whether **device type** is related to
        the **average number of drives** completed by users.
        """
    )

    st.markdown(
        """
        <div class="insight-box">
        <strong>Business question:</strong><br>
        Do Waze users on iPhone have the same average number of drives as Waze users on Android?
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### What this dashboard teaches")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="neutral-box">
            <strong>1. Descriptive statistics</strong><br>
            Learn how averages, medians, and standard deviations help summarize user behavior.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="neutral-box">
            <strong>2. Hypothesis testing</strong><br>
            Learn how to test whether a difference between two groups is statistically meaningful.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="neutral-box">
            <strong>3. Business interpretation</strong><br>
            Learn how to turn statistical results into stakeholder-friendly recommendations.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Case Study Context")

    st.write(
        """
        Waze leadership is interested in user churn and engagement. One question is whether
        users behave differently depending on their device type.

        In this project, we compare:

        - **iPhone users**
        - **Android users**

        The outcome variable is:

        - **Number of drives**
        """
    )

    with st.expander("Why are we doing a hypothesis test?"):
        st.write(
            """
            A simple average comparison can show that one group appears higher than another,
            but it does not tell us whether the difference is meaningful.

            A hypothesis test helps answer:

            > Is the difference large enough that we should treat it as statistically significant,
            or could it simply be due to random variation in the sample?
            """
        )

    with st.expander("What is the final output of this project?"):
        st.write(
            """
            The final output is a business-friendly conclusion for Waze leadership:

            - Whether device type is associated with different average driving activity.
            - Whether leadership should focus on device type as a possible engagement factor.
            - What other variables should be explored next.
            """
        )


# =========================================================
# PAGE 2: UNDERSTAND THE DATA
# =========================================================

elif page == "2. Understand the Data":

    st.markdown('<p class="section-header">📊 2. Understand the Data</p>', unsafe_allow_html=True)

    st.write(
        """
        Before running a statistical test, we first need to understand the dataset.
        This helps us check whether the required variables are available and whether
        the data looks reasonable.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            f"{df.shape[0]:,}",
            help="Each row represents a user record in the Waze dataset."
        )

    with col2:
        st.metric(
            "Columns",
            f"{df.shape[1]:,}",
            help="The number of variables available in the dataset."
        )

    with col3:
        st.metric(
            "Device Types",
            f"{test_df['device'].nunique()}",
            help="The number of device groups used in this analysis."
        )

    with col4:
        st.metric(
            "Missing Drives",
            f"{df['drives'].isna().sum():,}",
            help="Missing values in the drives column."
        )

    st.markdown("### Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
        <strong>What are we looking at?</strong><br>
        This preview shows the first few records in the dataset. It helps us confirm that
        the data loaded correctly and that the columns needed for the analysis are present.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Column Health Check")

    column_summary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Missing %": (df.isna().mean().values * 100).round(2)
    })

    st.dataframe(column_summary, use_container_width=True)

    st.markdown("### Descriptive Statistics")

    st.dataframe(df.describe(include="all"), use_container_width=True)

    with st.expander("Educational note: Why descriptive statistics matter"):
        st.write(
            """
            Descriptive statistics help us quickly understand the shape of the data.

            For this project, they help us answer questions like:

            - What is the average number of drives?
            - Are there users with very high or very low numbers of drives?
            - Is there a large spread in driving activity?
            - Do iPhone and Android users look different before testing?
            """
        )


# =========================================================
# PAGE 3: COMPARE DEVICES
# =========================================================

elif page == "3. Compare Devices":

    st.markdown('<p class="section-header">📱 3. Compare Devices</p>', unsafe_allow_html=True)

    st.write(
        """
        Now we compare the two device groups. This gives us an initial view before
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
    device_counts["Percentage"] = (
        device_counts["Number of Users"] / device_counts["Number of Users"].sum() * 100
    ).round(2)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(device_counts, use_container_width=True)

    with col2:
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.bar(device_counts["Device"], device_counts["Number of Users"])
        ax1.set_title("Number of Users by Device Type")
        ax1.set_xlabel("Device Type")
        ax1.set_ylabel("Number of Users")
        st.pyplot(fig1)

    st.markdown(
        """
        <div class="insight-box">
        <strong>Why this matters:</strong><br>
        Before comparing averages, we need to know how many users are in each group.
        If one group is much smaller, it may affect how confidently we interpret the result.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Average Drives by Device")

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

    rounded_summary = device_summary.copy()
    numeric_cols = [
        "average_drives",
        "median_drives",
        "standard_deviation",
        "minimum_drives",
        "maximum_drives"
    ]

    for col in numeric_cols:
        rounded_summary[col] = rounded_summary[col].round(2)

    st.dataframe(rounded_summary, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "iPhone Average Drives",
            f"{iphone_mean:,.2f}",
            help="The average number of drives for iPhone users."
        )

    with col2:
        st.metric(
            "Android Average Drives",
            f"{android_mean:,.2f}",
            help="The average number of drives for Android users."
        )

    with col3:
        st.metric(
            "Observed Difference",
            f"{difference:,.2f}",
            help="iPhone average drives minus Android average drives."
        )

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.bar(device_summary["device"], device_summary["average_drives"])
    ax2.set_title("Average Number of Drives by Device Type")
    ax2.set_xlabel("Device Type")
    ax2.set_ylabel("Average Number of Drives")
    st.pyplot(fig2)

    st.markdown(
        """
        <div class="warning-box">
        <strong>Important:</strong><br>
        A visible difference in averages does not automatically mean the difference is statistically significant.
        That is why we need the hypothesis test in the next section.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Distribution of Drives")

    max_display = int(test_df["drives"].quantile(0.95))

    selected_max = st.slider(
        "Adjust the maximum number of drives shown in the histogram:",
        min_value=1,
        max_value=max_display,
        value=max_display,
        step=1
    )

    visual_df = test_df[
        (test_df["drives"] >= 0) &
        (test_df["drives"] <= selected_max)
    ]

    iphone_visual = visual_df[visual_df["device"] == "iPhone"]["drives"]
    android_visual = visual_df[visual_df["device"] == "Android"]["drives"]

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.hist(iphone_visual, bins=40, alpha=0.6, label="iPhone")
    ax3.hist(android_visual, bins=40, alpha=0.6, label="Android")
    ax3.set_title(f"Distribution of Drives by Device Type: Up to {selected_max} Drives")
    ax3.set_xlabel("Number of Drives")
    ax3.set_ylabel("Number of Users")
    ax3.legend()
    st.pyplot(fig3)

    with st.expander("Educational note: How to read the histogram"):
        st.write(
            """
            The histogram shows how the number of drives is spread across users.

            Look for:

            - Whether most users are concentrated in lower or higher drive ranges.
            - Whether one device group has more users with very high drive counts.
            - Whether the two distributions look very different or mostly similar.

            Even when distributions look slightly different, the t-test helps us decide
            whether the average difference is statistically meaningful.
            """
        )


# =========================================================
# PAGE 4: LEARN THE HYPOTHESIS TEST
# =========================================================

elif page == "4. Learn the Hypothesis Test":

    st.markdown('<p class="section-header">🧪 4. Learn the Hypothesis Test</p>', unsafe_allow_html=True)

    st.write(
        """
        This section explains the statistical test used in the project.
        The goal is to make the method easy to understand before showing the final result.
        """
    )

    st.markdown("### Step 1: Define the Research Question")

    st.markdown(
        """
        <div class="insight-box">
        Do Waze users who open the app using an iPhone have the same average number of drives
        as Waze users who open the app using Android?
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Step 2: State the Hypotheses")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="neutral-box">
            <strong>Null Hypothesis H₀</strong><br><br>
            There is no difference in the average number of drives between iPhone users and Android users.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="neutral-box">
            <strong>Alternative Hypothesis H₁</strong><br><br>
            There is a difference in the average number of drives between iPhone users and Android users.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Step 3: Choose the Significance Level")

    st.markdown(
        f"""
        <div class="insight-box">
        The significance level for this test is <strong>{alpha:.0%}</strong>.
        This means we are using 5% as the threshold for deciding whether the result is statistically significant.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Step 4: Use the Correct Test")

    st.write(
        """
        We use a **two-sample t-test** because we are comparing the average number of drives
        between two independent groups:

        - iPhone users
        - Android users
        """
    )

    st.markdown(
        """
        <div class="warning-box">
        <strong>Why Welch's t-test?</strong><br>
        We use Welch's version of the t-test by setting <code>equal_var=False</code>.
        This is safer because it does not assume that iPhone and Android users have the same variance.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Show the Python code used for the test"):
        st.code(
            """
iphone = test_df[test_df["device"] == "iPhone"]["drives"].dropna()
android = test_df[test_df["device"] == "Android"]["drives"].dropna()

t_stat, p_value = stats.ttest_ind(
    a=iphone,
    b=android,
    equal_var=False
)
            """,
            language="python"
        )

    with st.expander("What is a p-value?"):
        st.write(
            """
            A p-value helps us decide whether the observed difference between two groups
            is surprising under the null hypothesis.

            In simple terms:

            - A small p-value suggests the observed difference is unlikely to be due to random chance.
            - A large p-value suggests the observed difference could reasonably happen by random chance.

            In this project, we compare the p-value to 0.05.
            """
        )

    with st.expander("What does 'fail to reject the null hypothesis' mean?"):
        st.write(
            """
            It does not mean the null hypothesis is definitely true.

            It means:

            > Based on this sample, we do not have enough statistical evidence to say
            that iPhone and Android users have different average numbers of drives.
            """
        )


# =========================================================
# PAGE 5: RESULTS AND INTERPRETATION
# =========================================================

elif page == "5. Results and Interpretation":

    st.markdown('<p class="section-header">📈 5. Results and Interpretation</p>', unsafe_allow_html=True)

    st.write(
        """
        This section presents the statistical test results and explains what they mean.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "T-statistic",
            f"{t_stat:,.4f}",
            help="Measures the difference between group means relative to the variation in the data."
        )

    with col2:
        st.metric(
            "P-value",
            f"{p_value:.4f}",
            help="Used to decide whether the result is statistically significant."
        )

    with col3:
        st.metric(
            "Alpha",
            f"{alpha:.0%}",
            help="The chosen significance level."
        )

    with col4:
        st.metric(
            "Effect Size",
            f"{effect_size:,.4f}",
            help="Cohen's d measures the practical size of the difference."
        )

    st.markdown("### Statistical Decision")

    if p_value < alpha:
        st.markdown(
            """
            <div class="success-box">
            <strong>Decision:</strong> Reject the null hypothesis.<br><br>
            The p-value is smaller than 0.05, meaning there is a statistically significant difference
            in average drives between iPhone and Android users.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="warning-box">
            <strong>Decision:</strong> Fail to reject the null hypothesis.<br><br>
            The p-value is greater than 0.05, meaning there is not enough evidence to conclude that
            iPhone and Android users have different average numbers of drives.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Plain English Interpretation")

    st.write(
        f"""
        The average number of drives for **iPhone users** is **{iphone_mean:,.2f}**.

        The average number of drives for **Android users** is **{android_mean:,.2f}**.

        The observed difference is **{difference:,.2f}** drives.

        The p-value is **{p_value:.4f}**.
        """
    )

    st.markdown(
        f"""
        <div class="insight-box">
        <strong>Effect size interpretation:</strong><br>
        Cohen's d is <strong>{effect_size:,.4f}</strong>, which suggests a <strong>{effect_interpretation.lower()}</strong>.
        This helps us understand whether the difference is practically meaningful, not only statistically significant.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Results Table")

    results_df = pd.DataFrame({
        "Metric": [
            "iPhone users",
            "Android users",
            "iPhone average drives",
            "Android average drives",
            "Observed difference",
            "T-statistic",
            "P-value",
            "Alpha",
            "Decision",
            "Effect size",
            "Effect size interpretation"
        ],
        "Value": [
            f"{len(iphone):,}",
            f"{len(android):,}",
            f"{iphone_mean:,.2f}",
            f"{android_mean:,.2f}",
            f"{difference:,.2f}",
            f"{t_stat:,.4f}",
            f"{p_value:.4f}",
            f"{alpha:.2f}",
            decision,
            f"{effect_size:,.4f}",
            effect_interpretation
        ]
    })

    st.dataframe(results_df, use_container_width=True)


# =========================================================
# PAGE 6: STAKEHOLDER SUMMARY
# =========================================================

elif page == "6. Stakeholder Summary":

    st.markdown('<p class="section-header">💼 6. Stakeholder Summary</p>', unsafe_allow_html=True)

    st.write(
        """
        This section translates the analysis into a leadership-friendly summary.
        """
    )

    st.markdown("### Executive Summary")

    if p_value < alpha:
        st.markdown(
            """
            <div class="success-box">
            The hypothesis test found a statistically significant difference in the average number
            of drives between iPhone and Android users.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="success-box">
            The hypothesis test did not find enough statistical evidence to conclude that iPhone
            and Android users have different average numbers of drives.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Key Business Insight")

    if p_value < alpha:
        st.write(
            """
            Device type may be related to driving activity. This means Waze may need to investigate
            whether user experience, platform performance, or engagement patterns differ between
            iPhone and Android users.
            """
        )
    else:
        st.write(
            """
            Device type does not appear to be a strong differentiator of average driving activity.
            Based on this test, Waze should be cautious about making device-specific decisions using
            this variable alone.
            """
        )

    st.markdown("### Recommendation")

    st.write(
        """
        Waze leadership should continue exploring other variables that may better explain
        user engagement and churn risk.
        """
    )

    recommended_variables = pd.DataFrame({
        "Variable to Explore": [
            "Sessions",
            "Driving days",
            "Total kilometers driven",
            "User tenure",
            "Churn status",
            "App activity level",
            "Platform-specific app performance"
        ],
        "Why It Matters": [
            "Shows how frequently users open or interact with the app.",
            "Indicates how often users actively drive with Waze.",
            "Captures the scale of driving activity.",
            "Helps identify whether newer or older users behave differently.",
            "Directly links behavior to retention risk.",
            "Helps separate casual users from highly engaged users.",
            "May reveal technical or experience differences between devices."
        ]
    })

    st.dataframe(recommended_variables, use_container_width=True)

    st.markdown("### Final Message to Leadership")

    st.markdown(
        """
        <div class="insight-box">
        Device type alone does not provide a complete explanation of user driving behavior.
        A stronger churn analysis should combine device information with behavioral variables,
        engagement patterns, and retention outcomes.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Limitation")

    st.markdown(
        """
        <div class="warning-box">
        This analysis compares only two groups using one outcome variable: number of drives.
        It does not explain why users behave the way they do, and it does not prove causation.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE 7: GLOSSARY
# =========================================================

elif page == "7. Glossary":

    st.markdown('<p class="section-header">📚 7. Glossary</p>', unsafe_allow_html=True)

    st.write(
        """
        This glossary explains the main statistical terms used in the dashboard.
        """
    )

    glossary = pd.DataFrame({
        "Term": [
            "Descriptive statistics",
            "Mean",
            "Median",
            "Standard deviation",
            "Hypothesis test",
            "Null hypothesis",
            "Alternative hypothesis",
            "P-value",
            "Significance level",
            "Two-sample t-test",
            "Welch's t-test",
            "Effect size",
            "Cohen's d"
        ],
        "Meaning": [
            "A set of summary measures used to understand a dataset.",
            "The average value.",
            "The middle value when data is ordered.",
            "A measure of how spread out values are.",
            "A statistical method used to test a claim about data.",
            "The default assumption that there is no meaningful difference or effect.",
            "The claim that there is a meaningful difference or effect.",
            "A value used to judge whether the observed result is statistically significant.",
            "The threshold used to decide whether to reject the null hypothesis.",
            "A test used to compare the means of two independent groups.",
            "A version of the t-test that does not assume equal variances between groups.",
            "A measure of how large or meaningful a difference is in practical terms.",
            "A common effect size measure for comparing two group means."
        ]
    })

    st.dataframe(glossary, use_container_width=True)

    st.markdown("###Simple Example")

    st.markdown(
        """
        <div class="neutral-box">
        Imagine two groups of students write a test: Group A and Group B.
        Group A gets an average of 75%, and Group B gets an average of 73%.
        <br><br>
        A hypothesis test helps us decide whether that 2% difference is meaningful,
        or whether it could have happened by random chance.
        <br><br>
        In this Waze project, we are doing the same thing, but instead of test marks,
        we are comparing average numbers of drives.
        </div>
        """,
        unsafe_allow_html=True
    )