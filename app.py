import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Analysis", layout="wide")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ----------------------------------
# Page Config
# ----------------------------------

# ----------------------------------
# Load Data (Cached)
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("clean_data.csv")

df = load_data()

# ----------------------------------
# Sidebar Navigation
# ----------------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select Section",
    ["Data Overview", "EDA", "ML Prediction"]
)

# ==================================
# DATA OVERVIEW
# ==================================
if section == "Data Overview":
    st.title("📊 Data Overview")

    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Min / Max (numeric columns)")
    try:
        min_max = df.select_dtypes(include=np.number).agg(['min', 'max']).T
        st.dataframe(min_max)
    except Exception:
        st.info("Could not compute min/max for numeric columns.")

# ==================================
# EDA
# ==================================
elif section == "EDA":
    st.title("📈 Exploratory Data Analysis")


    tabs = st.tabs(["Univariate", "Bivariate", "Correlation"])

    # -----------------------------
    # Univariate — notebook plots
    # -----------------------------
    with tabs[0]:
        st.subheader("Univariate")

        # Top 10 Person Income (side-by-side)
        top10 = df.sort_values(by="person_income", ascending=False).head(10)
        top10_emp = df.sort_values(by="person_emp_length", ascending=False).head(10)

        col_a, col_b = st.columns(2)

        fig, ax = plt.subplots()
        ax.bar(top10.index.astype(str), top10["person_income"])
        ax.set_title("Top 10 Person Income")
        ax.set_xlabel("Person ID")
        ax.set_ylabel("Person Income")
        plt.xticks(rotation=45)
        with col_a:
            st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.bar(top10_emp.index.astype(str), top10_emp["person_emp_length"])
        ax2.set_title("Top 10 Person Employment Length")
        ax2.set_xlabel("Person ID")
        ax2.set_ylabel("Person Employment Length")
        plt.xticks(rotation=45)
        with col_b:
            st.pyplot(fig2)

        # Categorical countplots (display two-per-row)
        cat_cols = [
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file",
        ]
        cols_cat = st.columns(2)
        for i, col in enumerate(cat_cols):
            if col in df.columns:
                fig, ax = plt.subplots()
                order = df[col].value_counts().index
                sns.countplot(x=col, data=df, order=order, ax=ax)
                ax.set_title(f"Count Plot of {col}")
                plt.xticks(rotation=45)
                with cols_cat[i % 2]:
                    st.pyplot(fig) 

        # Loan amount distribution + outlier boxplot (side-by-side)
        if "loan_amnt" in df.columns:
            col1, col2 = st.columns(2)

            fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
            sns.histplot(df["loan_amnt"], bins=30, kde=True, ax=ax_hist)
            ax_hist.set_title("Distribution of Loan Amount")
            with col1:
                st.pyplot(fig_hist)

            fig_box, ax_box = plt.subplots(figsize=(8, 3))
            sns.boxplot(x=df["loan_amnt"], ax=ax_box)
            ax_box.set_title("Loan Amount Outliers")
            with col2:
                st.pyplot(fig_box) 

    # -----------------------------
    # Bivariate — notebook plots
    # -----------------------------
    with tabs[1]:
        st.subheader("Bivariate")

        # Default rate by loan grade
        if "loan_grade" in df.columns and "loan_status" in df.columns:
            grade_default = df.groupby("loan_grade")["loan_status"].mean().sort_values()
            fig, ax = plt.subplots()
            grade_default.plot(kind="bar", ax=ax)
            ax.set_title("Default Rate by Loan Grade")
            ax.set_ylabel("Default Rate")
            st.pyplot(fig)

        # Boxplots used in the notebook (two-per-row)
        box_plots = [
            ("loan_percent_income", "Loan Percent Income vs Loan Status"),
            ("loan_int_rate", "Interest Rate vs Loan Status"),
            ("person_emp_length", "Employment Length vs Loan Status"),
            ("person_income", "Income vs Loan Status (Log Scale)"),
        ]
        cols_box = st.columns(2)
        for i, (col, title) in enumerate(box_plots):
            if col in df.columns:
                fig, ax = plt.subplots()
                sns.boxplot(x="loan_status", y=col, data=df, ax=ax)
                if col == "person_income":
                    ax.set_yscale("log")
                ax.set_title(title)
                with cols_box[i % 2]:
                    st.pyplot(fig)

        # Countplots with hue by loan_status (two-per-row)
        hue_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
        cols_hue = st.columns(2)
        for i, col in enumerate(hue_cols):
            if col in df.columns and "loan_status" in df.columns:
                fig, ax = plt.subplots()
                sns.countplot(x=col, hue="loan_status", data=df, ax=ax)
                ax.set_title(f"{col} vs Loan Status")
                plt.xticks(rotation=30)
                with cols_hue[i % 2]:
                    st.pyplot(fig)

    # -----------------------------
    # Correlation (notebook heatmap)
    # -----------------------------
    with tabs[2]:
        numeric_df = df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(22, 8))
        sns.heatmap(numeric_df.corr(), annot=True, linewidths=0.5, center=0, ax=ax)
        st.pyplot(fig)

# ==================================
# ML PREDICTION
# ==================================
elif section == "ML Prediction":
    st.title("🤖 Loan Prediction")

    # ----------------------------------
    # Encode Data (Cached)
    # ----------------------------------
    @st.cache_data
    def encode_data(df):
        df_encoded = pd.get_dummies(df, drop_first=True)
        X = df_encoded.drop("loan_status", axis=1)
        y = df_encoded["loan_status"]
        return X, y

    X, y = encode_data(df)

    # ----------------------------------
    # Train-Test Split
    # ----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # ----------------------------------
    # Scaling (Cached)
    # ----------------------------------
    @st.cache_resource
    def scale_data(X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return scaler, X_train_scaled, X_test_scaled

    scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # ----------------------------------
    # Train All Models (Cached)
    # ----------------------------------
    @st.cache_resource
    def train_models(X_train, y_train, X_test, y_test):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            ),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC()
        }

        accuracies = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies[name] = accuracy_score(y_test, y_pred)

        return models, accuracies

    models, accuracies = train_models(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    # ----------------------------------
    # Best Model (Random Forest)
    # ----------------------------------
    best_model = models["Random Forest"]

    # ----------------------------------
    # Risk Explanation Logic
    # ----------------------------------
    def explain_failure(raw_input, df):
        reasons = []

        if raw_input["person_income"].values[0] < df["person_income"].mean():
            reasons.append("Low income compared to average applicants")

        if raw_input["loan_amnt"].values[0] > df["loan_amnt"].mean():
            reasons.append("High loan amount requested")

        if raw_input["loan_int_rate"].values[0] > df["loan_int_rate"].mean():
            reasons.append("High interest rate")

        if raw_input["cb_person_default_on_file"].values[0] == "Y":
            reasons.append("Previous default history")

        if not reasons:
            reasons.append("Combination of multiple moderate risk factors")

        return reasons

    def validate_user_input_against_dataset(raw_input: pd.DataFrame, df: pd.DataFrame):
        """Return list of issues where user input is outside dataset min/max.
        Each issue is a dict: {column, value, min, max, type ('below_min'|'above_max')}
        """
        issues = []
        try:
            for col in raw_input.columns:
                if col not in df.columns:
                    continue
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    try:
                        v = raw_input[col].values[0]
                        if pd.isna(v):
                            continue
                        vf = float(v)
                        dmin = float(df[col].min())
                        dmax = float(df[col].max())
                        if vf < dmin:
                            issues.append({"column": col, "value": vf, "min": dmin, "max": dmax, "type": "below_min"})
                        elif vf > dmax:
                            issues.append({"column": col, "value": vf, "min": dmin, "max": dmax, "type": "above_max"})
                    except Exception:
                        # non-convertible values are ignored here
                        continue
        except Exception:
            return []
        return issues

    # ----------------------------------
    # User Input (side-by-side)
    # ----------------------------------
    st.subheader("🔮 Credit Risk Prediction")

    raw_user_data = {}
    # include all dataset fields (user provides values for every feature)
    input_cols = list(df.drop("loan_status", axis=1).columns)

    st.info("Note: all fields are used. Any numeric input below the dataset's historical minimum will automatically be rejected.")

    # Choose how many input columns per row (set to 2 for side-by-side)
    n_cols = 2
    cols = st.columns(n_cols)

    for i, col in enumerate(input_cols):
        with cols[i % n_cols]:
            if df[col].dtype == "object":
                raw_user_data[col] = st.selectbox(col, df[col].unique())
            else:
                # provide a sensible default (median) and keep numeric type
                if pd.api.types.is_integer_dtype(df[col].dtype):
                    default_val = int(df[col].median()) if not np.isnan(df[col].median()) else 0
                    raw_user_data[col] = st.number_input(col, min_value=0, value=default_val)
                else:
                    default_val = float(df[col].median()) if not np.isnan(df[col].median()) else 0.0
                    raw_user_data[col] = st.number_input(col, min_value=0.0, value=default_val)

    raw_input_df = pd.DataFrame([raw_user_data])

    encoded_input = pd.get_dummies(raw_input_df)
    encoded_input = encoded_input.reindex(columns=X.columns, fill_value=0)
    scaled_input = scaler.transform(encoded_input)

    # ----------------------------------
    # Prediction Output
    # ----------------------------------
    if st.button("Predict Loan Status"):
        with st.spinner("Evaluating credit risk..."):
            # check input values against dataset ranges first
            validation_issues = validate_user_input_against_dataset(raw_input_df, df)
            below_min = [i for i in validation_issues if i["type"] == "below_min"]
            above_max = [i for i in validation_issues if i["type"] == "above_max"]

            range_override_note = None
            if below_min:
                # override to Rejected when any input is below dataset minimum
                prediction = 1
                cols = ", ".join([f"{i['column']} (value={i['value']}, min={i['min']})" for i in below_min])
                range_override_note = f"Input outside dataset range — below minimum for: {cols}. Decision overridden to Rejected."
            else:
                # normal model prediction
                prediction = best_model.predict(scaled_input)[0]

            # try to get class probabilities (useful even when overridden)
            prob_approved = prob_rejected = None
            try:
                proba = best_model.predict_proba(scaled_input)[0]
                classes = list(best_model.classes_)
                if 0 in classes:
                    prob_approved = proba[classes.index(0)]
                if 1 in classes:
                    prob_rejected = proba[classes.index(1)]
            except Exception:
                pass

            # keep existing low-income percentile override as extra check
            percentile_override_note = None
            try:
                income_val = float(raw_input_df.get("person_income", pd.Series([np.nan])).values[0])
                low_income_cutoff = df["person_income"].quantile(0.05)
                if income_val < low_income_cutoff:
                    prediction = 1
                    percentile_override_note = f"Decision overridden: income ({income_val}) below 5th percentile ({low_income_cutoff:.0f})."
            except Exception:
                pass

        # Display final decision and messages
        if prediction == 0:
            if prob_approved is not None:
                st.success(f"✅ Loan Approved — confidence: {prob_approved*100:.1f}%")
            else:
                st.success("✅ Loan Approved (Low Credit Risk)")
            if range_override_note:
                st.warning(range_override_note)
            if percentile_override_note:
                st.warning(percentile_override_note)
            st.info("The applicant is unlikely to default.")
        else:
            if prob_rejected is not None:
                st.error(f"❌ Loan Rejected — confidence: {prob_rejected*100:.1f}%")
            else:
                st.error("❌ Loan Rejected (High Credit Risk)")
            if range_override_note:
                st.warning(range_override_note)
            if percentile_override_note:
                st.warning(percentile_override_note)
            if above_max:
                st.warning("Note: some inputs exceed dataset maxima — verify input values.")
                for i in above_max:
                    st.write(f"- {i['column']}: value={i['value']} max={i['max']}")
            st.subheader("📉 Possible Reasons")
            for r in explain_failure(raw_input_df, df):
                st.warning(f"⚠️ {r}")
