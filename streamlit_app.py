import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import sklearn

st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🧱",
    layout="wide"
)

# All other Streamlit code goes below this!
st.title("How Strong is Your Concrete?")
st.write("   ")
st.write("   ")
st.write("   ")


page = st.sidebar.selectbox("Select Page",["Introduction 📘","Visualization 📊", "Automated Report 📑","Prediction"])
st.image("concrete.jpeg")

df = pd.read_csv("concrete_data.csv")
## Step 02 - Load dataset
if page == "Introduction 📘":

    st.subheader("01 Introduction 📘")
    st.video("https://youtu.be/4PND-KnHF-I?si=mJUaNVsk2mOBPrBH")
    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))
    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ you have missing values")

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

elif page == "Visualization 📊":

    ## Step 03 - Data Viz
    st.subheader("02 Data Viz")

    # Get numeric columns only (concrete data is numeric)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Dropdowns for user selection
    col_x = st.selectbox("Select X-axis variable", numeric_cols, index=0)
    col_y = st.selectbox("Select Y-axis variable", numeric_cols, index=1)

    tab1, tab2, tab3 = st.tabs(["Scatter Plot 🔹", "Line Chart 📈", "Correlation Heatmap 🔥"])

    with tab1:
        st.subheader(f"Scatter Plot: {col_x} vs {col_y}")
        fig_scatter, ax_scatter = plt.subplots()
        sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax_scatter)
        ax_scatter.set_xlabel(col_x)
        ax_scatter.set_ylabel(col_y)
        st.pyplot(fig_scatter)

    with tab2:
        st.subheader(f"Line Chart: {col_y} over {col_x}")
        fig_line, ax_line = plt.subplots()
        df_sorted = df.sort_values(by=col_x)
        ax_line.plot(df_sorted[col_x], df_sorted[col_y], marker='o')
        ax_line.set_xlabel(col_x)
        ax_line.set_ylabel(col_y)
        st.pyplot(fig_line)

    with tab3:
        st.subheader("Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
        ax_corr.set_title("Correlation Matrix")
        st.pyplot(fig_corr)




elif page == "Automated Report 📑":
    st.subheader("03 Automated Report")

    detail_level = st.radio(
        "Select level of detail:",
        ["Basic Overview", "Standard EDA", "Advanced EDA"],
        index=0
    )

    if st.button("Generate Report"):
        with st.spinner("Generating report..."):

            # Always show summary stats and missing values
            st.markdown("### Summary Statistics 📊")
            st.dataframe(df.describe())

            st.markdown("### Missing Values")
            missing = df.isnull().sum()
            st.write(missing)
            if missing.sum() == 0:
                st.success("✅ No missing values found")
            else:
                st.warning("⚠️ Missing values detected")

            # If Standard or Advanced, add correlation + pairplot
            if detail_level in ["Standard EDA", "Advanced EDA"]:
                st.markdown("### Correlation Matrix 🔥")
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
                ax_corr.set_title("Correlation Matrix")
                st.pyplot(fig_corr)

                st.markdown("### Pairplot (sampled if large) 🔍")
                if df.shape[0] > 500:
                    df_sample = df.sample(500, random_state=42)
                    st.info("Pairplot shown for a 500-row sample (for speed).")
                else:
                    df_sample = df
                fig_pair = sns.pairplot(df_sample)
                st.pyplot(fig_pair.figure)

            # If Advanced, add additional analysis
            if detail_level == "Advanced EDA":
                st.markdown("### Constant Columns 🚩")
                constant_cols = [col for col in df.columns if df[col].nunique() == 1]
                if constant_cols:
                    st.warning(f"⚠️ Constant columns: {constant_cols}")
                else:
                    st.success("✅ No constant columns")

                st.markdown("### High Cardinality Columns 🚩")
                cardinality = {col: df[col].nunique() for col in df.columns}
                high_card_cols = [col for col, uniq in cardinality.items() if uniq > 50]
                if high_card_cols:
                    st.warning(f"⚠️ High cardinality columns: {high_card_cols}")
                else:
                    st.success("✅ No high cardinality columns")

                st.markdown("### Highly Correlated Pairs (> 0.85) 🚩")
                corr_matrix = df.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr = [
                    (col1, col2, upper.loc[col1, col2])
                    for col1 in upper.columns
                    for col2 in upper.index
                    if pd.notnull(upper.loc[col1, col2]) and upper.loc[col1, col2] > 0.85
                ]
                if high_corr:
                    for col1, col2, corr_val in high_corr:
                        st.warning(f"⚠️ {col1} and {col2} correlation: {corr_val:.2f}")
                else:
                    st.success("✅ No highly correlated pairs")

                st.markdown("### Outlier Detection (z-score > 3) 🚩")
                from scipy.stats import zscore
                z_scores = np.abs(zscore(df.select_dtypes(include=np.number)))
                outliers = (z_scores > 3).sum(axis=0)
                outlier_cols = {col: int(count) for col, count in zip(df.select_dtypes(include=np.number).columns, outliers) if count > 0}
                if outlier_cols:
                    st.warning(f"⚠️ Outliers detected:\n{outlier_cols}")
                else:
                    st.success("✅ No significant outliers detected")

            # Download summary
            csv = df.describe().to_csv(index=True).encode()
            st.download_button(
                label="📥 Download Summary Statistics CSV",
                data=csv,
                file_name="summary_statistics.csv",
                mime='text/csv'
            )








elif page == "Prediction":
    st.subheader("04 Prediction with Linear Regression")

    df2 = df.copy()

    # 1️ Let user choose features (X) and target (y)
    list_var = list(df2.columns)

    st.markdown("### Model Configuration")

    # Put X and Y selectors side-by-side
    col1, col2 = st.columns(2)

    with col1:
        target_selection = st.selectbox(
            "🎯 Select Target Variable (Y)",
            options=list_var,
            index=len(list_var) - 1
        )

    with col2:
        # Filter X options to exclude the selected target
        x_options = [col for col in list_var if col != target_selection]
        features_selection = st.multiselect(
            "📊 Select Features (X)",
            options=x_options,
            default=x_options
        )
        # 2️ Let user choose test size
    test_size = st.slider(
        "Select test set size (%)", 
        min_value=10, 
        max_value=50, 
        value=20, 
        step=5
    ) / 100.0

    # 3️ Show preview
    X = df2[features_selection]
    y = df2[target_selection]

    st.write("### Features (X) preview")
    st.dataframe(X.head())

    st.write("### Target (y) preview")
    st.dataframe(y.head())


    # 4️ Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    st.write(f"✅ **Train size:** {X_train.shape[0]} rows")
    st.write(f"✅ **Test size:** {X_test.shape[0]} rows")

    # 5️ Train model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)

 # 6️ Predict
    predictions = model.predict(X_test)


 # 7 Metrics choice
    selected_metrics = st.multiselect(
        "Select metrics to display",
        ["Mean Squared Error (MSE)","Root Mean Squared Error (RMSE)", "Mean Absolute Error (MAE)", "R² Score"],
        default=["Mean Absolute Error (MAE)"]
    )
    from sklearn import metrics
    if "Mean Squared Error (MSE)" in selected_metrics:
        mse = metrics.mean_squared_error(y_test, predictions)
        st.write(f"- **MSE**: {mse:,.2f}")
    if "Root Mean Squared Error (RMSE)" in selected_metrics:
        rmse = metrics.root_mean_squared_error(y_test,predictions)
        st.write(f"- **RMSE**: {rmse:,.2f}")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, predictions)
        st.write(f"- **MAE**: {mae:,.2f}")
    if "R² Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, predictions)
        st.write(f"- **R² Score**: {r2:,.3f}")

    # 8 Plot actual vs predicted
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
