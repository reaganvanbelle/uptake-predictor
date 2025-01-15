import streamlit as st
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

def calculate_vif(df, features):
    # Create a dataframe with only the numeric features
    X = df[features].copy()
    
    # Drop rows with any NaN values
    X = X.dropna()
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif_data

# Function to identify numeric columns with too many NaN values
def get_usable_numeric_columns(df, threshold=0.5):
    numeric_cols = []
    skipped_cols = []
    
    for col in df.columns:
        # Convert to numeric if possible
        if col not in ['Name', 'Status', 'DateOfFirstLight', 'DepositStartDate']:  # Skip these columns
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Check if column has less than threshold% NaN values
                nan_ratio = df[col].isna().mean()
                if nan_ratio < threshold:
                    numeric_cols.append(col)
                else:
                    skipped_cols.append(f"{col} ({nan_ratio:.1%} NaN)")
            except:
                skipped_cols.append(f"{col} (non-numeric)")
                
    return numeric_cols, skipped_cols

# Upload file
st.title("Uptake Model Analysis")
data_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
if data_file:
    df = pd.read_excel(data_file)
    st.success("File Uploaded Successfully!")
else:
    st.warning("Please upload an Excel file.")
    st.stop()

# Check if "Uptake" column exists
if "Uptake" not in df.columns:
    st.error("The dataset must contain an 'Uptake' column.")
    st.stop()

# Get usable numeric columns
numeric_features, skipped_columns = get_usable_numeric_columns(df)

# Remove target from features if it's there
if "Uptake" in numeric_features:
    numeric_features.remove("Uptake")

# Display skipped columns in sidebar
if skipped_columns:
    with st.sidebar.expander("Columns excluded from analysis"):
        st.write("\n".join(skipped_columns))

# Tabs
tabs = st.tabs(["Correlation and VIF Analysis", "Linear Regression", "Random Forest"])

# Tab 1: Correlation and VIF Analysis
with tabs[0]:
    st.header("Correlation and VIF Analysis")
    
    numeric_df = df[numeric_features].copy()

    # Calculate R^2 for each feature against the target
    correlations = []
    for feature in numeric_features:
        X = numeric_df[[feature]].dropna()
        y = df.loc[X.index, "Uptake"].dropna()
        # Get common indices where both X and y have values
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        if len(X) > 0:
            model = LinearRegression().fit(X, y)
            correlations.append((feature, model.score(X, y)))

    correlation_df = pd.DataFrame(correlations, columns=["feature", "R^2"])
    
    # Ensure proper sorting of correlation DataFrame
    correlation_df["R^2"] = pd.to_numeric(correlation_df["R^2"])  # Convert to numeric if not already
    correlation_df = correlation_df.sort_values(by="R^2", ascending=False)

    # Calculate VIF
    vif_df = calculate_vif(df, numeric_features)

    # Merge DataFrames
    analysis_df = pd.merge(correlation_df, vif_df, on="feature", how="outer")

    # Add most correlated feature
    most_correlated = []
    for feature in numeric_features:
        correlations = numeric_df.corrwith(numeric_df[feature]).drop(feature).sort_values(ascending=False)
        most_correlated.append(correlations.idxmax() if not correlations.empty else None)

    analysis_df["Most Correlated Feature"] = most_correlated
    
    # Sort final DataFrame by R^2
    analysis_df = analysis_df.sort_values(by="R^2", ascending=False)
    st.dataframe(analysis_df)

# Tab 2: Linear Regression
with tabs[1]:
    st.header("Linear Regression")

    selected_features = st.multiselect("Select Features", numeric_features, default=numeric_features)

    if selected_features:
        # Create feature matrix and target vector
        X = df[selected_features].copy()
        y = df["Uptake"].copy()
        
        # Get indices where both X and y have non-NaN values
        valid_rows = X.dropna().index.intersection(y.dropna().index)
        
        # Filter X and y to only include valid rows
        X = X.loc[valid_rows]
        y = y.loc[valid_rows]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(selected_features) - 1)
        mse = mean_squared_error(y_test, y_pred)

        st.write(f"R^2: {r2:.3f}")
        st.write(f"Adjusted R^2: {adjusted_r2:.3f}")
        st.write(f"MSE: {mse:.3f}")
        st.write(f"Number of samples used: {len(valid_rows)} out of {len(df)}")

        # Feature importance
        feature_importance = pd.DataFrame({
            "Feature": selected_features,
            "Importance": model.coef_
        })
        feature_importance["Absolute Importance"] = feature_importance["Importance"].abs()
        feature_importance = feature_importance.sort_values(by="Absolute Importance", ascending=True)

        plt.figure(figsize=(8, 6))
        plt.barh(feature_importance["Feature"], feature_importance["Absolute Importance"])
        for index, value in enumerate(feature_importance["Importance"]):
            plt.text(feature_importance["Absolute Importance"].iloc[index], index, f"{value:.3f}", va='center')
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance (Linear Regression)")
        st.pyplot(plt)

        # Predicted vs Actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, label="Predictions")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="y = x")
        plt.xlabel("Actual Uptake")
        plt.ylabel("Predicted Uptake")
        plt.title("Predicted vs Actual")
        plt.legend()
        st.pyplot(plt)

        # Worst predicted values
        residuals = abs(y_test - y_pred)
        worst_predictions = pd.DataFrame({
            "Name": df.loc[X_test.index, "Name"],
            "Actual": y_test.round(3),
            "Predicted": y_pred.round(3),
            "Residual": residuals.round(3)
        }).nlargest(10, "Residual")
        st.table(worst_predictions)

        # Search box for specific section
        section_name = st.text_input("Enter Section Name:")
        if section_name:
            if section_name in df["Name"].values:
                idx = df[df["Name"] == section_name].index.intersection(X_test.index)
                if not idx.empty:
                    actual = y_test.loc[idx].values[0]
                    predicted = y_pred[idx][0]
                    residual = abs(actual - predicted)
                    st.write(f"Actual: {actual:.3f}")
                    st.write(f"Predicted: {predicted:.3f}")
                    st.write(f"Residual: {residual:.3f}")
                else:
                    st.write("Section not in test set.")
            else:
                st.write("Section name not found.")

# Tab 3: Random Forest
with tabs[2]:
    st.header("Random Forest")

    selected_features_rf = st.multiselect("Select Features", numeric_features, default=numeric_features, key="rf")

    if selected_features_rf:
        # Create feature matrix and target vector
        X = df[selected_features_rf].copy()
        y = df["Uptake"].copy()
        
        # Get indices where both X and y have non-NaN values
        valid_rows = X.dropna().index.intersection(y.dropna().index)
        
        # Filter X and y to only include valid rows
        X = X.loc[valid_rows]
        y = y.loc[valid_rows]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(selected_features_rf) - 1)
        mse = mean_squared_error(y_test, y_pred)

        st.write(f"R^2: {r2:.3f}")
        st.write(f"Adjusted R^2: {adjusted_r2:.3f}")
        st.write(f"MSE: {mse:.3f}")
        st.write(f"Number of samples used: {len(valid_rows)} out of {len(df)}")

        # Feature importance
        feature_importance = pd.DataFrame({
            "Feature": selected_features_rf,
            "Importance": model.feature_importances_
        })
        feature_importance["Absolute Importance"] = feature_importance["Importance"].abs()
        feature_importance = feature_importance.sort_values(by="Absolute Importance", ascending=True)

        plt.figure(figsize=(8, 6))
        plt.barh(feature_importance["Feature"], feature_importance["Absolute Importance"])
        for index, value in enumerate(feature_importance["Importance"]):
            plt.text(feature_importance["Absolute Importance"].iloc[index], index, f"{value:.3f}", va='center')
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance (Random Forest)")
        st.pyplot(plt)

        # Predicted vs Actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, label="Predictions")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="y = x")
        plt.xlabel("Actual Uptake")
        plt.ylabel("Predicted Uptake")
        plt.title("Predicted vs Actual")
        plt.legend()
        st.pyplot(plt)

        # Worst predicted values
        residuals = abs(y_test - y_pred)
        worst_predictions = pd.DataFrame({
            "Name": df.loc[X_test.index, "Name"],
            "Actual": y_test.round(3),
            "Predicted": y_pred.round(3),
            "Residual": residuals.round(3)
        }).nlargest(10, "Residual")
        st.table(worst_predictions)

        # Cross-validation section
        st.subheader("Cross-Validation Analysis")
        n_splits = st.slider("Number of Cross-Validation Folds", min_value=2, max_value=10, value=5)
        
        # Initialize KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)
        
        # Store feature importance for each fold
        cv_feature_importance = []
        
        # Perform cross-validation
        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
            y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
            
            # Train model
            model_cv = RandomForestRegressor(random_state=42)
            model_cv.fit(X_train_cv, y_train_cv)
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'Feature': selected_features_rf,
                'Importance': model_cv.feature_importances_
            })
            importance_df = importance_df.sort_values('Importance', ascending=False).head(5)
            importance_df['Fold'] = fold + 1
            cv_feature_importance.append(importance_df)
        
        # Combine all folds' feature importance
        all_folds_importance = pd.concat(cv_feature_importance)
        
        # Display results in a formatted table
        st.write("Top 5 Features by Importance in Each Fold:")
        pivot_table = all_folds_importance.pivot(index='Feature', columns='Fold', values='Importance')
        # Sort by the first fold's values in descending order
        pivot_table = pivot_table.sort_values(by=1, ascending=False)
        pivot_table = pivot_table.round(3)
        st.dataframe(pivot_table)
