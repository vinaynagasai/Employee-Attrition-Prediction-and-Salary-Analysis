import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


np.random.seed(42)


# Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    """
    print("Loading and preprocessing data...")

    # Reading the CSV file
    try:
        df = pd.read_csv(file_path)
    except:
        print(f"Error reading file: {file_path}")
        return None

    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isnull().sum())


    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())


    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])


    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    if 'Extracurricular_Activities' in df.columns:
        df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'Yes': 1, 'No': 0})


    if df.isnull().sum().sum() > 0:
        print("\nRemaining missing values after preprocessing:")
        print(df.isnull().sum())
        print("\nFilling remaining missing values with median/mode...")
        df = df.fillna(df.median())

    return df


#  Classification
def train_attrition_models(df, target_col='Attrition'):
    """
    Train multiple classification models to predict attrition.
    """
    print("\n--- Part 1: Attrition Prediction ---")


    X = df.drop(columns=[target_col])

    if df[target_col].dtype == 'object':
        y = df[target_col].map({'Yes': 1, 'No': 0})
    else:
        y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    results = {}
    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)

        print(f"{name} - F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

        results[name] = {
            'model': model,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }

        if roc_auc > best_score:
            best_score = roc_auc
            best_model = name

    print(f"\nBest model: {best_model} with ROC AUC: {best_score:.4f}")

    if best_model in ['Decision Tree', 'Random Forest']:
        model = results[best_model]['model']
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        print("\nFeature Importance:")
        print(feature_importance.head(10))

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title(f'Top 10 Feature Importance - {best_model}')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

    return results, scaler, X.columns


# Data Augmentation
def simulate_future_salaries(df, salary_col='MonthlyIncome'):
    """
    Simulate future salaries based on current salary and performance.
    """
    print("\n--- Part 2: Simulating Future Salaries ---")

    df_augmented = df.copy()

    if 'PerformanceRating' in df_augmented.columns:
        performance_col = 'PerformanceRating'
    else:
        performance_col = 'JobSatisfaction'

    df_augmented['Performance'] = df_augmented[performance_col]

    df_augmented['Performance'] = df_augmented['Performance'].fillna(df_augmented['Performance'].median())

    if df_augmented['Performance'].max() > 4:
        df_augmented['Performance'] = pd.cut(
            df_augmented['Performance'],
            bins=[0, 1, 2, 3, float('inf')],
            labels=[1, 2, 3, 4]
        )

    df_augmented['Increment'] = df_augmented['Performance'].map({
        1: 1.03,
        2: 1.05,
        3: 1.08,
        4: 1.10
    }).fillna(1.05)

    df_augmented['FutureSalary'] = df_augmented[salary_col] * df_augmented['Increment']

    print(f"Added simulated future salaries with performance-based increments")
    print("Performance distribution:")
    print(df_augmented['Performance'].value_counts())
    print("\nIncrement statistics:")
    print(df_augmented['Increment'].describe())
    print("\nFuture Salary statistics:")
    print(df_augmented['FutureSalary'].describe())

    return df_augmented


# Regression
def train_salary_models(df_augmented, target_col='FutureSalary'):
    """
    Train regression models to predict future salaries.
    """
    print("\n--- Part 3: Salary Prediction ---")

    drop_cols = [target_col, 'Attrition', 'Increment', 'Performance']

    cols_to_drop = [col for col in drop_cols if col in df_augmented.columns]
    X = df_augmented.drop(columns=cols_to_drop)
    y = df_augmented[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg_models = {
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'Random Forest Regressor': RandomForestRegressor(random_state=42)
    }

    reg_results = {}
    best_reg_model = None
    best_r2 = -float('inf')

    for name, model in reg_models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        y_test_non_zero = np.where(y_test == 0, 1e-10, y_test)
        mape = mean_absolute_percentage_error(y_test_non_zero, y_pred)

        print(f"{name} - R² Score: {r2:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")

        reg_results[name] = {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mape': mape
        }

        if r2 > best_r2:
            best_r2 = r2
            best_reg_model = name

    print(f"\nBest regression model: {best_reg_model} with R² Score: {best_r2:.4f}")

    if 'Random Forest Regressor' in reg_results:
        model = reg_results['Random Forest Regressor']['model']
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        print("\nFeature Importance for Salary Prediction:")
        print(feature_importance.head(10))

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title('Top 10 Feature Importance - Random Forest Regressor for Salary')
        plt.tight_layout()
        plt.savefig('salary_feature_importance.png')
        plt.close()

    plt.figure(figsize=(10, 6))
    best_model = reg_results[best_reg_model]['model']
    y_pred = best_model.predict(X_test_scaled)

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Future Salary')
    plt.ylabel('Predicted Future Salary')
    plt.title(f'Actual vs Predicted Future Salary - {best_reg_model}')
    plt.tight_layout()
    plt.savefig('salary_predictions.png')
    plt.close()

    return reg_results, scaler, X.columns

def identify_likely_to_stay(df_augmented, attrition_results, attrition_scaler, feature_cols,
                            stay_threshold=0.6):
    """
    Identify employees who are likely to stay based on attrition predictions.
    """
    print(f"\n--- Part 4: Identifying Likely to Stay Employees (P_stay > {stay_threshold}) ---")

    df_with_probs = df_augmented.copy()

    best_model_name = max(attrition_results, key=lambda k: attrition_results[k]['roc_auc'])
    best_model = attrition_results[best_model_name]['model']

    X = df_with_probs[feature_cols]
    X_scaled = attrition_scaler.transform(X)

    probs = best_model.predict_proba(X_scaled)
    df_with_probs['P_leave'] = probs[:, 1]  #
    df_with_probs['P_stay'] = 1 - df_with_probs['P_leave']

    df_with_probs['LikelyToStay'] = df_with_probs['P_stay'] >= stay_threshold

    likely_to_stay_count = df_with_probs['LikelyToStay'].sum()
    likely_to_leave_count = len(df_with_probs) - likely_to_stay_count

    print(f"Employees likely to stay: {likely_to_stay_count} ({likely_to_stay_count / len(df_with_probs) * 100:.2f}%)")
    print(
        f"Employees likely to leave: {likely_to_leave_count} ({likely_to_leave_count / len(df_with_probs) * 100:.2f}%)")

    plt.figure(figsize=(10, 6))
    sns.histplot(df_with_probs['P_stay'], bins=20)
    plt.axvline(x=stay_threshold, color='r', linestyle='--')
    plt.title('Distribution of Probability to Stay')
    plt.xlabel('Probability of Staying')
    plt.ylabel('Count')
    plt.savefig('stay_probability_distribution.png')
    plt.close()

    return df_with_probs

def estimate_salary_loss(df_with_probs):
    """
    Estimate the expected salary loss due to potential attrition.
    """
    print("\n--- Part 5: Estimating Expected Salary Loss ---")

    df_with_probs['ExpectedLoss'] = df_with_probs['P_leave'] * df_with_probs['FutureSalary']

    total_expected_loss = df_with_probs['ExpectedLoss'].sum()

    print(f"Total expected salary loss: {total_expected_loss:.2f}")

    print("\nExpected Loss Statistics:")
    print(df_with_probs['ExpectedLoss'].describe())

    top_loss = df_with_probs.sort_values('ExpectedLoss', ascending=False).head(10)
    print("\nTop 10 employees with highest expected loss:")
    print(top_loss[['FutureSalary', 'P_leave', 'ExpectedLoss']])

    plt.figure(figsize=(10, 6))
    sns.histplot(df_with_probs['ExpectedLoss'], bins=20)
    plt.title('Distribution of Expected Salary Loss')
    plt.xlabel('Expected Loss')
    plt.ylabel('Count')
    plt.savefig('expected_loss_distribution.png')
    plt.close()

    df_with_probs['LeaveRiskBin'] = pd.cut(
        df_with_probs['P_leave'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )

    risk_group_loss = df_with_probs.groupby('LeaveRiskBin')['ExpectedLoss'].sum()

    plt.figure(figsize=(10, 6))
    risk_group_loss.plot(kind='bar')
    plt.title('Total Expected Salary Loss by Attrition Risk Group')
    plt.xlabel('Attrition Risk')
    plt.ylabel('Total Expected Loss')
    plt.tight_layout()
    plt.savefig('loss_by_risk_group.png')
    plt.close()

    return df_with_probs

def main():
    """
    Main execution function that runs the end-to-end pipeline.
    """
    print("===== Employee Attrition and Salary Analysis =====")

    file_path = 'E:/ml/employee-attrition.csv'

    try:
        df = load_and_preprocess_data(file_path)
        if df is None:
            print("Failed to load the data. Please check the file path and format.")
            return
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please make sure the file exists at the specified path.")
        return

    if df is None:
        print("Error loading data. Exiting...")
        return

    attrition_results, attrition_scaler, feature_cols = train_attrition_models(df)

    df_augmented = simulate_future_salaries(df)

    # Part 3: Salary Prediction
    salary_results, salary_scaler, salary_feature_cols = train_salary_models(df_augmented)

    df_with_probs = identify_likely_to_stay(df_augmented, attrition_results, attrition_scaler, feature_cols)

    final_df = estimate_salary_loss(df_with_probs)

    final_df.to_csv('employee_attrition_analysis_results.csv', index=False)
    print("\nAnalysis complete! Results saved to 'employee_attrition_analysis_results.csv'")


    print("\n===== Final Summary =====")
    print(f"Total employees analyzed: {len(final_df)}")


    print(f"Employees likely to leave: {len(final_df[final_df['P_leave'] > 0.5])}")
    print(f"Total expected salary loss: {final_df['ExpectedLoss'].sum():.2f}")

    high_risk_employees = final_df[final_df['P_leave'] > 0.7].sort_values('ExpectedLoss', ascending=False)

    print(f"\nHigh risk employees (P_leave > 0.7): {len(high_risk_employees)}")
    if len(high_risk_employees) > 0:
        print("\nTop 5 high-risk employees with highest expected loss:")
        print(high_risk_employees[['P_leave', 'FutureSalary', 'ExpectedLoss']].head(5))

    if 'JobRole' in final_df.columns:
        print("\nAttrition risk by job role:")
        role_risk = final_df.groupby('JobRole')['P_leave'].mean().sort_values(ascending=False)
        print(role_risk)

    if 'Department' in final_df.columns:
        print("\nAttrition risk by department:")
        dept_risk = final_df.groupby('Department')['P_leave'].mean().sort_values(ascending=False)
        print(dept_risk)

    return final_df

if __name__ == "__main__":
    main()

