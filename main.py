# conda env create -f enviornment.yml
# conda activate myenv

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix


from rf_training import rf_model_train
from dt_training import decision_tree_train
from lr_training import lr_model_train
from xgboost_training import xgb_model_train

from params import param_assign
from cross_vali import cross_val_with_smote

dataset = "healthcare_real_time_dataset.csv"

def main():

    def ytarget_binary_convert(y, positive_class='High'):
        return y.apply(lambda x: 1 if x == positive_class else 0)

    def exploratory_step(df):
        print(df.info())
        print()
        print(f"Total records: {df.shape[0]}")
        print(f"Number of patients aged 50 or older: {(df['Age'] >= 50).sum()}")
        print(f"Number of patients younger than 50: {(df['Age'] < 50).sum()}")
        print()
        print("CHECK FOR MISSING VALUES")
        print(df.isnull().sum())
        print()
        print("CHECK FOR DUPLICATE ROWS")
        print(df.duplicated().sum())

    def preprocess_data(df):
        # Fill missing values
        df['Chronic Disease History'] = df['Chronic Disease History'].fillna('Unknown')

        # Feature engineering
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young Adult', 'Middle Aged', 'Senior'])
        df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 32, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df['Is_Active'] = (df['Physical Activity (hours/week)'] >= 2.5).astype(int)

        def lifestyle_score(row):
            score = 0
            if row['Sleep Duration (hours/day)'] >= 7:
                score += 1
            if row['Alcohol Consumption (per week)'] < 7:
                score += 1
            if row['Physical Activity (hours/week)'] > 5:
                score += 1
            if row['Stress Level (1-10)'] <= 4:
                score += 1
            return score

        df['Lifestyle_Score'] = df.apply(lifestyle_score, axis=1)
        df['Stress_Sleep_Ratio'] = df['Stress Level (1-10)'] / df['Sleep Duration (hours/day)']

        y = df['Health Risk Level']
        X = df.drop('Health Risk Level', axis=1)

        X_encoded = pd.get_dummies(X, drop_first=True)
        return X_encoded, y

    def training_split(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123, stratify=y
        )
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

    def xgb_hyperparameter_tuning(X_train, y_train):
        param_dist = {
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'scale_pos_weight': [1, 2, 5, 10]
        }

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=123
        )

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=20,
            scoring='roc_auc',
            cv=5,
            verbose=2,
            n_jobs=-1,
            random_state=123
        )

        random_search.fit(X_train, y_train)

        print("Best parameters found:")
        print(random_search.best_params_)
        print(f"Best ROC AUC: {random_search.best_score_:.3f}")

        return random_search.best_estimator_

    # Main execution starts here
    rf_params, dt_params, lr_params, xgb_params = param_assign()

    df = pd.read_csv(dataset)
    exploratory_step(df)
    X, y = preprocess_data(df)
    # Convert target to binary
    y_binary = ytarget_binary_convert(y, positive_class='High')
    X_train, X_test, y_train, y_test = training_split(X, y_binary)

    # Run Tuning
    best_xgb_model = xgb_hyperparameter_tuning(X_train, y_train)
    # Evaluate tuned model on test set
    y_pred = best_xgb_model.predict(X_test)
    y_prob = best_xgb_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Tuned XGBoost Model Accuracy: {accuracy:.3f}")
    print(f"Tuned XGBoost Model ROC AUC: {roc_auc:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Train models on binary target
    rf_model_train(X_train, X_test, y_train, y_test)
    decision_tree_train(X_train, X_test, y_train, y_test)
    lr_model_train(X_train, X_test, y_train, y_test)

    # Cross-validation with SMOTE on binary target
    print("Random Forest CV with SMOTE:")
    cross_val_with_smote(RandomForestClassifier, X, y_binary, model_params=rf_params, n_splits=5)

    print("Decision Tree CV with SMOTE:")
    cross_val_with_smote(DecisionTreeClassifier, X, y_binary, model_params=dt_params, n_splits=5)

    print("Logistic Regression CV with SMOTE:")
    cross_val_with_smote(LogisticRegression, X, y_binary, model_params=lr_params, n_splits=5, use_scaler=True)

    print("XGBoost CV with SMOTE:")
    cross_val_with_smote(xgb.XGBClassifier, X, y_binary, model_params=xgb_params, n_splits=5)

    print(y_train.value_counts())

if __name__ == "__main__":
    main()