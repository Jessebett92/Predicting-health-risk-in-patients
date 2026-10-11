import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

def xgb_model_train(X_train, X_test, y_train, y_test):

    # Calculate scale_pos_weight as ratio of negative to positive samples in training data
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=123,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1)
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy of the XGBoost Model: {accuracy:.3f}")
    print(f"ROC AUC Score: {roc_auc:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model
