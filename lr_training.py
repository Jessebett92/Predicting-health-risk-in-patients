from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def lr_model_train(X_train, X_test, y_train, y_test):
    # Handle class imbalance with class_weight='balanced'
    model = LogisticRegression(max_iter=1000, random_state=123, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    # Probability estimates for ROC AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy of the Logistic Regression Model: {accuracy:.3f}")
    print(f"ROC AUC Score: {roc_auc:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model
