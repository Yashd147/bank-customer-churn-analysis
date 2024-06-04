from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_models(models, X_test, y_test):
    lgr, rfc = models

    # Evaluate models
    lgr_y_pred = lgr.predict(X_test)
    rfc_y_pred = rfc.predict(X_test)

    lgr_accuracy = accuracy_score(y_test, lgr_y_pred)
    rfc_accuracy = accuracy_score(y_test, rfc_y_pred)

    # Print accuracy scores
    print("Logistic Regression Accuracy Score:", lgr_accuracy * 100, "%")
    print("Random Forest Accuracy Score:", rfc_accuracy * 100, "%")

    # Print classification report for Random Forest
    print("Classification Report (Random Forest):")
    print(classification_report(y_test, rfc_y_pred))

    # Plot confusion matrix for Random Forest
    cm_rfc = confusion_matrix(y_test, rfc_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_rfc, annot=True, fmt='d', cmap='Blues', square=True)
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.title('Confusion Matrix (Random Forest)')
    plt.show()

    # Plot ROC curve for both models
    y_probs_rfc = rfc.predict_proba(X_test)[:, 1]
    fpr_rfc, tpr_rfc, _ = roc_curve(y_test, y_probs_rfc)

    y_probs_lgr = lgr.predict_proba(X_test)[:, 1]
    fpr_lgr, tpr_lgr, _ = roc_curve(y_test, y_probs_lgr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rfc, tpr_rfc, label='Random Forest')
    plt.plot(fpr_lgr, tpr_lgr, label='Logistic Regression')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    return lgr_accuracy, rfc_accuracy
