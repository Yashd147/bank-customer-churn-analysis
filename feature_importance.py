from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def feature_importance_analysis(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    feature_importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances, y=X_train.columns, palette='viridis')
    plt.xlabel('Feature Importance Score', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title('Feature Importance Scores', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print("Feature Importance Values:")
    print(feature_importance_df.sort_values(by='Importance', ascending=False))
