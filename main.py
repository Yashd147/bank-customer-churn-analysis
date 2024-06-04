import data_preprocessing
import model_training
import model_evaluation
import feature_importance

# Load and preprocess the data
data_encoded = data_preprocessing.preprocess_data('churn.csv')

# Split the data into features and target variable
X = data_encoded.drop('Exited', axis=1)
y = data_encoded['Exited']

# Train the models
models = model_training.train_models(X, y)

# Evaluate the models
lgr_accuracy, rfc_accuracy = evaluation.evaluate_models(models, X_test, y_test)

# Perform feature importance analysis
feature_importance.feature_importance_analysis(X_train, y_train)
