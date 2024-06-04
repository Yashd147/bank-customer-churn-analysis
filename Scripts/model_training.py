from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_models(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    lgr = LogisticRegression()
    lgr.fit(X_train, y_train)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    return lgr, rfc, X_test, y_test
