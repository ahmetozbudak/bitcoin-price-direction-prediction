from sklearn.ensemble import RandomForestClassifier


def run(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred
