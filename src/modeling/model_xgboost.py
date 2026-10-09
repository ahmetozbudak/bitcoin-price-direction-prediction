from xgboost import XGBClassifier


def run(X_train, X_test, y_train, y_test):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred
