from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def run(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        random_state=42,
        max_iter=5000,
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    return y_pred
