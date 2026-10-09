from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def run(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    return y_pred
