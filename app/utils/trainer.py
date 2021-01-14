from sklearn.linear_model import LogisticRegression

class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return LogisticRegression(C=0.01, penalty='l1', solver='liblinear').fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)