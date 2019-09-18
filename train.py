from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

class Model():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = LogisticRegression(solver='liblinear')

    def train(self, X, y):    
        self.model.fit(X, y.ravel())
    
    def test_accuracy(self, X, y):
        predicted_classes = self.model.predict(X)
        accuracy = accuracy_score(y.flatten(), predicted_classes)
        return accuracy
