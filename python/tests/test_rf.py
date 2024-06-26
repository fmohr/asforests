import sklearn.datasets
import unittest
import asforests

class ASRFTest(unittest.TestCase):

    def test_functionality(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        rf = asforests.RandomForestClassifier()
        rf.fit(X, y)

if __name__ == "main":
    print("OK")