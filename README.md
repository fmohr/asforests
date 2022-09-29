# asforests
Automatically Stopping Random Forests - a sklearn extension

## Why to use?
Never worry again about the number of trees you should use in the forest:
Standard Random Forest implementations like in sklearn require a hyperparameter for the number of trees. However, this number can be determined automatically online analyzing the performance curve of the forest.
This is what *asforests* do.

Implementations in asforests *inherit* from the forests in scikit-learn, so they can be used exactly in the same way, except that you do not (cannot) specify the number of trees.

## Usage
### Installation
```
pip install asforests
```

### Example Code
```
from asforests import RandomForestClassifier
rf = RandomForestClassifier()
```

To test this with some dataset:

```
import sklearn.model_selection
import sklearn.datasets
X, y = sklearn.datasets.load_iris(return_X_y = True)
sklearn.model_selection.cross_validate(rf, X, y)
```
