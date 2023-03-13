from typing import Any

import numpy as np


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        pass

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        # we expect to get two valid on the dict x0, and x1
        # PassengerId,Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Cabin,Embarked
        return [[body.get("PassengerId", None), body.get("Pclass", None), body.get("Sex", None), body.get("Age", None),
        body.get("SibSp", None), body.get("Parch", None), body.get("Fare", None), body.get("Cabin", None), body.get("Embarked", None)], ]

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        return dict(Survived=data.tolist() if isinstance(data, np.ndarray) else data)
