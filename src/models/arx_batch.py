import pandas as pd
import numpy as np
import collections

from river import time_series
from sklearn.linear_model import LinearRegression
from typing import Protocol, Any, Optional


class SKLEARNModelProtocol(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any:
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


class ARBatch(time_series.SNARIMAX):
    def __init__(
        self,
        p: int,
        regressor: SKLEARNModelProtocol = LinearRegression(),
        train_size: int = 1000,
        y_hat_min: float = float("-Inf"),
        y_hat_max: float = float("Inf"),
        y_hat_scaling: Optional[str] = None,
        y_hat_scaling_confidence: Optional[float] = None
    ):
        super().__init__(p, 0, 0, 1, 0, 0, 0, None)
        self.regressor = regressor
        self.train_size = train_size - p
        self.__train_queue = []
        self.__trained = False
        self.y_hat_min = y_hat_min
        self.y_hat_max = y_hat_max
        self.y_hat_scaling = y_hat_scaling
        self.y_hat_scaling_confidence = y_hat_scaling_confidence
    
    def _bound_prediction(self, y):
        if self.y_hat_scaling == "minmax":
            if abs(self.y_hat_max) != float("Inf") and abs(self.y_hat_min) != float("Inf"):
                diff = self.y_hat_max - self.y_hat_min
                lower_bound = self.y_hat_min - self.y_hat_scaling_confidence * diff
                upper_bound = self.y_hat_max + self.y_hat_scaling_confidence * diff
                y = max(lower_bound, y)
                y = min(upper_bound, y)
        else:
            y = max(self.y_hat_min, y)
            y = min(self.y_hat_max, y)
        return y

    def learn_one(self, y, x=None):
        if not self.__trained:
            if len(self.y_diff) >= self.p:
                x = self._add_lag_features(x=x, Y=list(self.y_diff), errors=[0 for _ in range(len(self.y_diff))])
                self.__train_queue.append((x, y))

                if len(self.__train_queue) >= self.train_size:
                    # batch train if enough data
                    X = pd.DataFrame([elem[0] for elem in self.__train_queue])
                    Y = pd.Series([elem[1] for elem in self.__train_queue])
                    self.regressor.fit(X, Y)
                    self.__trained = True
                    self.__train_queue = []

        self.y_diff.appendleft(y)

    def forecast(self, horizon, xs=None):
        if xs is None:
            xs = [{}] * horizon

        if len(xs) != horizon:
            raise ValueError("the length of xs should be equal to the specified horizon")

        y_diff = collections.deque(self.y_diff)
        forecasts = [None] * horizon

         # update max and min if min-max-scaling chosen
        if self.y_hat_scaling == "minmax":
            if len(y_diff) > 0:
                self.y_hat_min = min(min(y_diff), self.y_hat_min)
                self.y_hat_max = max(max(y_diff), self.y_hat_max)

        for t, x in enumerate(xs):
            x = self._add_lag_features(x=x, Y=y_diff, errors=[0 for _ in range(len(y_diff))])

            x = pd.DataFrame(x, index=[0])
            y_pred = self.regressor.predict(x)[0]
            y_pred = self._bound_prediction(y_pred)

            forecasts[t] = y_pred
            y_diff.appendleft(y_pred)

        return forecasts
