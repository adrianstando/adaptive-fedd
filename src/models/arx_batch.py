import pandas as pd
import numpy as np
import collections

from river import time_series
from sklearn.linear_model import LinearRegression
from typing import Protocol, Any


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
        train_size: int = 1000
    ):
        super().__init__(p, 0, 0, 1, 0, 0, 0, None)
        self.regressor = regressor
        self.train_size = train_size
        self.__train_queue = []
        self.__trained = False

    def learn_one(self, y, x=None):
        if not self.__trained:
            if len(self.y_diff) >= self.p:
                x = self._add_lag_features(x=x, Y=self.y_diff, errors=[0 for _ in range(len(self.y_diff))])
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

        for t, x in enumerate(xs):
            x = self._add_lag_features(x=x, Y=y_diff, errors=[0 for _ in range(len(y_diff))])

            x = pd.DataFrame(x, index=[0])
            y_pred = self.regressor.predict(x)[0]

            forecasts[t] = y_pred
            y_diff.appendleft(y_pred)

        return forecasts
