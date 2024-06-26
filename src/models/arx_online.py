import collections
from river import time_series
from river import base


class AROnline(time_series.SNARIMAX):
    def __init__(
        self,
        p: int,
        regressor: base.Regressor | None = None,
        y_hat_min: float = 0,
        y_hat_max: float = 100
    ):
        super().__init__(p, 0, 0, 1, 0, 0, 0, regressor)
        self.y_hat_min = y_hat_min
        self.y_hat_max = y_hat_max
    
    def _bound_prediction(self, y):
        y = max(self.y_hat_min, y)
        y = min(self.y_hat_max, y)
        return y
    
    def forecast(self, horizon, xs=None):
        if xs is None:
            xs = [{}] * horizon

        if len(xs) != horizon:
            raise ValueError("the length of xs should be equal to the specified horizon")

        y_hist = collections.deque(self.y_hist)
        y_diff = collections.deque(self.y_diff)
        errors = collections.deque(self.errors)
        forecasts = [None] * horizon

        for t, x in enumerate(xs):
            x = self._add_lag_features(x=x, Y=y_diff, errors=errors)

            y_pred = self.regressor.predict_one(x)
            y_pred = self._bound_prediction(y_pred)
            
            y_diff.appendleft(y_pred)

            forecasts[t] = self.differencer.undiff(y_pred, y_hist)
            y_hist.appendleft(forecasts[t])

            errors.appendleft(0)

        return forecasts
