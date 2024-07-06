import collections
from river import time_series
from river import base

from typing import Optional


class AROnline(time_series.SNARIMAX):
    def __init__(
        self,
        p: int,
        regressor: base.Regressor | None = None,
        y_hat_min: float = float("-Inf"),
        y_hat_max: float = float("Inf"),
        y_hat_scaling: Optional[str] = None,
        y_hat_scaling_confidence: Optional[float] = None
    ):
        super().__init__(p, 0, 0, 1, 0, 0, 0, regressor)
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
    
    def forecast(self, horizon, xs=None):
        if xs is None:
            xs = [{}] * horizon

        if len(xs) != horizon:
            raise ValueError("the length of xs should be equal to the specified horizon")

        y_hist = collections.deque(self.y_hist)
        y_diff = collections.deque(self.y_diff)
        errors = collections.deque(self.errors)
        forecasts = [None] * horizon

        # update max and min if min-max-scaling chosen
        if self.y_hat_scaling == "minmax":
            if len(y_hist) > 0:
                self.y_hat_min = min(min(y_hist), self.y_hat_min)
                self.y_hat_max = max(max(y_hist), self.y_hat_max)

        for t, x in enumerate(xs):
            x = self._add_lag_features(x=x, Y=y_diff, errors=errors)

            y_pred = self.regressor.predict_one(x)            
            y_diff.appendleft(y_pred)

            forecasts[t] = self.differencer.undiff(y_pred, y_hist)
            forecasts[t] = self._bound_prediction(forecasts[t])
            y_hist.appendleft(forecasts[t])

            errors.appendleft(0)

        return forecasts
