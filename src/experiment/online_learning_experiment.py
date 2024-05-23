import pandas as pd

from river import time_series
from typing import Union

from src.experiment.base import Experiment


class OnlineLearningExperiment(Experiment):
    def __init__(self, data: pd.DataFrame, model: time_series.base.Forecaster, 
                 initial_grace_period: int, horizon: int, stride: int) -> None:
        super().__init__(data)
        self.model = model
        self.horizon = horizon
        self.stride = stride
        self.initial_grace_period = initial_grace_period
        self._modulo_for_predictions = initial_grace_period % stride

    def _step(self, idx: int, timestamp: Union[pd.Timestamp, int], value: Union[float, int]) -> None:
        x = {'timestamp': timestamp}
        y = value

        # initial case: grace period to be able to compare with other methods
        if idx < self.initial_grace_period:
            self.model.learn_one(y, x) # type: ignore
        else:
            if idx % self.stride == self._modulo_for_predictions:
                # normal predictions with stride
                horizon = self.future_timestamps(self.horizon)
                y_hat = self.model.forecast(
                    horizon=self.horizon, 
                    xs=[{'timestamp': date} for date in horizon]
                )

                self.add_predictions(
                    origin=timestamp, 
                    values=pd.DataFrame({
                        'timestamp': horizon,
                        'value': y_hat
                    })
                )

            self.model.learn_one(y, x) # type: ignore
