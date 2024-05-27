import pandas as pd
import numpy as np

from river import time_series
from sklearn.metrics import mean_squared_error as mse
from typing import Union, Callable, List

from src.experiment.base import Experiment
from src.detectors import AdaptiveFEDD
from src.detectors.base import VirtualDriftDetector


class BatchLearningExperiment(Experiment):
    def __init__(self, data: pd.DataFrame, base_model: time_series.base.Forecaster, base_detector: VirtualDriftDetector, 
                 train_size: int, horizon: int, stride: int, validation_size: int = 0, metric: Callable[[np.ndarray, np.ndarray], float] = mse) -> None: # type: ignore
        super().__init__(data)
        self.base_model = base_model
        self.base_detector = base_detector

        self.model = self.base_model.clone()
        self.detector = self.base_detector.clone()

        self.horizon = horizon
        self.metric = metric
        self.train_size = train_size
        self.validation_size = validation_size
        self.stride = stride
        self._modulo_for_predictions = train_size % stride
        self._n_observations_before_model_replacement = self.train_size + self.validation_size + self.horizon if self.validation_size > 0 else self.train_size
        
        self._validation_predictions = pd.DataFrame(columns=['origin', 'timestamp', 'value'])
        self._background_model = None
        self._background_detector = None
        self._background_counter = 0
        self._background_y_true = pd.DataFrame(columns=['timestamp', 'value'])

        self.drift_history = []
        self.replacement_history = []
        self.was_replaced_history = []

    def _compare_models(self, origins: List, y_true: pd.DataFrame, y_hat_1: pd.DataFrame, y_hat_2: pd.DataFrame) -> int:
        values_m1 = []
        values_m2 = []

        for origin in origins:
            horizon = y_hat_1[y_hat_1['origin'] == origin]['timestamp'].tolist()
            y_true_horizon = y_true[y_true['timestamp'].isin(horizon)]['value'].to_numpy()

            values_m1.append(self.metric(
                y_hat_1[np.logical_and(y_hat_1['origin'] == origin, y_hat_1['timestamp'].isin(horizon))]['value'].to_numpy(),
                y_true_horizon
            ))
            values_m2.append(self.metric(
                y_hat_2[np.logical_and(y_hat_2['origin'] == origin, y_hat_2['timestamp'].isin(horizon))]['value'].to_numpy(),
                y_true_horizon
            ))
        
        if np.mean(values_m1) < np.mean(values_m2):
            return 1
        else:
            return 2
    
    def _add_validation_predictions(self, origin: Union[pd.Timestamp, int], values: pd.DataFrame):
        out = values.copy()
        out['origin'] = origin
        out = out[['origin', 'timestamp', 'value']]

        if self._validation_predictions.shape[0] == 0:
            self._validation_predictions = out
        else:
            self._validation_predictions = pd.concat([self._validation_predictions, out])

    def _step(self, idx: int, timestamp: Union[pd.Timestamp, int], value: Union[float, int]) -> None:
        x = {'timestamp': timestamp}
        y = value

        # initial case: no model trained before
        if idx < self.train_size:
            self.model.learn_one(x, y) # type: ignore
            self.detector.update(y)
        else:
            # if enough data for retraining
            if self._background_model is not None and self._background_detector is not None and self._background_counter > self._n_observations_before_model_replacement:
                # the case with validation period
                if self.validation_size > 0:
                    val_origins = set(self._validation_predictions['origin'])
                    y_true = self._background_y_true
                    y_hat_1 = self.predictions[self.predictions['origin'].isin(val_origins)]
                    y_hat_2 = self._validation_predictions

                    # compare performance
                    better_new_model = self._compare_models(list(val_origins), y_true, y_hat_1, y_hat_2)
                    if better_new_model == 1:
                        # the old one is better -- remove the new one, but keep the new detector
                        self._background_model = None
                        self.was_replaced_history.append(False)
                    else:
                        # the new one is better
                        self.model = self._background_model
                        self.was_replaced_history.append(True)
                    
                    # for Adaptive FEDD - push changes and retrain detector
                    if isinstance(self.detector, AdaptiveFEDD):
                        self.detector.push_weight_changes(is_better=better_new_model == 2)
                        self._background_detector = self.base_detector.clone()
                        for y_val in self._background_y_true['value']:
                            self._background_detector.update(y_val) # type: ignore

                else:
                    # when no validation period
                    self.model = self._background_model

                self._background_model = None
                self.detector = self._background_detector
                self._background_detector = None
                self._background_counter = 0
                self.replacement_history.append(timestamp)
                self._background_y_true = pd.DataFrame(columns=['timestamp', 'value'])
                self._validation_predictions = pd.DataFrame(columns=['origin', 'timestamp', 'value'])

            if idx % self.stride == self._modulo_for_predictions:
                # normal predictions
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

            self.model.learn_one(x, y) # type: ignore

            self.detector.update(y) # type: ignore
            if self.detector.drift_detected and self._background_model is None:
                self._background_model = self.base_model.clone()
                self._background_detector = self.base_detector.clone()
                self.drift_history.append(timestamp)

            # learn background model
            if self._background_model is not None and self._background_detector is not None and self._background_counter <= self._n_observations_before_model_replacement:
                # if validation is enabled
                if self.validation_size > 0:
                    # collect y_true for validation
                    new = pd.DataFrame({
                        'timestamp': [timestamp],
                        'value': [y]
                    })
                    if self._background_y_true.shape[0] == 0:
                        self._background_y_true = new
                    else:
                        self._background_y_true = pd.concat([self._background_y_true, new])

                    if self._background_counter >= self.train_size and self._background_counter <= self.train_size + self.validation_size:
                        # if it is after training and inside "validation forecasting origins"
                        # collect validation predictions 
                        if idx % self.stride == self._modulo_for_predictions:
                            if self._background_counter > self.train_size:
                                horizon = self.future_timestamps(self.horizon)
                                y_hat = self._background_model.forecast(
                                    horizon=self.horizon, 
                                    xs=[{'timestamp': date} for date in horizon]
                                )

                                self._add_validation_predictions(
                                    origin=timestamp, 
                                    values=pd.DataFrame({
                                        'timestamp': horizon,
                                        'value': y_hat
                                    })
                                )

                self._background_counter += 1
                self._background_model.learn_one(x, y) # type: ignore
                self._background_detector.update(y) # type: ignore
