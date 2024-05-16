import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import acf, pacf
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import cosine
from tsfresh.feature_extraction.feature_calculators import c3 as bicorrelation
from collections import deque
from typing import Union, List

from src.detectors.base import VirtualDriftDetector, FeatureExtractor, BasicDriftDetector


class EWMA(BasicDriftDetector):
    def __init__(self, Lambda: float = 0.2, drift_threshold: float = 3):
        super().__init__()
        self.Lambda = Lambda
        self.drift_threshold = drift_threshold

        self.initial_average = 0
        self.initial_std = 0
        self.ewma_std = 0
        self.ewma = 0
        self.time = 0

    def add_training_elements(self, values: list) -> None:
        self.initial_average = np.mean(values)
        self.initial_std = np.std(values)
        self.ewma = self.initial_average

    def detect(self, value: float) -> bool:
        self.update_ewma(value)
        return bool(self.ewma > self.initial_average + self.drift_threshold * self.ewma_std)

    def update_ewma(self, distance: float) -> None:
        self.ewma = (1 - self.Lambda) * self.ewma + self.Lambda * distance
        self.time += 1

        parte1 = self.Lambda / (2 - self.Lambda)
        parte2 = 1 - self.Lambda
        parte3 = 2 * self.time
        parte4 = 1 - (parte2 ** parte3)

        parte5 = (parte1 * parte4 * self.initial_std)
        self.ewma_std = np.sqrt(parte5) * self.initial_std


class OriginalFeatureExtractor(FeatureExtractor):
    def __init__(self, autocorrelation_lags: int = 5, partial_autocorrelation_lags: int = 5,
                 bicorrelation_lags: int = 3, mutual_information_lags: int = 3):
        self.autocorrelation_lags = autocorrelation_lags
        self.partial_autocorrelation_lags = partial_autocorrelation_lags
        self.bicorrelation_lags = bicorrelation_lags
        self.mutual_information_lags = mutual_information_lags

    def extract_features(self, time_series: pd.DataFrame) -> list:
        series_diff = pd.Series(time_series['value'])
        series_diff = series_diff - series_diff.shift()
        series_diff = series_diff[1:]

        features = []

        # feature 1
        # autocorrelation
        # it calculates also for lag 0, which is always 1
        autocorrelation = acf(series_diff, nlags=self.autocorrelation_lags, fft=True)
        for i in range(1, len(autocorrelation)):
            features.append(autocorrelation[i])

        # feature 2:
        # partial autocorrelation
        # it calculates also for lag 0, which is always 1
        partial_autocorrelation = pacf(series_diff, nlags=self.partial_autocorrelation_lags)
        for i in range(1, len(partial_autocorrelation)):
            features.append(partial_autocorrelation[i])

        # feature 3:
        # variance
        variance = series_diff.var()
        features.append(variance)

        # feature 4:
        # skewness
        skewness = series_diff.skew()
        features.append(skewness)

        # feature 5:
        # kurtosis
        kurtosis = series_diff.kurtosis()
        features.append(kurtosis)

        # feature 6:
        # turning points rate
        # describes the degree of oscillation of the time series
        # CALCULATED ON THE ORIGINAL TS
        dx = pd.Series(time_series['value']).to_numpy()
        turning_p = float(np.sum(dx[1:] * dx[:-1] < 0))
        features.append(turning_p)

        # feature 7:
        # bicorrelation (three-point autocorrelation / c3 statistic for standardised vector)
        series_diff_standardised = (series_diff.to_numpy() - series_diff.mean()) / np.sqrt(np.array(variance))
        for i in range(1, self.bicorrelation_lags):
            features.append(
                bicorrelation(series_diff_standardised, i)
            )

        # feature 8:
        # mutual information
        for lag in range(1, self.mutual_information_lags + 1):
            ts1 = series_diff[:-lag].to_numpy().reshape(-1, 1)
            ts2 = series_diff[lag:].to_numpy()
            features.append(
                mutual_info_regression(ts1, ts2)[0]
            )

        return features
    

def slice_deque(dq, i_min, i_max):
    return [
        dq[i]
        for i in range(i_min, i_max)
    ]

def append_to_queue(queue, x):
    if isinstance(x, list):
        for elem in x:
            queue.append(elem)
    else:
        queue.append(x)


class FEDD(VirtualDriftDetector):
    def __init__(self, Lambda: float = 0.2, drift_threshold: float = 3, window_size: int = 100, padding: int = 10, 
                 train_size: int = 10, queue_data: bool = True):
        self.detector = EWMA(Lambda, drift_threshold)
        self.feature_extractor = OriginalFeatureExtractor()

        self.window_size = window_size
        self.train_size = train_size
        self.padding = padding

        self._queue = deque(maxlen=window_size)
        self._drift_detected = False
        self.queue_data = queue_data

        self._is_fitted = False
        self.v0 = np.array([])

        if self.queue_data:
            self._training_length = window_size + padding * train_size
            self._training_queue = deque(maxlen=self._training_length)
        else:
            self._training_length = window_size * (train_size + 1)
            self._training_queue = deque(maxlen=self._training_length)

    def update(self, x: Union[int, float, List]) -> None:        
        append_to_queue(self._queue, x)

        # before training - just collect data
        if not self._is_fitted and len(self._training_queue) < self._training_length:            
            append_to_queue(self._training_queue, x)
        
        # training
        elif not self._is_fitted and len(self._training_queue) == self._training_length:
            s0 = slice_deque(self._training_queue, 0, self.window_size)
            v0 = self.feature_extractor.extract_features(pd.DataFrame({'value': s0}))
            self.v0 = v0

            d_list = []
            for i in range(1, self.train_size):
                if self.queue_data:
                    s = slice_deque(self._training_queue, i * self.padding, self.window_size + i * self.padding)
                else:
                    s = slice_deque(self._training_queue, i * self.window_size, (i + 1) * self.window_size)
                v = self.feature_extractor.extract_features(pd.DataFrame({'value': s}))
                d = self.compute_distance_to_initial(np.array(v))
                d_list.append(d)
            
            self.detector.add_training_elements(d_list)
            self._is_fitted = True

        # monitoring
        elif self._is_fitted:
            if self.queue_data and len(self._queue) < self.window_size:
                return
            s = list(self._queue)
            v = self.feature_extractor.extract_features(pd.DataFrame({'value': s}))
            d = self.compute_distance_to_initial(np.array(v))

            if self.detector.detect(d):
                self._drift_detected = True

    def compute_distance_to_initial(self, array: np.ndarray) -> float:
        return float(cosine(self.v0, array) / 2)
    
    @property
    def drift_detected(self):
        if self.queue_data:
            self._queue = deque(maxlen=self.window_size)
        return self._drift_detected
