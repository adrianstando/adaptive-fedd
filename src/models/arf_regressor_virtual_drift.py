import numbers
import random
from river.forest import ARFRegressor
from river.utils.random import poisson
from typing import Optional, List

from src.detectors.base import VirtualDriftDetector


class ARFRegressorVirtualDrift(ARFRegressor):
    def __init__(self, drift_detector: VirtualDriftDetector, warning_detector: VirtualDriftDetector, 
                 virtual_drift_columns: Optional[List[str]] = None, *args, **kwargs):
        super().__init__(
            drift_detector=drift_detector, warning_detector=warning_detector, 
            *args, **kwargs
        )
        self._rng = random.Random(self.seed)
        self.virtual_drift_columns = virtual_drift_columns

    def learn_one(self, x: dict, y: numbers.Number, **kwargs):
        # the function is very similar to the original one with one major change: 
        # the detectors are virtual drift detectors, which accept as an input a list of values
        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))

        # create a list of values of the selected columns in the input dict
        # drift input is a list
        if self.virtual_drift_columns is None:
            drift_input = list(x.values())
        else:
            drift_input = [
                x[column]
                for column in self.virtual_drift_columns
            ]

        for i, model in enumerate(self):
            y_pred = model.predict_one(x)

            # Update performance evaluator
            self._metrics[i].update(
                y_true=y,
                y_pred=y_pred,
            )

            k = poisson(rate=self.lambda_value, rng=self._rng) # type: ignore
            if k > 0:
                if not self._warning_detection_disabled and self._background[i] is not None:
                    self._background[i].learn_one(x=x, y=y, w=k)  # type: ignore

                model.learn_one(x=x, y=y, w=k)

                if not self._warning_detection_disabled:
                    self._warning_detectors[i].update(drift_input) # type: ignore

                    if self._warning_detectors[i].drift_detected:
                        self._background[i] = self._new_base_model()  # type: ignore
                        # Reset the warning detector for the current object
                        self._warning_detectors[i] = self.warning_detector.clone()

                        # Update warning tracker
                        self._warning_tracker[i] += 1

                if not self._drift_detection_disabled:
                    self._drift_detectors[i].update(drift_input) # type: ignore

                    if self._drift_detectors[i].drift_detected:
                        if not self._warning_detection_disabled and self._background[i] is not None:
                            self.data[i] = self._background[i]
                            self._background[i] = None
                            self._warning_detectors[i] = self.warning_detector.clone()
                            self._drift_detectors[i] = self.drift_detector.clone()
                            self._metrics[i] = self.metric.clone()
                        else:
                            self.data[i] = self._new_base_model()
                            self._drift_detectors[i] = self.drift_detector.clone()
                            self._metrics[i] = self.metric.clone()

                        # Update warning tracker
                        self._drift_tracker[i] += 1
