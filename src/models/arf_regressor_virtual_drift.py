import numbers
import random

from river.forest import ARFRegressor
from river.utils.random import poisson
from typing import Optional, List

from src.detectors.base import VirtualDriftDetector
from src.detectors.adaptive_fedd import AdaptiveFEDD


class ARFRegressorVirtualDrift(ARFRegressor):
    def __init__(self, drift_detector: VirtualDriftDetector, warning_detector: VirtualDriftDetector, 
                 virtual_drift_columns: Optional[List[str]] = None, *args, **kwargs):
        super().__init__(
            drift_detector=drift_detector, warning_detector=warning_detector, 
            *args, **kwargs
        )
        self._rng = random.Random(self.seed)
        self.virtual_drift_columns = virtual_drift_columns

        if hasattr(drift_detector, 'grace_period'):
            self.grace_period = drift_detector.grace_period # type: ignore
        else:
            self.grace_period = 0

        self._background_old_drift_detectors = [None for _ in range(self.n_models)]
        self._background_old_trees = [None for _ in range(self.n_models)]
        self._background_old_metric = [None for _ in range(self.n_models)]
        self._background_data_grace_period = [[] for _ in range(self.n_models)]

        if isinstance(drift_detector, AdaptiveFEDD) and isinstance(warning_detector, AdaptiveFEDD):
            for i in range(len(self._warning_detectors)):
                self._warning_detectors[i].observed_features = self._drift_detectors[i].observed_features # type: ignore

    def learn_one(self, x: dict, y: numbers.Number, **kwargs):
        # create a list of values of the selected columns in the input dict
        # drift input is a list
        if self.virtual_drift_columns is None:
            drift_input = list(x.values())
        else:
            drift_input = []
            for column in self.virtual_drift_columns:
                if column not in x.keys():
                    return
                else:
                    drift_input.append(x[column])
        
        # the function is very similar to the original one with one major change: 
        # the detectors are virtual drift detectors, which accept as an input a list of values
        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))

        for i, model in enumerate(self):
            y_pred = model.predict_one(x)

            # Update performance evaluator
            self._metrics[i].update(
                y_true=y,
                y_pred=y_pred,
            )

            # update validation performance for Adaptive FEDD
            if self._background_old_trees[i] is not None:
                self._background_old_metric[i].update(
                    y_true=y,
                    y_pred=self._background_old_trees[i].predict_one(x),
                )

            k = poisson(rate=self.lambda_value, rng=self._rng) # type: ignore
            if k > 0:
                if not self._warning_detection_disabled and self._background[i] is not None:
                    self._background[i].learn_one(x=x, y=y, w=k)  # type: ignore

                model.learn_one(x=x, y=y, w=k)

                if not self._warning_detection_disabled:
                    for _ in range(int(k)):
                        self._warning_detectors[i].update(drift_input) # type: ignore

                    if self._warning_detectors[i].drift_detected:
                        self._background[i] = self._new_base_model()  # type: ignore
                        # Reset the warning detector for the current object
                        if isinstance(self._warning_detectors[i], AdaptiveFEDD):
                            fs = self._warning_detectors[i].observed_features # type: ignore
                            self._warning_detectors[i] = self.warning_detector.clone()
                            self._warning_detectors[i].observed_features = fs # type: ignore
                        else:
                            self._warning_detectors[i] = self.warning_detector.clone()

                        # Update warning tracker
                        self._warning_tracker[i] += 1

                if not self._drift_detection_disabled:
                    if self.grace_period > 0  \
                        and self._background_old_drift_detectors[i] is not None \
                        and isinstance(self._drift_detectors[i], AdaptiveFEDD):

                        self._background_old_drift_detectors[i].update(drift_input) # type: ignore
                        self._background_data_grace_period[i].append([k, drift_input])

                    for _ in range(int(k)):
                        self._drift_detectors[i].update(drift_input) # type: ignore

                    # if grace period is over, push weight changes on the old detector and train a new one
                    if self.grace_period > 0 \
                        and self._background_old_drift_detectors[i] is not None \
                        and isinstance(self._drift_detectors[i], AdaptiveFEDD) \
                        and len(self._background_data_grace_period[i]) == self.grace_period:

                        self._background_old_drift_detectors[i].push_weight_changes(
                            is_better=self._metrics[i].is_better_than(self._background_old_metric[i]) # checks test-than-train metric between the new and ol model
                        )
                        self._warning_detectors[i] = self.warning_detector.clone()
                        self._drift_detectors[i] = self.drift_detector.clone()

                        for k_weight, elem in self._background_data_grace_period[i]:
                            for _ in range(k_weight):
                                self._warning_detectors[i].update(elem)
                                self._drift_detectors[i].update(elem)

                        self._background_data_grace_period[i] = []
                        self._background_old_drift_detectors[i] = None
                        self._background_old_trees[i] = None
                        self._background_old_metric[i] = None

                    if self._drift_detectors[i].drift_detected:
                        if not self._warning_detection_disabled and self._background[i] is not None:
                            # old detector and model to background
                            if self.grace_period > 0 and isinstance(self._drift_detectors[i], AdaptiveFEDD):
                                self._background_old_drift_detectors[i] = self._drift_detectors[i] # type: ignore
                                self._background_old_trees[i] = self.data[i]
                                self._background_old_metric[i] = self.metric.clone()

                            # model background
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
                        