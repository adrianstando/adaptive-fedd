import pandas as pd
import numpy as np
import inspect
import copy

from copy import deepcopy
from tsfresh import extract_features
from tsfresh.feature_extraction import feature_calculators
from sklearn.feature_selection import mutual_info_regression
from tsfresh.feature_extraction import ComprehensiveFCParameters
from typing import Union, List, Dict, Optional

from .adwin import ADWIN
from .fedd import FEDD, append_to_queue, slice_deque
from .base import FeatureExtractor, BasicDriftDetector


@feature_calculators.set_property("fctype", "simple")
def turning_points(x):
    dx = np.diff(x)
    return float(np.sum(dx[1:] * dx[:-1] < 0))


@feature_calculators.set_property("fctype", "simple")
def mutual_information(x, lag):
    ts1 = np.array(x[:-lag]).reshape(-1, 1)
    ts2 = np.array(x[lag:])
    return mutual_info_regression(ts1, ts2)[0]


setattr(feature_calculators, turning_points.__name__, turning_points)
setattr(feature_calculators, mutual_information.__name__, mutual_information)


def metrics(truth, detected, return_counts=False):
    if len(detected) == 0:
        if return_counts:
            return 0, 0, 0, len(truth)
        else:
            return 0, 0

    true_positive = 0
    false_positive = 0  

    i_truth = 0
    i_detected = 0
    n_detected_in_interval = 0

    # check conditions over true data drifts, look forward!
    while i_truth < len(truth) and i_detected < len(detected):
        # handle the initial case
        if i_truth == 0 and detected[i_detected] < truth[0]:
            false_positive += 1
            i_detected += 1
        # handle the normal case
        elif i_truth + 1 < len(truth):
            # there is a drift in the current interval
            if detected[i_detected] < truth[i_truth + 1]:
                n_detected_in_interval += 1
                if n_detected_in_interval == 1:
                    true_positive += 1
                else:
                    false_positive += 1
                i_detected += 1
            # drift is in the next interval
            else:
                # if any drift is not detected, delay is equal to the concept length
                i_truth += 1
                n_detected_in_interval = 0
        # handle the case when the current true data drift is the last true
        # that means the end of the time series
        else:
            if detected[i_detected] > truth[i_truth]:
                n_detected_in_interval += 1
                if n_detected_in_interval == 1:
                    true_positive += 1
                else:
                    false_positive += 1
                i_detected += 1
            elif n_detected_in_interval == 0:
                i_truth += 1
    
    if len(detected) == 0:
        fdr = 0
    else:
        fdr = false_positive / len(detected)

        tpr = true_positive / len(truth)
    
    if return_counts:
        return false_positive, len(detected), true_positive, len(truth)
    else:
        return fdr, tpr
    

class MetadataManager:
    def __init__(self, metadata: pd.DataFrame = pd.DataFrame()) -> None:
        self.metadata = metadata

    def get_feature_names(self) -> List:
        return self.metadata['features'].tolist()
    
    def sample_features(self, n: int = 30) -> List:
        sum_of_weights = np.sum(self.metadata['weight']) # type: ignore
        p = np.array(self.metadata['weight'] / sum_of_weights) # type: ignore
        features = np.array(self.metadata['features']) # type: ignore

        selected_idx = np.random.choice(
            a=np.arange(self.metadata.shape[0]), # type: ignore
            size=n,
            replace=False,
            p=p
        )

        return list(features[selected_idx])
    
    def update_weight(self, feature: str, false_positive: int, n_detected: int, true_positive: int, n_truth: int) -> None:
        self.metadata.loc[self.metadata.features == feature, 'false_positives'] += false_positive
        self.metadata.loc[self.metadata.features == feature, 'n_detected'] += n_detected
        self.metadata.loc[self.metadata.features == feature, 'true_positives'] += true_positive
        self.metadata.loc[self.metadata.features == feature, 'n_truth'] += n_truth

        tp = self.metadata.loc[self.metadata.features == feature, 'true_positives'].iloc[0]
        fp = self.metadata.loc[self.metadata.features == feature, 'false_positives'].iloc[0]
        nt = self.metadata.loc[self.metadata.features == feature, 'n_truth'].iloc[0]
        nd = self.metadata.loc[self.metadata.features == feature, 'n_detected'].iloc[0]

        tpr = tp / nt if not np.isnan(tp / nt) else 0
        fdr = fp / nd if not np.isnan(fp / nd) else 1

        if tpr == 0 and fdr == 0:
            self.metadata.loc[self.metadata.features == feature, 'weight'] = 0
        else:
            self.metadata.loc[self.metadata.features == feature, 'weight'] = np.sqrt(2) - np.sqrt(
                (1 - tpr) ** 2 + fdr ** 2
            )


class AdaptiveFeatureExtarctor(FeatureExtractor):
    def __init__(self, metadata: Union[pd.DataFrame, MetadataManager] = pd.DataFrame(), drift_detector: BasicDriftDetector = ADWIN()):
        super().__init__()
        self.drift_detector = drift_detector

        if isinstance(metadata, pd.DataFrame):
            if metadata.shape[0] == 0:
                self.main_params = ComprehensiveFCParameters()
                self.all_feature_names = list(
                    self.extract_features(pd.DataFrame({
                        'values': [1 for _ in range(10)],
                        'timestamp': [i for i in range(10)]
                    })).index
                )
                self.all_feature_names = [
                    elem.replace('value__', '').replace('values__', '')
                    for elem in self.all_feature_names
                ]

                self.metadata = MetadataManager(pd.DataFrame({
                    'features': self.all_feature_names,
                    'weight': [1 for _ in range(len(self.all_feature_names))],
                    'true_positives': [0 for _ in range(len(self.all_feature_names))],
                    'false_positives': [0 for _ in range(len(self.all_feature_names))],
                    'n_truth': [0 for _ in range(len(self.all_feature_names))],
                    'n_detected': [0 for _ in range(len(self.all_feature_names))]
                }))

            else:
                self.main_params = self.names_to_dict_with_params(metadata['features'].tolist())
                # columns: feature, weight, true_positives, false_positives, n_truth, n_detected
                self.metadata = MetadataManager(metadata )
                self.all_feature_names = self.metadata.get_feature_names()
        else:
            self.metadata = metadata
            self.all_feature_names = self.metadata.get_feature_names()
            self.main_params = self.names_to_dict_with_params(self.all_feature_names)
    
    def sample_features(self, n: int = 30) -> List:
        return self.metadata.sample_features(n=n)

    @staticmethod
    def names_to_dict_with_params(lst: List[str]) -> Dict:
        out = {}
        for i in range(len(lst)):
            name_split = lst[i].replace('value__', '').replace('values__', '').split('__')
            statistic_name = name_split[0]

            if len(name_split) == 1:
                # a statistic without params
                out[statistic_name] = None
                continue
            
            params = {}
            for j in range(1, len(name_split)):
                param_split = name_split[j].split('_')
                param_name = param_split[0] if len(param_split) == 2 else '_'.join(param_split[:-1])
                param_val = param_split[-1]

                if param_val == 'True':
                    param_val = True
                elif param_val == 'False':
                    param_val = False
                elif param_val[0] == '"' and param_val[-1] == '"':
                    param_val = param_val.replace('"', '')
                elif param_val[0] == '(' and param_val[-1] == ')':
                    param_val = tuple([
                        int(elem)
                        for elem in param_val.replace('(', '').replace(')', '').split(',')
                    ])
                else:
                    if float(param_val) - int(float(param_val)) == 0 and not param_val.endswith('.0'):
                        param_val = int(float(param_val))
                    else:
                        param_val = float(param_val)
                
                params[param_name] = param_val
            
            if statistic_name in out.keys():
                out[statistic_name].append(params)
            else:
                out[statistic_name] = [params]
        return out

    def update(self, idx: Optional[int], feature_history: Dict, observed_features: List[str]) -> None:
        for feature_name, feature_ts in feature_history.items():
            adwin = deepcopy(self.drift_detector)
            detected = []

            for i, val in enumerate(feature_ts):
                if adwin.detect(val):
                    detected.append(i)
            
            if idx is None:
                n_detected = len(detected)
                if feature_name in observed_features:
                     # make sure that features which led to not well-based drift detection are punished
                    n_detected = max(1, n_detected)
                
                self.metadata.update_weight(
                    feature=feature_name, 
                    false_positive=n_detected, n_detected=n_detected, 
                    true_positive=0, n_truth=0
                )

            else:
                drift_start = idx - (len(feature_ts) - idx)
                false_positive, n_detected, true_positive, n_truth = metrics([drift_start], detected, True) # type: ignore

                if feature_name in observed_features:
                    # make sure that features which led to well-based drift detection are rewarded
                    true_positive = max(1, true_positive)

                self.metadata.update_weight(
                    feature=feature_name, 
                    false_positive=int(false_positive), n_detected=int(n_detected), 
                    true_positive=true_positive, n_truth=n_truth
                )

    def extract_features(self, time_series: pd.DataFrame) -> pd.Series:
        main_params = self.main_params.copy()

        bicor_param = None
        if 'bicorrelation' in main_params.keys():
            bicor_param = main_params.pop('bicorrelation')
        
        ts = time_series.copy()
        ts['id'] = 0
        features = extract_features(
            ts, column_id='id', column_sort='timestamp',
            default_fc_parameters=main_params,
            disable_progressbar=True,
            n_jobs=0
        ).fillna(0).replace(np.nan, 0).replace(float('nan'), 0).replace([np.inf, -np.inf], 0) # type: ignore

        if bicor_param is None:
            return features.iloc[0] # type: ignore
        
        else:
            std = ts['value'].std()
            if std == 0:
                std = 1e-5
            ts['value'] = (ts['value'] - ts['value'].mean()) / std
            features1 = extract_features(
                ts, column_id='id', column_sort='timestamp',
                default_fc_parameters={
                    "c3": bicor_param
                },
                disable_progressbar=True,
                n_jobs=0
            ).fillna(0).replace(np.nan, 0).replace(float('nan'), 0).replace([np.inf, -np.inf], 0) # type: ignore

            features1.columns = [f'value__bicorrelation__lag_{lag['lag']}' for lag in bicor_param] # type: ignore
            df_features = pd.concat([features, features1], axis=1) # type: ignore
            return df_features.iloc[0]


class AdaptiveFEDD(FEDD):
    def __init__(self, window_size: int = 100, padding: int = 10, queue_data: bool = True,
                 feature_extractor: AdaptiveFeatureExtarctor = AdaptiveFeatureExtarctor(), n_observed_features: int = 30, *args, **kwargs):
        detector = ADWIN(*args, **kwargs)
        self._grace_period = detector.adwin.grace_period
        super().__init__(0.2, float("Inf"), window_size, padding, self._grace_period, queue_data)
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.observed_features = self.feature_extractor.sample_features(n_observed_features)
        self.n_observed_features = n_observed_features
        self.feature_history = {
            f: []
            for f in self.feature_extractor.all_feature_names
        }
        self._drift_index = -1

    @property
    def grace_period(self):
        return self._grace_period
    
    @grace_period.setter
    def grace_period(self, value):
        self._grace_period = value
        self.detector.adwin.grace_period = value

    def add_features_to_history(self, features: pd.Series) -> None:
        for name in features.index:
            self.feature_history[name.replace('value__', '').replace('values__', '')].append(features[name])
    
    def push_weight_changes(self, is_better: bool = True):
        if self._drift_index != -1:
            self.feature_extractor.update(
                idx=self._drift_index if is_better else None,
                feature_history=self.feature_history,
                observed_features=self.observed_features
            )

    def update(self, x: Union[int, float, List]) -> None:        
        append_to_queue(self._queue, x)

        # before training - just collect data
        if not self._is_fitted and len(self._training_queue) < self._training_length:            
            append_to_queue(self._training_queue, x)
        
        # training
        elif not self._is_fitted and len(self._training_queue) == self._training_length:          
            v_list = []
            
            for i in range(self.train_size):
                if self.queue_data:
                    s = slice_deque(self._training_queue, i * self.padding, self.window_size + i * self.padding)
                else:
                    s = slice_deque(self._training_queue, i * self.window_size, (i + 1) * self.window_size)

                v = self.feature_extractor.extract_features(pd.DataFrame({'value': s, 'timestamp': [i for i in range(len(s))]}))
                v.index = [x.replace('value__', '').replace('values__', '') for x in v.index] # type: ignore
                self.add_features_to_history(v)
                v = list(v[self.observed_features])

                v_list.append(v)
            
            self.v0 = np.mean(v_list, axis=0).tolist()

            d_list = [
                self.compute_distance_to_initial(v_list[i])
                for i in range(len(v_list))
            ]
            
            self.detector.add_training_elements(d_list)
            self._is_fitted = True

        # monitoring
        elif self._is_fitted:
            if not self.queue_data and len(self._queue) < self.window_size:
                return
            s = list(self._queue)

            v = self.feature_extractor.extract_features(pd.DataFrame({'value': s, 'timestamp': [i for i in range(len(s))]}))
            v.index = [x.replace('value__', '').replace('values__', '') for x in v.index] # type: ignore
            self.add_features_to_history(v)
            v = list(v[self.observed_features])

            d = self.compute_distance_to_initial(np.array(v))

            if self.detector.detect(d):
                self._drift_index = len(self.feature_history[list(self.feature_history.keys())[0]])
                self._drift_detected = True
    
    def clone(self, new_params: dict | None = None, include_attributes=False):
        # Overwrite clone with small changes so as to preserve FeatureExtractor when cloning

        def is_class_param(param):
            return (
                isinstance(param, tuple)
                and inspect.isclass(param[0])
                and isinstance(param[1], dict)
            )

        # Override the default parameters with the new ones
        params = self._get_params()
        params.update(new_params or {})

        # Clone by recursing
        clone = self.__class__(
            *(params.get("_POSITIONAL_ARGS", [])),
            **{
                name: (
                    getattr(self, name).clone(param[1])
                    if is_class_param(param)
                    else copy.deepcopy(param)
                )
                for name, param in params.items()
                if name != "_POSITIONAL_ARGS"
            },
        )

        if not include_attributes:
            clone.feature_extractor = self.feature_extractor
            return clone

        for attr, value in self.__dict__.items():
            if attr not in params:
                setattr(clone, attr, copy.deepcopy(value))

        clone.feature_extractor = self.feature_extractor # preserve feature extractor when creating a clone
        clone.observed_features = clone.feature_extractor.sample_features(clone.n_observed_features) # new feature set for a clone
        return clone
