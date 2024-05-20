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

from src.detectors import FEDD, ADWIN
from src.detectors.fedd import append_to_queue, slice_deque
from src.detectors.base import FeatureExtractor, BasicDriftDetector


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


class AdaptiveFeatureExtarctor(FeatureExtractor):
    def __init__(self, metadata: pd.DataFrame = pd.DataFrame(), drift_detector: BasicDriftDetector = ADWIN()):
        super().__init__()
        self.drift_detector = drift_detector
        
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

            self.metadata = pd.DataFrame({
                'features': self.all_feature_names,
                'weight': [1 for _ in range(len(self.all_feature_names))],
                'true_positives': [0 for _ in range(len(self.all_feature_names))],
                'false_positives': [0 for _ in range(len(self.all_feature_names))],
                'n_truth': [0 for _ in range(len(self.all_feature_names))],
                'n_detected': [0 for _ in range(len(self.all_feature_names))]
            })

        else:
            self.main_params = self.names_to_dict_with_params(metadata['features'].tolist())
            # columns: feature, weight, true_positives, false_positives, n_truth, n_detected
            self.metadata = metadata 
            self.all_feature_names = self.metadata['features'].tolist() # type: ignore
    
    def sample_features(self, n: int = 30):
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

    def update(self, idx: Optional[int], feature_history: Dict) -> None:
        for feature_name, feature_ts in feature_history.items():
            adwin = deepcopy(self.drift_detector)
            detected = []

            for i, val in enumerate(feature_ts):
                if adwin.detect(val):
                    detected.append(i)
            
            if idx is None:
                self.metadata.loc[self.metadata.feature == feature_name, 'false_positives'] += len(detected) # type: ignore
                self.metadata.loc[self.metadata.feature == feature_name, 'n_detected'] += len(detected) # type: ignore
            else:
                drift_start = idx - (len(feature_ts) - idx)
                false_positive, n_detected, true_positive, n_truth = metrics([drift_start], detected, True) # type: ignore

                self.metadata.loc[self.metadata.features == feature_name, 'false_positives'] += false_positive # type: ignore
                self.metadata.loc[self.metadata.features == feature_name, 'n_detected'] += n_detected # type: ignore
                self.metadata.loc[self.metadata.features == feature_name, 'true_positives'] += true_positive # type: ignore
                self.metadata.loc[self.metadata.features == feature_name, 'n_truth'] += n_truth # type: ignore

            self.metadata = self.metadata.copy()
            self.metadata['tpr'] = (self.metadata['true_positives'] / self.metadata['n_truth']).fillna(0)
            self.metadata['fdr'] = (self.metadata['false_positives'] / self.metadata['n_detected']).fillna(1)
            self.metadata['weight'] = 2 ** 0.5 - ((1 - self.metadata['tpr']) ** 2 + self.metadata['fdr'] ** 2) ** 0.5
            self.metadata.loc[np.logical_and(self.metadata.tpr == 0, self.metadata.fdr == 0), 'weight'] = 0
            self.metadata = self.metadata[['features', 'weight', 'true_positives', 'false_positives', 'n_truth', 'n_detected']]
            self.metadata = self.metadata.copy()

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
        self.grace_period = detector.adwin.grace_period
        super().__init__(0.2, float("Inf"), window_size, padding, self.grace_period, queue_data)
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.observed_features = self.feature_extractor.sample_features(n_observed_features)
        self.n_observed_features = n_observed_features
        self.feature_history = {
            f: []
            for f in self.feature_extractor.all_feature_names
        }
        self._drift_index = -1

    def add_features_to_history(self, features: pd.Series) -> None:
        for name in features.index:
            self.feature_history[name.replace('value__', '').replace('values__', '')].append(features[name])
    
    def push_weight_changes(self):
        if self._drift_index != -1:
            self.feature_extractor.update(
                idx=self._drift_index,
                feature_history=self.feature_history
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
