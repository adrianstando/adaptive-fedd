# %% [markdown]
# # Running experiments on the artificial data

# %%
import os
import pandas as pd
import numpy as np

# %%


# %% [markdown]
# ### Find time series files

# %%
def find_csv_files(path):
    files = os.listdir(path)
    files = [filename for filename in files if filename.endswith('.csv')]
    files = [os.path.join(path, file) for file in files]
    return files

artificial_ts_files = find_csv_files('../data/raw/2024_04_11_artificial_data')

# %%


# %% [markdown]
# ### Create model input preprocessing

# %%
from river import compose

# %%
def get_daily_dummies(x):
    n = x['timestamp'] % 168
    n = n // 24
    
    return {
        f'day_{i}': 1 if i == n else 0
        for i in range(7)
    }

def get_hourly_dummies(x):
    n = x['timestamp'] % 168
    n = n % 24

    return {
        f'hour_{i}': 1 if i == n else 0
        for i in range(24)
    }

extract_features = compose.TransformerUnion(
    get_daily_dummies, get_hourly_dummies
)

# %%


# %% [markdown]
# ### Definition of models and detectors

# %%
from river.forest import ARFRegressor
from river.time_series import HoltWinters
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

# %%
from src.models import ARBatch, AROnline, ARFRegressorVirtualDrift
from src.detectors import ADWIN, FEDD, AdaptiveFEDD, KSWIN, NoDrift

# %%
from multiprocessing.managers import BaseManager
from src.detectors.adaptive_fedd import MetadataManager, AdaptiveFeatureExtarctor

# %%
random_seed = 42

day_seasonality = 24 # 24 hours
week_seasonality = day_seasonality * 7

model_window_size = day_seasonality # the last day
model_train_size = 14 * model_window_size # two weeks
model_horizon = day_seasonality # one day in the future
model_validation_size = 2 * day_seasonality # two days in the future

stride = 6 # 6 hours = 1/4 of a day
detector_window_size = week_seasonality # the last week
detector_train_size = model_train_size # the same train size as for the model
detector_train_size_n_shifts = int((detector_train_size - detector_window_size) / stride) # number of shifts for training

adaptive_fedd_observed_features = 15 # number of observed features in Adaptive FEDD

# %%


# %% [markdown]
# Detectors

# %%
adwin_virtual_warning = ADWIN(delta=0.01, grace_period=detector_train_size_n_shifts, clock=4)
adwin_virtual_drift = ADWIN(delta=0.001, grace_period=detector_train_size_n_shifts, clock=4)

kswin_virtual_warning = KSWIN(alpha=0.01, window_size=model_train_size, stat_size=detector_window_size, seed=random_seed)
kswin_virtual_drift = KSWIN(alpha=0.005, window_size=model_train_size, stat_size=detector_window_size, seed=random_seed)

fedd_arf_warning = FEDD(drift_threshold=3, window_size=detector_window_size, stride=stride, train_size=detector_train_size_n_shifts, queue_data=False)
fedd_arf_drift = FEDD(drift_threshold=5, window_size=detector_window_size, stride=stride, train_size=detector_train_size_n_shifts, queue_data=False)
fedd_drift = FEDD(drift_threshold=5, window_size=detector_window_size, stride=stride, train_size=detector_train_size_n_shifts, queue_data=True)

# %%
class MultiprocessingMetadataManager(BaseManager):
    pass

metadata = pd.read_csv('../data/interim/2024_04_11_artificial_data/feature_metadata.csv', index_col=0)
metadata.columns = ['features', 'weight', 'true_positives', 'false_positives', 'n_truth', 'n_detected']

MultiprocessingMetadataManager.register('MetadataManager', MetadataManager)
manager = MultiprocessingMetadataManager()
manager.start()

metadata_manager = manager.MetadataManager(metadata=metadata, random_seed=random_seed)

# %%
adaptive_fedd_arf_warning = AdaptiveFEDD(window_size=detector_window_size, stride=stride, queue_data=False, n_observed_features=adaptive_fedd_observed_features,
                                         feature_extractor=AdaptiveFeatureExtarctor(metadata=metadata_manager, drift_detector=adwin_virtual_drift.clone()),
                                         delta=0.8, grace_period=detector_train_size_n_shifts, clock=4)
adaptive_fedd_arf_drift = AdaptiveFEDD(window_size=detector_window_size, stride=stride, queue_data=False, n_observed_features=adaptive_fedd_observed_features,
                                       feature_extractor=AdaptiveFeatureExtarctor(metadata=metadata_manager, drift_detector=adwin_virtual_drift.clone()),
                                       delta=0.5, grace_period=detector_train_size_n_shifts, clock=4)

adaptive_fedd_drift = AdaptiveFEDD(window_size=detector_window_size, stride=stride, queue_data=True, n_observed_features=adaptive_fedd_observed_features,
                                   feature_extractor=AdaptiveFeatureExtarctor(metadata=metadata_manager, drift_detector=adwin_virtual_drift.clone()),
                                   delta=0.001, grace_period=detector_train_size_n_shifts, clock=4)

# %%


# %% [markdown]
# Models

# %%
arx_batch = (
    extract_features | 
    ARBatch(p=model_window_size, train_size=model_train_size)
)

arx_online = (
    extract_features | 
    AROnline(p=model_window_size)
)

# %%
lightgbm_batch = (
    extract_features | 
    ARBatch(p=model_window_size, train_size=model_train_size, 
            regressor=LGBMRegressor(n_jobs=1, reg_alpha=0.1, reg_lambda=0.1, random_state=random_seed))
)

random_forest_batch = (
    extract_features | 
    ARBatch(p=model_window_size, train_size=model_train_size, 
            regressor=RandomForestRegressor(n_jobs=1, random_state=random_seed))
)

# %%
adaptive_random_forest = (
    extract_features |
    AROnline(p=model_window_size, regressor=ARFRegressor(grace_period=10, seed=random_seed))
)

adaptive_random_forest_virtual_drift_adaptive_fedd = (
    extract_features |
    AROnline(p=max(model_window_size, detector_window_size), 
            regressor=ARFRegressorVirtualDrift(
                drift_detector=adaptive_fedd_arf_drift.clone(), warning_detector=adaptive_fedd_arf_warning.clone(), 
                virtual_drift_columns=[f"y-{i+1}" for i in range(detector_window_size)], 
                model_columns=[f"y-{i+1}" for i in range(model_window_size)] + list(extract_features.transform_one({'timestamp': 321}).keys()),
                seed=random_seed, grace_period=10 # grace period for tree split
            )
    )
)

# %%
# holt_winters_forecasting = HoltWinters(alpha=0.3, beta=0.3, gamma=0.4, seasonality=week_seasonality)

# %%


# %% [markdown]
# ### Parameters

# %%
from src.experiment import OnlineLearningExperiment, BatchLearningExperiment

# %%
parameters = []

# %%
# online learning
model_list = [
    #('online__arx', arx_online),
    #('online__arf', adaptive_random_forest), 
    ('online__arf__virtual_adaptive_fedd', adaptive_random_forest_virtual_drift_adaptive_fedd),
    # ('online__holt_winters', holt_winters_forecasting)
]

# %%
for time_series_path in artificial_ts_files:
    for model_name, model_obj in model_list:
        parameters.append(
            (
                time_series_path, model_name, OnlineLearningExperiment(
                    data=pd.read_csv(time_series_path, index_col=0), model=model_obj.clone(),
                    initial_grace_period=detector_train_size_n_shifts, horizon=model_horizon, stride=stride
                )
            )
        )

# %%
# simple batch learning
model_list = [
    ('batch__arx', arx_batch),
    ('batch__lightgbm', lightgbm_batch), 
    ('batch__rf', random_forest_batch),
]

detector_list = [
    ('no_drift', NoDrift()),
    ('adwin', adwin_virtual_drift),
    ('kswin', kswin_virtual_drift),
    ('fedd', adaptive_fedd_drift)
]

# %%
#for time_series_path in artificial_ts_files:
#    for model_name, model_obj in model_list:
#        for detector_name, detector_obj in detector_list:
#            model_full_name = f"{model_name}__{detector_name}"
#            parameters.append(
#                (
#                    time_series_path, model_full_name, BatchLearningExperiment(
#                        data=pd.read_csv(time_series_path, index_col=0), base_model=model_obj.clone(), horizon=model_horizon,
#                        base_detector=detector_obj.clone(), train_size=model_train_size, stride=stride, validation_size=0
#                    )
#                )
#            )

# %%
# advanced batch learning with validation
model_list = [
    ('batch__arx', arx_batch),
    ('batch__lightgbm', lightgbm_batch), 
    ('batch__rf', random_forest_batch),
]

# %%
#for time_series_path in artificial_ts_files:
#    for model_name, model_obj in model_list:
#        model_full_name = f"{model_name}__adaptive_fedd"
#        parameters.append(
#            (
#                time_series_path, model_full_name, BatchLearningExperiment(
#                    data=pd.read_csv(time_series_path, index_col=0), base_model=model_obj.clone(), horizon=model_horizon,
#                    base_detector=adaptive_fedd_drift.clone(), train_size=model_train_size, stride=stride, validation_size=model_validation_size
#                )
#            )
#        )

# %%


# %% [markdown]
# ### Run experiments

# %%
import pickle
from multiprocessing import Pool

# %%
N_CPU = 24
SAVE_PATH = '../data/processed/2024_05_26_artificial_data'
os.makedirs(SAVE_PATH, exist_ok=True)

# %%
def run_experiment_and_save_output(time_series_path, model_full_name, experiment_obj, save_path):
    print(f"Starting experiment: {model_full_name} on {time_series_path}", flush=True)
    ts_name = os.path.basename(time_series_path).replace('.csv', '')
    path_directory_to_save = os.path.join(save_path, ts_name)
    os.makedirs(path_directory_to_save, exist_ok=True)
    path_to_save_pickle_file = os.path.join(path_directory_to_save, f"{model_full_name}.pickle")

    #if os.path.exists(path_to_save_pickle_file):
    #    return 

    #if 'adaptive_fedd' in path_to_save_pickle_file:
    #    return

    ts_length = experiment_obj.max_len
    for q in range(ts_length):
        #if q == 3000:
        #    for i in range(len(experiment_obj.model[1].regressor._warning_detectors)):
        #        experiment_obj.model[1].regressor._warning_detectors[i]._drift_detected = True
        #if q == 3500:
        #    for i in range(len(experiment_obj.model[1].regressor._warning_detectors)):
        #        experiment_obj.model[1].regressor._drift_detectors[i]._drift_detected = True
        #        experiment_obj.model[1].regressor._drift_detectors[i]._drift_index = len(experiment_obj.model[1].regressor._drift_detectors[i].feature_history[list(experiment_obj.model[1].regressor._drift_detectors[i].feature_history.keys())[0]])
        experiment_obj.step()
    
    print(f"Finished the experiment: {model_full_name} on {time_series_path}", flush=True)

    # remove any connections to the shared metadata object
    if isinstance(experiment_obj, OnlineLearningExperiment):
        experiment_obj.model = None
    elif isinstance(experiment_obj, BatchLearningExperiment):
        experiment_obj.base_model = None
        experiment_obj.base_detector = None
        experiment_obj.model = None
        experiment_obj.detector = None
        experiment_obj._background_model = None
        experiment_obj._background_detector = None
    
    object_to_save = {
        'path_to_save_pickle_file': path_to_save_pickle_file, 
        'time_series_path': time_series_path, 
        'model_full_name': model_full_name, 
        'experiment_obj': experiment_obj
    }

    with open(path_to_save_pickle_file, 'wb') as f:
        pickle.dump(object_to_save, f)
    
    print(f"Saved files for the experiment: {model_full_name} on {time_series_path}", flush=True)

# %%
# add save path to parameters
for i in range(len(parameters)):
    parameters[i] = parameters[i] + (SAVE_PATH, )

# %%
with Pool(N_CPU) as pool:
    pool.starmap(run_experiment_and_save_output, parameters)

# %%
# metadata_manager.get_metadata().to_csv(os.path.join(SAVE_PATH, 'feature_metadata_after_batch_adaptive_fedd.csv'))
metadata_manager.get_metadata().to_csv(os.path.join(SAVE_PATH, 'feature_metadata_after_arf_online_with_adaptive_fedd.csv'))

# %%
manager.shutdown()

# %%


# %%


# %%








