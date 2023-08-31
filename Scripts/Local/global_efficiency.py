#%%
import bct 
import numpy as np
import pickle
from pipeline import get_data, pair_age, split_data, get_1_connectivity, neg_abs

# def paths
signal_path = '/Users/amnesia/Desktop/Master_Thesis/root_dir/end_processing_signal/handy-introduction-022-glbml-21786.mp3'
path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/pipeline_timeseries"
age_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/combined.tsv"
output_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/outputs"

# load data
data_total = get_data(path = path)
data = pair_age(data_total, age_path, clean=True)
data_space, data_predict, data_lockbox = split_data(data = data, 
                                                    SUBJECT_COUNT_SPACE = 51, 
                                                    SUBJECT_COUNT_PREDICT = 199, 
                                                    SUBJECT_COUNT_LOCKBOX = 51)

storage = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["199_subjects_1152_pipelines"]

# %% get connectivity matrix
sub_idx = 200
sub = data["ts"][sub_idx] 
connectivity = get_1_connectivity(sub, "correlation")
tmp = neg_abs(connectivity)
tmp = bct.threshold_proportional(tmp, 0.2, copy=True)   # thresholding - prooning weak connections
x = bct.weight_conversion(tmp, "normalize", copy = True) 
new = bct.efficiency_wei(x)
new

# %%
