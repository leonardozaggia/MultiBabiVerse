
# %%
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from pipeline import get_data, pair_age
from pipeline import get_FORKs
import pickle

# Relevant paths
signal_path = '/Users/amnesia/Desktop/Master_Thesis/root_dir/end_processing_signal/handy-introduction-022-glbml-21786.mp3'
path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/pipeline_timeseries"
age_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/combined.tsv"
output_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir_G_&_L/outputs"

# Load data
data_total = get_data(path = path)
data = pair_age(data_total, age_path, clean=True)
BCT_models, neg_options, thresholds, weight_options, graph_measures, connectivities = get_FORKs()

# Load results
accs_dict_301 = pickle.load(open(str(output_path + "/" + 'accs_dict_301.p'), 'rb'))
accs_dict_150 = pickle.load(open(str(output_path + "/" + 'accs_dict_150.p'), 'rb'))
accs_dict_95 = pickle.load(open(str(output_path + "/" + 'accs_dict_95.p'), 'rb'))
accs_dict_all = {301: accs_dict_301, 150: accs_dict_150, 95: accs_dict_95}

# Load necessary variables
storage = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["301_subjects_936_pipelines"]
pipe_choices = np.asanyarray(pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["pipeline_choices"])
ROIs = list(data["ts"][0].keys())

# Set the rank value as needed
rank = 25  

# %% Inefficient
# Linear regression
print("Model Linear")
# ------------------- 301 participants ------------------- #
accs_0_301participant = np.asanyarray(accs_dict_301["0"])
sort_idx = np.argsort(accs_0_301participant)
pipe_sorted = pipe_choices[sort_idx]
accs_0_301participant_sorted = accs_0_301participant[sort_idx]
idx = np.where(pipe_choices == pipe_sorted[-rank])
print(f"{pipe_choices[idx]}; index = {idx}")
print(accs_0_301participant_sorted[-rank])
# ------------------- 150 participants ------------------- #
accs_0_150participant = np.asanyarray(accs_dict_150["0"])
sort_idx = np.argsort(accs_0_150participant)
pipe_sorted = pipe_choices[sort_idx]
accs_0_150participant_sorted = accs_0_150participant[sort_idx]
idx = np.where(pipe_choices == pipe_sorted[-rank])
print(f"{pipe_choices[idx]}; index = {idx}")
print(accs_0_150participant_sorted[-rank])
# ------------------- 95 participants -------------------- #
accs_0_95participant = np.asanyarray(accs_dict_95["0"])
sort_idx = np.argsort(accs_0_95participant)
pipe_sorted = pipe_choices[sort_idx]
accs_0_95participant_sorted = accs_0_95participant[sort_idx]
idx = np.where(pipe_choices == pipe_sorted[-rank])
print(f"{pipe_choices[idx]}; index = {idx}")
print(accs_0_95participant_sorted[-rank])
print()
print()

# K = 1
print("Model K = 1")
# ------------------- 301 participants ------------------- #
accs_1_301participant = np.asanyarray(accs_dict_301["1"])
sort_idx = np.argsort(accs_1_301participant)
pipe_sorted = pipe_choices[sort_idx]
accs_1_301participant_sorted = accs_1_301participant[sort_idx]
idx = np.where(pipe_choices == pipe_sorted[-rank])
print(f"{pipe_choices[idx]}; index = {idx}")
print(accs_1_301participant_sorted[-rank])
# ------------------- 150 participants ------------------- #
accs_1_150participant = np.asanyarray(accs_dict_150["1"])
sort_idx = np.argsort(accs_1_150participant)
pipe_sorted = pipe_choices[sort_idx]
accs_1_150participant_sorted = accs_1_150participant[sort_idx]
idx = np.where(pipe_choices == pipe_sorted[-rank])
print(f"{pipe_choices[idx]}; index = {idx}")
print(accs_1_150participant_sorted[-rank])
# ------------------- 95 participants -------------------- #
accs_1_95participant = np.asanyarray(accs_dict_95["1"])
sort_idx = np.argsort(accs_1_95participant)
pipe_sorted = pipe_choices[sort_idx]
accs_1_95participant_sorted = accs_1_95participant[sort_idx]
idx = np.where(pipe_choices == pipe_sorted[-rank])
print(f"{pipe_choices[idx]}; index = {idx}")
print(accs_1_95participant_sorted[-rank])
print()
print()

# K = 2
print("Model K = 2")
# ------------------- 301 participants ------------------- #
accs_2_301participant = np.asanyarray(accs_dict_301["2"])
sort_idx = np.argsort(accs_2_301participant)
pipe_sorted = pipe_choices[sort_idx]
accs_2_301participant_sorted = accs_2_301participant[sort_idx]
idx = np.where(pipe_choices == pipe_sorted[-rank])
print(f"{pipe_choices[idx]}; index = {idx}")
print(accs_2_301participant_sorted[-rank])
# ------------------- 150 participants ------------------- #
accs_2_150participant = np.asanyarray(accs_dict_150["2"])
sort_idx = np.argsort(accs_2_150participant)
pipe_sorted = pipe_choices[sort_idx]
accs_2_150participant_sorted = accs_2_150participant[sort_idx]
idx = np.where(pipe_choices == pipe_sorted[-rank])
print(f"{pipe_choices[idx]}; index = {idx}")
print(accs_2_150participant_sorted[-rank])
# ------------------- 95 participants -------------------- #
accs_2_95participant = np.asanyarray(accs_dict_95["2"])
sort_idx = np.argsort(accs_2_95participant)
pipe_sorted = pipe_choices[sort_idx]
accs_2_95participant_sorted = accs_2_95participant[sort_idx]
idx = np.where(pipe_choices == pipe_sorted[-rank])
print(f"{pipe_choices[idx]}; index = {idx}")
print(accs_2_95participant_sorted[-rank])
print()
print()

# K = 3
print("Model K = 3")
# ------------------- 301 participants ------------------- #
accs_3_301participant = np.asanyarray(accs_dict_301["3"])
sort_idx = np.argsort(accs_3_301participant)
pipe_sorted = pipe_choices[sort_idx]
accs_3_301participant_sorted = accs_3_301participant[sort_idx]
print(np.where(pipe_choices == pipe_sorted[-rank]))
print(accs_3_301participant_sorted[-rank])
# ------------------- 150 participants ------------------- #
accs_3_150participant = np.asanyarray(accs_dict_150["3"])
sort_idx = np.argsort(accs_3_150participant)
pipe_sorted = pipe_choices[sort_idx]
accs_3_150participant_sorted = accs_3_150participant[sort_idx]
idx = np.where(pipe_choices == pipe_sorted[-rank])
print(f"{pipe_choices[idx]}; index = {idx}")
print(accs_3_150participant_sorted[-rank])
# ------------------- 95 participants -------------------- #
accs_3_95participant = np.asanyarray(accs_dict_95["3"])
sort_idx = np.argsort(accs_3_95participant)
pipe_sorted = pipe_choices[sort_idx]
accs_3_95participant_sorted = accs_3_95participant[sort_idx]
idx = np.where(pipe_choices == pipe_sorted[-rank])
print(f"{pipe_choices[idx]}; index = {idx}")
print(accs_3_95participant_sorted[-rank])
print()
print()




# %% Efficient

def print_rank_results(accs_dict, k, num_participants, pipe_choices, rank):
    accs = np.asanyarray(accs_dict[str(k)])
    sort_idx = np.argsort(accs)
    pipe_sorted = pipe_choices[sort_idx]
    accs_sorted = accs[sort_idx]
    idx = np.where(pipe_choices == pipe_sorted[-rank])
    print(f"------------------- {num_participants} participants -------------------")
    print(f"{pipe_choices[idx]}; index: " + str(idx))
    print(accs_sorted[-rank])

num_participants = [301, 150, 95]

for k in range(4):  # Assuming the models run from 0 to 3
    if k == 0:
        print("Model Linear")
    else:
        print(f"Model K = {k}")
    for participants in num_participants:
        accs_dict_key = f"{k}_{participants}participant"
        accs_dict = accs_dict_all[participants]
        print_sorted_results(accs_dict, k, participants, pipe_choices, rank)
    print("\n")


# %%
