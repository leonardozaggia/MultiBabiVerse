
# %%
import numpy as np
import seaborn as sns
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.gridspec as gridspec
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
        print_rank_results(accs_dict, k, participants, pipe_choices, rank)
    print("\n")


# %% 


####################################################
#    Pipeline diagnosis: checking the variation    #
####################################################
pipe_variation_all_rois = np.zeros((len(pipe_choices), len(ROIs)))
pipe_variation_avg = np.zeros(len(pipe_choices))

for pipeline_n in range(len(pipe_choices)):
    
    var_roi = []
    
    # Load your data and set up your variables
    y = np.asanyarray(storage[pipeline_n])
    var_roi = np.var(y, 0)
    pipe_variation_all_rois[pipeline_n] = var_roi
    pipe_variation_avg[pipeline_n] = np.mean(var_roi)

# %% Plotting the variation of each pipeline
plt.figure(figsize=(8, 6))
plt.style.use('seaborn-whitegrid')
plt.hist(pipe_variation_avg, bins=52, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlim([0, 3])
plt.xlabel('Variation', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Variation of each pipeline', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(labelsize=12)
plt.legend(['Variation'], fontsize=12)
plt.tight_layout()
plt.show()


# %% new function to calculate the r2 score
# TODO: check if this is the correct way to calculate the r2 score
#          - then move it to pipeline.py

def show_all_results(data, storage, k):
    regional_r2 = []
    ROIs = list(data["ts"][0].keys())
    pipe_best_ROI = []
    pipe_best_ROI_idx = []
    pipe_r2 = []

    for pipeline_n in range(936):
        # Load your data and set up your variables
        x = np.asanyarray(data["b_age"])
        y = np.asanyarray(storage[pipeline_n])

        # Sort the data
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]

        # Define the intervals and spline model
        intervals = [28, 31, 37]
        best_ROI_idx = []
        best_ROI = 0
        for i, ROI in enumerate(ROIs):
            spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=k)

            # Calculate the R-squared value
            y_pred = spline_model(x)
            r2 = r2_score(y[:, i], y_pred)
            regional_r2.append(r2)
            if r2 > best_ROI:
                best_ROI = r2
                best_ROI_idx = i
        pipe_best_ROI_idx.append(i)
        pipe_best_ROI.append(best_ROI)
        pipe_r2.append(np.mean(regional_r2))

    return pipe_r2, pipe_best_ROI_idx, pipe_best_ROI

pipe_r2_k1, pipe_best_ROI_idx, pipe_best_ROI = show_all_results(data, storage, 1)
pipe_r2_k2, pipe_best_ROI_idx, pipe_best_ROI = show_all_results(data, storage, 2)
pipe_r2_k3, pipe_best_ROI_idx, pipe_best_ROI = show_all_results(data, storage, 3)


# plot the distribution of r2 for the model
plt.plot(pipe_r2_k1)
plt.plot(pipe_r2_k3)
plt.plot(pipe_r2_k2)

plt.show()



# %% Specification curve analysis - boolean sorted and not
# define forking paths
fork_dict = {
    "Global Signal Regression": ['GSR', 'noGSR'], 
    "neg_options": neg_options,
    "thresholds": ["_" + str(threshold) + "_" for threshold in thresholds], 
    "weight_options": weight_options, 
    "graph_measures": graph_measures, 
    "connectivities": connectivities
}

# create boolean list for each item within each forking path
bool_list = {}
for key, values in fork_dict.items():
    for value in values: 
        bool_list[value] = np.array([True if value in choice else False for choice in pipe_choices])

# corretting for GSR and correlation
bool_list["GSR"] = [not value for value in bool_list["noGSR"]]
bool_list["correlation"] &= ~bool_list["partial correlation"]
items = list(bool_list.keys())
bool_values = list(bool_list.values())

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(bool_values, cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=items)
plt.xlabel('Pipe Choices')
plt.ylabel('Forking Paths')
plt.title('Boolean Values Visualization')
plt.show()

# Sort the pipeline accordingly to accuracy
sort_idx = np.argsort(accs_dict_301["1"])
pipe_r2_sort = accs_dict_301["1"][sort_idx]
pipe_choices_sort = np.asarray(pipe_choices)[sort_idx]
bool_values_sort = np.asarray(bool_values)[:, sort_idx]
""" Sort the pipeline accordingly to accuracy
pipe_r2 = np.asarray(pipe_r2)
sort_idx = np.argsort(pipe_r2)[::-1]
pipe_r2_sort = pipe_r2[sort_idx]
pipe_choices_sort = np.asarray(pipe_choices)[sort_idx]
bool_values_sort = np.asarray(bool_values)[:, sort_idx]
bool_values = np.asarray(bool_values)
# repmat pipe_r2 to create a heatmap
new_pipe_r2 = np.tile(pipe_r2, (24, 1))
"""

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(bool_values_sort, cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=items)
plt.xlabel('Pipe Choices')
plt.ylabel('Forking Paths')
plt.title('SORTED: Boolean Values Visualization')
plt.show()


# %% USABLE PLOT

# Create a grid for the subplots with specific heights for the plots
fig = plt.figure(figsize=(8, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.5])  # 2 rows, 1 column, height ratio of 2:1

# Plot the line plot in the upper subplot
ax0 = plt.subplot(gs[0])
ax0.plot(accs_dict_301["0"], color='red', alpha=0.7)
ax0.plot(accs_dict_301["1"], color='skyblue', alpha=0.7)
ax0.plot(accs_dict_301["2"], color='green', alpha=0.7)
ax0.plot(accs_dict_301["3"], color='purple', alpha=0.7)
ax0.set_xlim(0, 936)
ax0.set_ylim(0, 0.18)
ax0.set_ylabel('Accuracy')
ax0.set_xticks([])  # Disable x-ticks for the upper subplot

# Plot the heatmap in the lower subplot
ax1 = plt.subplot(gs[1])
sns.heatmap(bool_values, cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=items, ax=ax1)
ax1.set_xlabel('Pipe Choices')
ax1.set_ylabel('Forking Paths')

plt.tight_layout()  # Adjust layout for better spacing
plt.show()

# %%SORTEd SPECIFICATION CURVE
# Sorting the pipeline according to model flexibility
sort_idx_0 = np.argsort(accs_dict_301["0"])
sort_idx_1 = np.argsort(accs_dict_301["1"])
sort_idx_2 = np.argsort(accs_dict_301["2"])
sort_idx_3 = np.argsort(accs_dict_301["3"])

accs_0_sort = accs_dict_301["0"][sort_idx_0]
accs_1_sort = accs_dict_301["1"][sort_idx_1]
accs_2_sort = accs_dict_301["2"][sort_idx_2]
accs_3_sort = accs_dict_301["3"][sort_idx_3]


pipe_choices_sort = np.asarray(pipe_choices)[sort_idx]
bool_values_sort = np.asarray(bool_values)[:, sort_idx]

# Create a grid for the subplots with specific heights for the plots
fig = plt.figure(figsize=(8, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.5])  # 2 rows, 1 column, height ratio of 2:1

# Plot the line plot in the upper subplot
ax0 = plt.subplot(gs[0])
ax0.plot(pipe_r2_sort)
ax0.set_xlim(0, 936)
ax0.axvline(x=930, color='r', linestyle='--', alpha=0.7)
ax0.set_ylabel('Accuracy')
ax0.set_xticks([])  # Disable x-ticks for the upper subplot

# Plot the heatmap in the lower subplot
ax1 = plt.subplot(gs[1])
sns.heatmap(bool_values_sort, cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=items, ax=ax1)
ax1.set_xlabel('Pipe Choices')
ax1.set_ylabel('Forking Paths')

plt.tight_layout()  # Adjust layout for better spacing
plt.show()

# %% --------------------------------------------------------------------------------- ##
# ##                         END TO END MULTIVERSE CREATION                            ##
# ## --------------------------------------------------------------------------------- ##


# ## ------------------------------ ##
# ##   Model flexibility Variation  ##
# ## ------------------------------ ##


# concatenate pipe_choices into pipe_choices_e2e
pipe_choices_k0 = [pipe_choice + "_linear" for pipe_choice in pipe_choices]
pipe_choices_k1 = [pipe_choice + "_k1" for pipe_choice in pipe_choices]
pipe_choices_k2 = [pipe_choice + "_k2" for pipe_choice in pipe_choices]
pipe_choices_k3 = [pipe_choice + "_k3" for pipe_choice in pipe_choices]
pipe_choices_e2e = pipe_choices_k0 + pipe_choices_k1 + pipe_choices_k2 + pipe_choices_k3

# concatenate accs_dict into accs_dict_e2e
accs_dict_301_e2e = np.hstack((accs_dict_301["0"], accs_dict_301["1"], accs_dict_301["2"], accs_dict_301["3"]))

fork_dict_e2e = {
    "Global Signal Regression": ['GSR', 'noGSR'], 
    "neg_options": neg_options,
    "thresholds": ["_" + str(threshold) + "_" for threshold in thresholds], 
    "weight_options": weight_options, 
    "graph_measures": graph_measures, 
    "connectivities": connectivities,
    "k": ["linear", "k1", "k2", "k3"]
}

# create boolean list for each item within each forking path
bool_list = {}
for key, values in fork_dict_e2e.items():
    for value in values: 
        bool_list[value] = np.array([True if value in choice else False for choice in pipe_choices_e2e])

# corretting for GSR and correlation
bool_list["GSR"] = [not value for value in bool_list["noGSR"]]
bool_list["correlation"] &= ~bool_list["partial correlation"]
items = list(bool_list.keys())
bool_values = list(bool_list.values())

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(bool_values, cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=items)
plt.xlabel('Pipe Choices')
plt.ylabel('Forking Paths')
plt.title('Boolean Values Visualization')
plt.show()

# Sort the pipeline accordingly to accuracy
sort_idx = np.argsort(accs_dict_301_e2e)
pipe_r2_sort = accs_dict_301_e2e[sort_idx]
pipe_choices_sort = np.asarray(pipe_choices_e2e)[sort_idx]
bool_values_sort = np.asarray(bool_values)[:, sort_idx]

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(bool_values_sort, cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=items)
plt.xlabel('Pipe Choices')
plt.ylabel('Forking Paths')
plt.title('SORTED: Boolean Values Visualization')
plt.show()

""" larger plot
fig, ax = plt.subplots(figsize=(12, 6))  # Increase the width to space columns
"""

# Create a grid for the subplots with specific heights for the plots
fig = plt.figure(figsize=(8, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.5])  # 2 rows, 1 column, height ratio of 1:2.5

# Plot the line plot in the upper subplot
ax0 = plt.subplot(gs[0])
ax0.plot(accs_dict_301_e2e, color='green', alpha=0.7)
ax0.set_xlim(0, 3744)
ax0.set_ylim(0, 0.18)
ax0.set_ylabel('Accuracy in R squared')
ax0.set_xticks([])  # Disable x-ticks for the upper subplot

# Plot the heatmap in the lower subplot
ax1 = plt.subplot(gs[1])
sns.heatmap(bool_values, cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=items, ax=ax1)
ax1.set_xlabel('Pipe Choices')
ax1.set_ylabel('Forking Paths')

plt.tight_layout()  # Adjust layout for better spacing
plt.show()

# SORTED SPECIFICATION CURVE
sort_idx_e2e = np.argsort(accs_dict_301_e2e)
pipe_choices_sort = np.asarray(pipe_choices_e2e)[sort_idx_e2e]
bool_values_sort = np.asarray(bool_values)[:, sort_idx_e2e]
accs_dict_301_e2e_sort = accs_dict_301_e2e[sort_idx_e2e]

# Create a grid for the subplots with specific heights for the plots
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.5])  # 2 rows, 1 column, height ratio of 1:2.5

# Plot the line plot in the upper subplot
ax0 = plt.subplot(gs[0])
ax0.plot(accs_dict_301_e2e_sort, color='green', alpha=0.7)
ax0.set_xlim(0, 3744)
ax0.set_ylim(0, 0.18)
ax0.set_ylabel('Accuracy in R squared')
ax0.set_xticks([])  # Disable x-ticks for the upper subplot

# Plot the heatmap in the lower subplot
ax1 = plt.subplot(gs[1])
sns.heatmap(bool_values_sort, cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=items, ax=ax1)
ax1.set_xlabel('Pipe Choices')
ax1.set_ylabel('Forking Paths')

plt.tight_layout()  # Adjust layout for better spacing
plt.show()
# %% --------------------------------------------------------------------------------- ##
# ##                          FINAL END TO END MULTIVERSE                              ##
# ## --------------------------------------------------------------------------------- ##


# ## ------------------------------ ##
# ##      Sample size Variation     ##
# ## ------------------------------ ##

# concatenate pipe_choices_e2e into pipe_choices_e2e_final
pipe_choices_301 = [pipe_choice + "_301p" for pipe_choice in pipe_choices_e2e]
pipe_choices_150 = [pipe_choice + "_150p" for pipe_choice in pipe_choices_e2e]
pipe_choices_95 = [pipe_choice + "_95p" for pipe_choice in pipe_choices_e2e]
pipe_choices_e2e_final = pipe_choices_301 + pipe_choices_150 + pipe_choices_95

# concatenate accuracies
accs_dict_301_e2e = np.hstack((accs_dict_301["0"], accs_dict_301["1"], accs_dict_301["2"], accs_dict_301["3"]))
accs_dict_150_e2e = np.hstack((accs_dict_150["0"], accs_dict_150["1"], accs_dict_150["2"], accs_dict_150["3"]))
accs_dict_95_e2e = np.hstack((accs_dict_95["0"], accs_dict_95["1"], accs_dict_95["2"], accs_dict_95["3"]))
accs_dict_e2e = np.hstack((accs_dict_301_e2e, accs_dict_150_e2e, accs_dict_95_e2e))

# boolean creation
""""
fork_dict_e2e_final = {
    "Global Signal Regression": ['GSR', 'noGSR'], 
    "neg_options": neg_options,
    "thresholds": ["_" + str(threshold) + "_" for threshold in thresholds], 
    "weight_options": weight_options, 
    "graph_measures": graph_measures, 
    "connectivities": connectivities,
    "k": ["linear", "k1", "k2", "k3"],
    "sample_size": ["301p", "150p", "95p"]
}
"""
fork_dict_e2e_final = {
    "sample_size": ["301p", "150p", "95p"],
    "k": ["linear", "k1", "k2", "k3"],
}

# create boolean list for each item within each forking path
bool_list = {}
for key, values in fork_dict_e2e_final.items():
    for value in values: 
        bool_list[value] = np.array([True if value in choice else False for choice in pipe_choices_e2e_final])

# corretting for GSR and correlation
"""
#bool_list["GSR"] = [not value for value in bool_list["noGSR"]]
#bool_list["correlation"] &= ~bool_list["partial correlation"]
"""
items = list(bool_list.keys())
bool_values = list(bool_list.values())

# Create a heatmap
plt.figure(figsize=(8,6))
sns.heatmap(bool_values, cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=items)
plt.xlabel('Pipe Choices')
plt.ylabel('Forking Paths')
plt.title('Boolean Values Visualization')
plt.show()

# Sort the pipeline accordingly to accuracy
sort_idx = np.argsort(accs_dict_e2e)
pipe_r2_sort = accs_dict_e2e[sort_idx]
pipe_choices_sort = np.asarray(pipe_choices_e2e_final)[sort_idx]
bool_values_sort = np.asarray(bool_values)[:, sort_idx]

# Create a heatmap
plt.figure(figsize=(8,6))
sns.heatmap(bool_values_sort, cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=items)
plt.xlabel('Pipe Choices')
plt.ylabel('Forking Paths')
plt.title('SORTED: Boolean Values Visualization')
plt.show()

#
#
#

## Create a grid for the subplots with specific heights for the plots
fig = plt.figure(figsize=(8, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.5])  # 2 rows, 1 column, height ratio of 1:2.5

# Plot the line plot in the upper subplot
ax0 = plt.subplot(gs[0])
ax0.plot(accs_dict_e2e, color='green', alpha=0.7)
ax0.set_xlim(0, len(accs_dict_e2e))
ax0.set_ylim(0, 0.2)
for i in [3744, 3744*2]:
    ax0.axvline(x=i, color='b', linestyle='--', alpha=0.7)
ax0.set_ylabel('Accuracy in R squared')
ax0.set_xticks([])  # Disable x-ticks for the upper subplot

# Plot the heatmap in the lower subplot
ax1 = plt.subplot(gs[1])
sns.heatmap(bool_values, cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=items, ax=ax1)
ax1.set_xlabel('Pipe Choices')
ax1.set_ylabel('Forking Paths')

plt.tight_layout()  # Adjust layout for better spacing
plt.show()

# SORTED SPECIFICATION CURVE
sort_idx_e2e = np.argsort(accs_dict_e2e)
pipe_choices_sort = np.asarray(pipe_choices_e2e_final)[sort_idx_e2e]
bool_values_sort = np.asarray(bool_values)[:, sort_idx_e2e]
accs_dict_e2e_sort = accs_dict_e2e[sort_idx_e2e]

# Create a grid for the subplots with specific heights for the plots
fig = plt.figure(figsize=(8, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.5])  # 2 rows, 1 column, height ratio of 1:2.5

# Plot the line plot in the upper subplot
ax0 = plt.subplot(gs[0])
ax0.plot(accs_dict_e2e_sort, color='green', alpha=0.7)
ax0.set_xlim(0, len(accs_dict_e2e))
ax0.set_ylim(0, 0.2)
ax0.set_ylabel('Accuracy in R squared')
ax0.set_xticks([])  # Disable x-ticks for the upper subplot

# Plot the heatmap in the lower subplot
ax1 = plt.subplot(gs[1])
sns.heatmap(bool_values_sort, cmap='YlGnBu', cbar=False, xticklabels=False, yticklabels=items, ax=ax1)
ax1.set_xlabel('Pipe Choices')
ax1.set_ylabel('Forking Paths')

plt.tight_layout()  # Adjust layout for better spacing
plt.show()
# %%
