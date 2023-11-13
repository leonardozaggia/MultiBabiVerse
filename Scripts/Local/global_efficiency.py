#%% Imports + initializations
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

# %% using one arbitrary pipeline -> get participant global efficiency
participant_global_efficiency = []

for sub_idx in range(0, len(data["ts"])):
    # extract participant time series
    sub = data["ts"][sub_idx] 
    # get connectivity matrix
    connectivity = get_1_connectivity(sub, "correlation")
    # apply pipeline
    tmp = neg_abs(connectivity)
    tmp = bct.threshold_proportional(tmp, 0.2, copy=True)   # thresholding - prooning weak connections
    tmp = bct.weight_conversion(tmp, "normalize", copy = True) 
    global_efficiency = bct.efficiency_wei(tmp)
    # reshape global efficiency
    global_efficiency_ext = np.zeros((52,))
    global_efficiency_ext[:] = [global_efficiency for i in range(0, len(global_efficiency_ext))]

    participant_global_efficiency.append(global_efficiency)

# %% Apply the model -> gestational age predicting global efficiency -> plot the model
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load your data and set up your variables
x = np.asanyarray(data["b_age"])
y = np.asanyarray(participant_global_efficiency)

# Sort the data
sort_idx = np.argsort(x)
x = x[sort_idx]
y = y[sort_idx]

# Define the intervals and spline model
for k in range (1, 4):
    intervals = [30, 35, 38]
    spline_model = LSQUnivariateSpline(x, y[:], t=intervals, k=k)

    # Calculate the R-squared value
    y_pred = spline_model(x)
    r2 = r2_score(y[:], y_pred)

    # smooth the data
    xs = np.linspace(x.min(), x.max(), 1000)

    # Plot the data points, spline fit, and intervals
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-whitegrid')
    plt.plot(x, y[:], 'ro', ms=8, label='Data', alpha=0.5)
    plt.plot(xs, spline_model(xs), 'g-', lw=3, label='Spline Fit')
    for interval in intervals:
        plt.axvline(x=interval, color='b', linestyle='--', alpha=0.7, label='Intervals' if interval == intervals[0] else '')
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel("Explained variance = " + str(r2), fontsize=14)
    plt.ylabel('Global Efficiency', fontsize=14)
    plt.title('Spline Regression - Global Efficiency (k={})'.format(k), fontsize=16)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()

# %% Plot using subplots
"""
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load your data and set up your variables
x = np.asanyarray(data["b_age"])
y = np.asanyarray(participant_global_efficiency)

# Sort the data
sort_idx = np.argsort(x)
x = x[sort_idx]
y = y[sort_idx]

# Define the intervals and spline model
fig, axs = plt.subplots(3, 1, figsize=(10, 20))
for k in range(1, 4):
    intervals = [30, 35, 38]
    spline_model = LSQUnivariateSpline(x, y[:], t=intervals, k=k)

    # Calculate the R-squared value
    y_pred = spline_model(x)
    r2 = r2_score(y[:], y_pred)

    # smooth the data
    xs = np.linspace(x.min(), x.max(), 1000)

    # Plot the data points, spline fit, and intervals
    axs[k-1].plot(x, y[:], 'ro', ms=8, label='Data', alpha=0.5)
    axs[k-1].plot(xs, spline_model(xs), 'g-', lw=3, label='Spline Fit')
    for interval in intervals:
        axs[k-1].axvline(x=interval, color='b', linestyle='--', alpha=0.7, label='Intervals' if interval == intervals[0] else '')
    axs[k-1].legend(loc='upper left', fontsize=12)
    axs[k-1].set_xlabel("Explained variance = " + str(r2), fontsize=14)
    axs[k-1].set_ylabel('Global Efficiency', fontsize=14)
    axs[k-1].set_title('Spline Regression - Global Efficiency (k={})'.format(k), fontsize=16)
    axs[k-1].tick_params(labelsize=12)

plt.tight_layout()
plt.show()
"""
# %% Comparison to averaged local efficiency
participant_LOCAL_efficiency = []

for sub_idx in range(0, len(data["ts"])):
    # extract participant time series
    sub = data["ts"][sub_idx] 
    # get connectivity matrix
    connectivity = get_1_connectivity(sub, "correlation")
    # apply pipeline
    tmp = neg_abs(connectivity)
    tmp = bct.threshold_proportional(tmp, 0.2, copy=True)   # thresholding - prooning weak connections
    tmp = bct.weight_conversion(tmp, "normalize", copy = True) 
    LOCAL_efficiency = bct.efficiency_wei(tmp, local=True)

    participant_LOCAL_efficiency.append(LOCAL_efficiency)
# %% Apply the model -> gestational age predicting LOCAL efficiency -> plot the model    
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load your data and set up your variables
avg_participant_LOCAL_efficiency = [np.mean(sublist) for sublist in participant_LOCAL_efficiency]
x = np.asanyarray(data["b_age"])
y = np.asanyarray(avg_participant_LOCAL_efficiency)

# Sort the data
sort_idx = np.argsort(x)
x = x[sort_idx]
y = y[sort_idx]

# Define the intervals and spline model
for k in range (1, 4):
    intervals = [30, 35, 38]
    spline_model = LSQUnivariateSpline(x, y[:], t=intervals, k=k)

    # Calculate the R-squared value
    y_pred = spline_model(x)
    r2 = r2_score(y[:], y_pred)

    # smooth the data
    xs = np.linspace(x.min(), x.max(), 1000)

    # Plot the data points, spline fit, and intervals
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-whitegrid')
    plt.plot(x, y[:], 'ro', ms=8, label='Data', alpha=0.5)
    plt.plot(xs, spline_model(xs), 'g-', lw=3, label='Spline Fit')
    for interval in intervals:
        plt.axvline(x=interval, color='b', linestyle='--', alpha=0.7, label='Intervals' if interval == intervals[0] else '')
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel("Explained variance = " + str(r2), fontsize=14)
    plt.ylabel('LOCAL Efficiency', fontsize=14)
    plt.title('Spline Regression - LOCAL Efficiency (k={})'.format(k), fontsize=16)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()
# %% Plotting LOCAL using subplots
"""
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load your data and set up your variables
pipeline_n = 900
x = np.asanyarray(data["b_age"])
y = np.asanyarray(participant_global_efficiency)

# Sort the data
sort_idx = np.argsort(x)
x = x[sort_idx]
y = y[sort_idx]

# Define the intervals and spline model
fig, axs = plt.subplots(3, 1, figsize=(10, 20))
for k in range(1, 4):
    intervals = [30, 35, 38]
    spline_model = LSQUnivariateSpline(x, y[:], t=intervals, k=k)

    # Calculate the R-squared value
    y_pred = spline_model(x)
    r2 = r2_score(y[:], y_pred)

    # smooth the data
    xs = np.linspace(x.min(), x.max(), 1000)

    # Plot the data points, spline fit, and intervals
    axs[k-1].plot(x, y[:], 'ro', ms=8, label='Data', alpha=0.5)
    axs[k-1].plot(xs, spline_model(xs), 'g-', lw=3, label='Spline Fit')
    for interval in intervals:
        axs[k-1].axvline(x=interval, color='b', linestyle='--', alpha=0.7, label='Intervals' if interval == intervals[0] else '')
    axs[k-1].legend(loc='upper left', fontsize=12)
    axs[k-1].set_xlabel("Explained variance = " + str(r2), fontsize=14)
    axs[k-1].set_ylabel('Global Efficiency', fontsize=14)
    axs[k-1].set_title('Spline Regression - Global Efficiency (k={})'.format(k), fontsize=16)
    axs[k-1].tick_params(labelsize=12)

plt.tight_layout()
plt.show()
"""
# %% Creating the space using global efficiency
#
#
def new_mv_with_global_efficiency(d):

    BCT_models       = {
        'local efficiency': bct.efficiency_bin,                     # segregation measure
        'modularity (louvain)': bct.modularity_louvain_und_sign,    # segregation measure
        'modularity (probtune)': bct.modularity_probtune_und_sign,  # segregation measure
        'betweennness centrality': bct.betweenness_bin,             # integration measure
        'global efficiency': bct.efficiency_wei,                    # integration measure
        }
    
    weight_options   = ["binarize", "normalize"]
    neg_options      = [ "abs", "zero", "keep"]
    thresholds       = [0.4, 0.3, 0.25, 0.2, 0.175, 0.150, 0.125, 0.1]
    connectivities   = ["covariance", "correlation", "partial correlation"]

    tot_sub          = len(list(d.values())[0])              # size of our data - subject dimension 
    n_ROIs           = len(list(d["ts"][0].keys()))
    
    data             = deepcopy(d)
    data["Multiverse"] = []                                  # initiate new multiverse key
    pipelines_graph  = []
    pipelines_conn   = []
    pipe_code        = []
    temp_dic         = {}
    GSR              = ["GSR", "noGSR"]

    n_w = len(weight_options)
    n_n = len(neg_options)
    n_t = len(thresholds)
    n_c = len(connectivities)
    n_b = len(BCT_models)
    n_g = len(GSR)

    BCT_Run          = {}
    Sparsities_Run   = {}
    Connectivity_Run = {}
    Negative_Run     = {}
    Weight_Run       = {}
    Data_Run         = {}
    GroupSummary     = {}
    Results = np.zeros(((n_w * n_n * n_t * n_c * n_b * n_g), n_ROIs))
    ResultsIndVar = np.zeros(((n_w * n_n * n_t * n_c * n_b * n_g), 1275))               #TODO: soft code it ((tot_subject * (tot_subject-1))/2)
    count = 0

    with tqdm(range(n_w * n_n * n_t * n_c * n_b * n_g)) as pbar:
        for G in GSR:
            temp_dic[G] = {}
            for BCT_Num in BCT_models:
                temp_dic[G][BCT_Num] = {}                                               # get subject connectivity data 
                for connectivity in connectivities:                                     # connectivity measure FORK
                    temp_dic[G][BCT_Num][connectivity] = {}
                    for negative_values_approach in neg_options:                        # what-to-do-with negative values FORK
                        temp_dic[G][BCT_Num][connectivity][negative_values_approach] = {}                    
                        for treshold in thresholds:                                     # tresholds FORK
                            temp_dic[G][BCT_Num][connectivity][negative_values_approach][str(treshold)] = {}
                            for weight in weight_options:                               # handling weights FORK
                                temp_dic[G][BCT_Num][connectivity][negative_values_approach][str(treshold)][weight] = {}
                                pipe_c = []
                                pipe_g = np.zeros((tot_sub, n_ROIs))

                                for idx in range(0,tot_sub):
                                    sub  = data["ts"][idx]
                                    if G == "GSR":
                                        sub = fork_GSR(sub)
                                    f    = get_1_connectivity(sub, connectivity)
                                    tmp = []
                                    tmp = neg_corr(negative_values_approach, f)  # address negative values
                                    tmp = bct.threshold_proportional(tmp, treshold, copy = True)# apply sparsity treshold
                                    tmp = bct.weight_conversion(tmp, weight)
                                    ss   = analysis_space(BCT_Num, BCT_models, tmp, weight)
                                    temp_dic[G][BCT_Num][connectivity][negative_values_approach][str(treshold)][weight][data["IDs"][idx]] = deepcopy(tmp)
                                    pipe_c.append(tmp)
                                    pipe_g[idx, :] = ss

                                pipe_code.append("_".join([G, BCT_Num,connectivity, negative_values_approach, str(treshold), weight]))
                                pipelines_graph.append(pipe_g)
                                pipelines_conn.append(np.asanyarray(pipe_c))
                                data["Multiverse"].append(temp_dic)
                                pbar.update(1)


                                BCT_Run[count]          = BCT_Num
                                Sparsities_Run[count]   = treshold
                                Data_Run[count]         = G
                                Connectivity_Run[count] = connectivity
                                Negative_Run[count]     = negative_values_approach
                                Weight_Run[count]       = weight
                                GroupSummary[count]     = 'Mean'
                                
                                # Build an array of similarities between subjects for each
                                # analysis approach
                                cos_sim = cosine_similarity(pipe_g, pipe_g)
                                Results[count, :] = np.mean(pipe_g, axis=0)
                                ResultsIndVar[count, :] = cos_sim[np.triu_indices(tot_sub, k=1)].T
                                count += 1
                                pbar.update(1)
                                
    ModelsResults = {"Results": Results,
                    "ResultsIndVar": ResultsIndVar,
                    "BCT": BCT_Run,
                    "Sparsities": Sparsities_Run,
                    "Data": Data_Run,
                    "Connectivity": Connectivity_Run,
                    "Negatives": Negative_Run,
                    "Weights":Weight_Run,
                    "SummaryStat": GroupSummary,
                    "pipelines_graph": pipelines_graph,
                    "pipelines_conn": pipelines_conn,
                    "Multiverse_dict": data,
                    "pipe_code": pipe_code
                    }
            
    return ModelsResults 
#
#
#
#%% Performing the search
"""
This requires the cluster in order to be run efficiently, otherwise it will take a lot of time.
See file containing the code for the cluster.

"""