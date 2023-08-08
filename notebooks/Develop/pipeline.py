# %% PIPELIINE TESTING
import os
import bct
import pygame
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %% ------------------------------------------------------------------
# ##                WORKING DATA EXTRACTION
# ## ------------------------------------------------------------------

def get_data(path):

    directory_path = path
    files = os.listdir(directory_path)
    df = {"IDs": [], "ts": []}

    for file in files:
        if file!= '.DS_Store': # check for hidden folder causing errors
            df["IDs"].append(file)
            session_folder = os.listdir(os.path.join(directory_path, file))

            if len(session_folder)==0:
                df["ts"].append({})
                continue

            elif session_folder[0] != '.DS_Store':
                items = os.listdir(os.path.join(directory_path, file, session_folder[0]))

                # check if the session folder contains the target file
                target_file_present = any(item == 'time_s_LR.csv' for item in items)

                # append an empty dictionary to df["ts"] if the target file is not present
                if not target_file_present:
                    df["ts"].append({})
                    continue

                # loop through each item in the directory
                for item in items: 
                    # check if the item is the target file
                    if item == 'time_s_LR.csv':
                        # extract timeseries
                        tempdf = pd.read_csv(os.path.join(directory_path, file, session_folder[0], item))
                        # Extract the ROI names from the first row
                        roi_names = tempdf.columns.tolist()
                        # Extract the time series data for each ROI
                        time_series = {roi: np.array(tempdf[roi].tolist()) for roi in roi_names}
                        df["ts"].append(time_series)
            ####
            else:
                items = os.listdir(os.path.join(directory_path, file, session_folder[1]))

                # check if the session folder contains the target file
                target_file_present = any(item == 'time_s_LR.csv' for item in items)

                # append an empty dictionary to df["ts"] if the target file is not present
                if not target_file_present:
                    df["ts"].append({})
                    continue

                # loop through each item in the directory
                for item in items: 
                    # check if the item is the target file
                    if item == 'time_s_LR.csv':
                        # extract timeseries
                        tempdf = pd.read_csv(os.path.join(directory_path, file, session_folder[1], item))
                        # Extract the ROI names from the first row
                        roi_names = tempdf.columns.tolist()
                        # Extract the time series data for each ROI
                        time_series = {roi: np.array(tempdf[roi].tolist()) for roi in roi_names}
                        df["ts"].append(time_series)
    return df                 


# %% ------------------------------------------------------------------
# ##                LOAD AND PAIR GESTATIONAL AGE
# ## ------------------------------------------------------------------

def pair_age(data, path, clean = False):

    #directory_path = os.path.dirname(path)
    #TODO change the path/file system so that only one path needs to be initialized
    path = path
    df = pd.read_csv(path, sep = "\t")
    data["b_age"] = []

    for id in data["IDs"]:
        row = df.loc[df["participant_id"] == id]
        birth_age = row["birth_age"].values[0]
        # approximate the age to the 3rd digit after the comma
        data["b_age"].append(round(birth_age,3))

    if clean:
        # transform the data to panda data frame to perform further operations
        df = pd.DataFrame(data)
        # locate empty entries
        df = df.loc[df.ts!={}]
        # reallign the index so to not have empty rows
        df = df.reset_index(drop=True)
        # convert back to dictionary
        data = df.to_dict("list")
    return data


# %% ------------------------------------------------------------------
# ##                MULTIVERSE BUILDING BLOCKS
# ## ------------------------------------------------------------------

# Extracting Connectivity matrixes
def get_connectivity(data, corr_mets = {"covariance", "correlation", "partial correlation"}):
    temp = []
    connectivity = {}

    for i, n in enumerate(list(data.values())[0]):                    # getting the subject dimension
        temp.append(np.zeros((2300,52)))
        temp[i] = np.array(list(data["ts"][i].values())).T            # getting the ts data of a participant in the format requested by nilearn

    for corr_met in corr_mets:                                        # multiverse of connectivity
        myclass = ConnectivityMeasure(kind = corr_met)
        connectivity[corr_met] = myclass.fit_transform(temp)

    return connectivity

### single participant
def get_3_connectivity(data, corr_mets = {"covariance", "correlation", "partial correlation"}, plot = False, sub_idx = np.random.randint(0,300)):
    temp = []
    connectivity = {}

    temp.append(np.zeros((2300,52)))
    subject = {key: value[sub_idx] for key, value in data.items()}
    temp[0] = np.array(list(subject["ts"].values())).T            # getting the ts data of a participant in the format requested by nilearn

    for corr_met in corr_mets:                                        # multiverse of connectivity
        myclass = ConnectivityMeasure(kind = corr_met)
        connectivity[corr_met] = np.squeeze(myclass.fit_transform(temp))

    if plot:
        fig, axs = plt.subplots(1,3, figsize = (15,5))
        fig.suptitle(f'Connectivities Multiverse Step -sub_n: {sub_idx} ')

        for i, corr_met in enumerate(connectivity.keys()):
            im = axs[i].matshow(connectivity[corr_met], cmap='coolwarm')
            axs[i].set_title(corr_met)
            axs[i].set_axis_off()
            fig.colorbar(im, ax = axs[i])
        plt.show()

    return connectivity
# Dealing with negative correlations
def neg_corr(option, f):   
    if  option == "abs":
        out = abs(f)
    elif option == "zero": 
        out = f
        out[np.where(f < 0)] = 0
    elif option == "keep":
        out = f
    return out

### new connectivity
def get_1_connectivity(sub_data, connect_met, p = False):
    temp = []
    connectivity = []

    n_ROIs = len(list(sub_data.keys()))
    n_timeStamp = len(list(sub_data.values())[0])
    temp.append(np.zeros((n_timeStamp,n_ROIs)))
    temp[0] = np.array(list(sub_data.values())).T            # getting the ts data of a participant in the format requested by nilearn

    myclass = ConnectivityMeasure(kind = connect_met)
    connectivity = np.squeeze(myclass.fit_transform(temp))

    if p:
        plt.matshow(connectivity, cmap='coolwarm')
        plt.title(f'{connect_met}')
        plt.colorbar([-1,1])
        plt.show()

    return connectivity




# GSR
def fork_noGSR(sub):
    return sub
def fork_GSR(sub):
    from nilearn.signal import clean
    df = pd.DataFrame(sub)

    # Extract the ROI names from the first row
    roi_names = df.columns.tolist()

    # Global signal
    global_signal = df.mean(axis=1).values

    # Extract the time series data for each ROI
    time_series = {}
    for roi in roi_names:
        time_series[roi] = np.array(df[roi].tolist())
    
    # Regressing the global signal from ROI
    # TODO: change the name
    gsr_df = {}
    for roi in df.columns:
        roi_data = df[roi].values
        roi_data = clean(roi_data.reshape(-1, 1), detrend=True, standardize='zscore_sample', confounds=global_signal.reshape(-1, 1))
        gsr_df[roi] = np.squeeze(roi_data)

    return gsr_df

# Run FORKing path Mapping
def run_mv(data, neg_options, thresholds, weight_options):

    tot_sub          = range(0, len(list(data.values())[0]))                    # create iterable with the size of our data                                      
    d2               = deepcopy(data)
    data["Multiverse"] = []                                                       # initiate new multiverse key
    f1               = {}

    for sub_idx in tot_sub:                                                     # get subject connectivity data 
        f   = get_3_connectivity(d2, sub_idx = sub_idx)
        for connectivity in f.keys():                                           # connectivity measure FORK
            f1[connectivity] = {}

            for negative_values_approach in neg_options:                        # what-to-do-with negative values FORK
                f1[connectivity][negative_values_approach] = {} 
                
                for treshold in thresholds:                                     # tresholds FORK
                    f1[connectivity][negative_values_approach][str(treshold)] = {}
                    temp = neg_corr(negative_values_approach, f[connectivity])  # address negative values
                    temp = bct.threshold_proportional(temp, treshold, copy = True)# apply sparsity treshold
                    
                    for weight in weight_options:                               # handling weights FORK
                        f1[connectivity][negative_values_approach][str(treshold)][weight] = bct.weight_conversion(temp, weight)
        data["Multiverse"].append(f1)


    return data

# Run FORKing path - sgregation only
def new_mv(d):

    BCT_models       = {
        'local efficiency': bct.efficiency_bin,                     # segregation measure
        'modularity (louvain)': bct.modularity_und_louvain_sign,         # segregation measure
        'modularity (probtune)': bct.modularity_probtune_und_sign,  # segregation measure
        'betweennness centrality': bct.betweenness_bin,             # integration measure
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
    ResultsIndVar = np.zeros(((n_w * n_n * n_t * n_c * n_b * n_g), 45150))
    count = 0

    with tqdm(range(n_w * n_n * n_t * n_c * n_b * n_g)) as pbar:
        for G in GSR:
            temp_dic[G] = {}
            for BCT_Num in BCT_models:
                temp_dic[G][BCT_Num] = {}                                               # get subject connectivity data 
                for connectivity in connectivities:                                           # connectivity measure FORK
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
                                    f    = get_1_connectivity(sub, connectivity, sub_idx = idx)
                                    tmp = []
                                    tmp = neg_corr(negative_values_approach, f)  # address negative values
                                    tmp = bct.threshold_proportional(tmp, treshold, copy = True)# apply sparsity treshold
                                    tmp = bct.weight_conversion(tmp, weight)
                                    ss   = analysis_space(BCT_Num, BCT_models, tmp)
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

def analysis_space(BCT_Num, BCT_models, x):
    if (BCT_Num == 'local efficiency' & (weight == "binarize")):
        ss = BCT_models[BCT_Num](x,1)
    elif (BCT_Num == 'local efficiency' & (weight == "normalize")):
        ss = bct.efficiency_wei(x,1)
    elif BCT_Num == 'modularity (louvain)':
        ss, _ = BCT_models[BCT_Num](x, seed=2)
    elif BCT_Num== 'modularity (probtune)':
        ss, _ = BCT_models[BCT_Num](x, seed=2)
    else:
        ss = BCT_models[BCT_Num](x)
    return ss

# Get default FORKs
def get_FORKs():
    
    BCT_models       = {
        'local efficiency': bct.efficiency_bin,                     # segregation measure
        'modularity (louvain)': bct.modularity_louvain_und,         # segregation measure
        'modularity (probtune)': bct.modularity_probtune_und_sign,  # segregation measure
        'betweennness centrality': bct.betweenness_bin,             # integration measure
        "global efficiency": bct.efficiency_bin                     # integration measure
        }

    graph_measures   = ['local efficiency','modularity (louvain)','modularity (probtune)']#'global efficiency': nx.global_efficiency  
    weight_options   = ["binarize", "normalize"]
    neg_options      = [ "abs", "zero", "keep"]
    thresholds       = [0.4, 0.3, 0.25, 0.2, 0.175, 0.150, 0.125, 0.1]
    connectivities   = ["covariance", "correlation", "partial correlation"]

    return BCT_models, neg_options, thresholds, weight_options, graph_measures, connectivities

# Display FORKs
def print_FORKs():

    GSR              = ["GSR, no-GSR"]   
    graph_measures   = ['local efficiency','modularity (louvain)','modularity (probtune)', 'betweennness centrality', 'global efficiency']#'global efficiency': nx.global_efficiency  
    weight_options   = ["binarize", "normalize"]
    neg_options      = [ "abs", "zero", "keep"]
    thresholds       = [0.4, 0.3, 0.25, 0.2, 0.175, 0.150, 0.125, 0.1]
    connectivities   = ["covariance", "correlation", "partial correlation"]

    print("Global Signal Regression:", GSR)
    print("Graph measures:", graph_measures)
    print("Weights preprocessing:", weight_options)
    print("Addressing negative values:", neg_options)
    print("Thresholds:", thresholds)
    print("Connectivities:", connectivities)

    return 

def get_segregation_FORKs():
    
    BCT_models       = {
        'local efficiency': bct.efficiency_bin,                     # segregation measure
        'modularity (louvain)': bct.modularity_louvain_und,         # segregation measure
        'modularity (probtune)': bct.modularity_probtune_und_sign,  # segregation measure
        }

    graph_measures   = ['local efficiency','modularity (louvain)','modularity (probtune)']#'global efficiency': nx.global_efficiency  
    weight_options   = ["binarize", "normalize"]
    neg_options      = [ "abs", "zero", "keep"]
    thresholds       = [0.4, 0.3, 0.25, 0.2, 0.175, 0.150, 0.125, 0.1]
    connectivities   = ["covariance", "correlation", "partial correlation"]

    return BCT_models, neg_options, thresholds, weight_options, graph_measures, connectivities

# More eficient functions to speed up exaustive search
def neg_abs(f):   
    out = abs(f)
    return out
def neg_keep(f):   
    out = f
    return out
def neg_zero(f):
    out = f
    out[np.where(f < 0)] = 0
    return out



# %% Extra

def play_monumental_sound(sound_path):
    pygame.mixer.init()
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()
