# %% PIPELIINE TESTING
import os
import bct
#import pygame
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from numdifftools import Derivative
from sklearn.preprocessing import StandardScaler
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics.pairwise import cosine_similarity
from scipy.interpolate import LSQUnivariateSpline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %% ------------------------------------------------------------------
# ##                     DATA 
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

def split_data(data, SUBJECT_COUNT_SPACE = 51, SUBJECT_COUNT_PREDICT = 199, SUBJECT_COUNT_LOCKBOX = 51):
    df = pd.DataFrame(data)                          # use pandas to facilitate permutation process
    shuffled_df = df.sample(frac=1, random_state=2)  # Set random_state for reproducibility
    df_space = shuffled_df[: SUBJECT_COUNT_SPACE]
    df_predict = shuffled_df[SUBJECT_COUNT_SPACE: (SUBJECT_COUNT_SPACE + SUBJECT_COUNT_PREDICT) ]
    df_lockbox = shuffled_df[(SUBJECT_COUNT_SPACE + SUBJECT_COUNT_PREDICT):]
    # Convert back to the proper dictionary format
    data_space = df_space.to_dict("list")
    data_predict = df_predict.to_dict("list")
    data_lockbox = df_lockbox.to_dict("list")
    return data_space, data_predict, data_lockbox

# TODO: add a dunction that extract the participants used by juan
def get_juan_splits():
    return [63, 63]

# %% ------------------------------------------------------------------
# ##                MULTIVERSE BUILDING BLOCKS
# ## ------------------------------------------------------------------

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

### new connectivity
def get_1_connectivity(sub_data, connect_met, p = False):
    temp = []
    connectivity = []

    n_ROIs = len(list(sub_data.keys()))
    n_timeStamp = len(list(sub_data.values())[0])
    temp.append(np.zeros((n_timeStamp,n_ROIs)))
    temp[0] = np.array(list(sub_data.values())).T            # getting the ts data of a participant in the format requested by nilearn

    myclass = ConnectivityMeasure(kind = connect_met, )
    connectivity = np.squeeze(myclass.fit_transform(temp, ))

    if p:
        plt.matshow(connectivity, cmap='coolwarm')
        plt.title(f'{connect_met}')
        plt.colorbar()
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
    gsr_df = {}
    for roi in df.columns:
        roi_data = df[roi].values
        roi_data = clean(roi_data.reshape(-1, 1), detrend=True, standardize='zscore_sample', confounds=global_signal.reshape(-1, 1))
        gsr_df[roi] = np.squeeze(roi_data)

    return gsr_df

# Run FORKing path - 1 sgregation, 1 integration
def new_mv(d):

    BCT_models       = {
        'local efficiency': bct.efficiency_bin,              # segregation measure
        'global efficiency': bct.efficiency_bin,            # integration measure
        }
    
    weight_options   = ["binarize", "normalize"]
    neg_options      = [ "abs", "zero", "keep"]
    thresholds       = [0.65, 0.6, 0.55, 0.5, 0.55, 0.4, 0.35, 0.3, 0.25, 0.2, 0.150, 0.1, 0.05]
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
    ResultsIndVar = np.zeros(((n_w * n_n * n_t * n_c * n_b * n_g), int((tot_sub * (tot_sub-1))/2)))         
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

# Cluster FORKing path - 1 sgregation, 1 integration
def set_mv_forking(d):

    BCT_models       = {
        'local efficiency': bct.efficiency_bin,              # segregation measure
        'global efficiency': bct.efficiency_bin,            # integration measure
        }
    
    weight_options   = ["binarize", "normalize"]
    neg_options      = [ "abs", "zero", "keep"]
    thresholds       = [0.65, 0.6, 0.55, 0.5, 0.55, 0.4, 0.35, 0.3, 0.25, 0.2, 0.150, 0.1, 0.05]
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
    ResultsIndVar = np.zeros(((n_w * n_n * n_t * n_c * n_b * n_g), int((tot_sub * (tot_sub-1))/2)))         
    count = 0


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
#                                pipe_c = []
#                                pipe_g = np.zeros((tot_sub, n_ROIs))

#                                for idx in range(0,tot_sub):
#                                    sub  = data["ts"][idx]
#                                    if G == "GSR":
#                                        sub = fork_GSR(sub)
#                                    f    = get_1_connectivity(sub, connectivity)
#                                    tmp = []
#                                    tmp = neg_corr(negative_values_approach, f)  # address negative values
#                                    tmp = bct.threshold_proportional(tmp, treshold, copy = True)# apply sparsity treshold
#                                    tmp = bct.weight_conversion(tmp, weight)
#                                    ss   = analysis_space(BCT_Num, BCT_models, tmp, weight)
#                                    temp_dic[G][BCT_Num][connectivity][negative_values_approach][str(treshold)][weight][data["IDs"][idx]] = deepcopy(tmp)
#                                    pipe_c.append(tmp)
#                                    pipe_g[idx, :] = ss

                            pipe_code.append("_".join([G, BCT_Num,connectivity, negative_values_approach, str(treshold), weight]))
#                                pipelines_graph.append(pipe_g)
#                                pipelines_conn.append(np.asanyarray(pipe_c))
#                                data["Multiverse"].append(temp_dic)


                            BCT_Run[count]          = BCT_Num
                            Sparsities_Run[count]   = treshold
                            Data_Run[count]         = G
                            Connectivity_Run[count] = connectivity
                            Negative_Run[count]     = negative_values_approach
                            Weight_Run[count]       = weight
                            GroupSummary[count]     = 'Mean'
                            
                            # Build an array of similarities between subjects for each
                            # analysis approach
#                                cos_sim = cosine_similarity(pipe_g, pipe_g)
#                                Results[count, :] = np.mean(pipe_g, axis=0)
#                                ResultsIndVar[count, :] = cos_sim[np.triu_indices(tot_sub, k=1)].T
                            count += 1
                            
    ModelsResults = {"Results": Results,
                    "ResultsIndVar": ResultsIndVar,
                    "BCT": BCT_Run,
                    "Sparsities": Sparsities_Run,
                    "Data": Data_Run,
                    "Connectivity": Connectivity_Run,
                    "Negatives": Negative_Run,
                    "Weights":Weight_Run,
                    "SummaryStat": GroupSummary,
#                    "pipelines_graph": pipelines_graph,
#                    "pipelines_conn": pipelines_conn,
#                    "Multiverse_dict": data,
                    "pipe_code": pipe_code
                    }
            
    return ModelsResults 
def get_mv(TempModelNum, Sparsities_Run,
    Data_Run, BCT_models, BCT_Run,
    Negative_Run, Weight_Run,
    Connectivity_Run, data):
    
    # load the correct connectivity for this pipeline    
    if Connectivity_Run[TempModelNum] == "correlation":
        Connectivity = "correlation"
    elif Connectivity_Run[TempModelNum] == "covariance":
        Connectivity = "covariance"
    elif Connectivity_Run[TempModelNum] == "partial correlation":
        Connectivity = "partial correlation"
    # load the correct neg_values_option for this pipeline
    if Negative_Run[TempModelNum] == "abs":
        Neg    = {"neg_opt": neg_abs}
    elif Negative_Run[TempModelNum] == "keep":        
        Neg    = {"neg_opt": neg_keep}
    elif Negative_Run[TempModelNum] == "zero":        
        Neg    = {"neg_opt": neg_zero}
    # load the correct weight_option for this pipeline
    if Weight_Run[TempModelNum] == "binarize":
        weight = "binarize"
    else:
        weight = "normalize"
    # load the correct preprocessing for this pipeline
    if Data_Run[TempModelNum] == 'noGSR':
        prep   = {"preprocessing": fork_noGSR}
    elif Data_Run[TempModelNum] == 'GSR':        
        prep   = {"preprocessing": fork_GSR}
    
    TotalSubjects = len(list(data.values())[0])
    TotalRegions  = 52

    TempThreshold = Sparsities_Run[TempModelNum]
    BCT_Num = BCT_Run[TempModelNum]

    TempResults = np.zeros([TotalSubjects, TotalRegions])
    for SubNum in range(0, TotalSubjects):
        sub = data["ts"][SubNum]                                        # extract subject
        sub = prep["preprocessing"](sub)                                # apply preprocessing
        f = get_1_connectivity(sub, Connectivity)                       # calculate connectivity
        tmp = Neg["neg_opt"](f)                                         # address negative values
        tmp = bct.threshold_proportional(tmp, TempThreshold, copy=True) # thresholding - prooning weak connections
        x = bct.weight_conversion(tmp, weight, copy = True)             # binarization - normalization
        if (BCT_Num == 'local efficiency' and (weight == "binarize")):
            ss = BCT_models[BCT_Num](x,1)
        elif (BCT_Num == 'local efficiency' and (weight == "normalize")):
            ss = bct.efficiency_wei(x,1)
        elif (BCT_Num == 'global efficiency' and (weight == "normalize")):
            ge = bct.efficiency_wei(x)
            ss = np.zeros((52,))
            ss[:] = [ge for i in range(0, len(ss))]    
        else:
            ge = BCT_models[BCT_Num](x)
            ss = np.zeros((52,))
            ss[:] = [ge for i in range(0, len(ss))]        
    
        #For each subject for each approach keep the 52 regional values.
        TempResults[SubNum, :] = ss

    return TempResults



def analysis_space(BCT_Num, BCT_models, x, weight):
    if (BCT_Num == 'local efficiency' and (weight == "binarize")):
        ss = BCT_models[BCT_Num](x,1)
    elif (BCT_Num == 'local efficiency' and (weight == "normalize")):
        ss = bct.efficiency_wei(x,1)
    elif (BCT_Num == 'global efficiency' and (weight == "normalize")):
        ge = bct.efficiency_wei(x)
        ss = np.zeros((52,))
        ss[:] = [ge for i in range(0, len(ss))]    
    else:
        ge = BCT_models[BCT_Num](x)
        ss = np.zeros((52,))
        ss[:] = [ge for i in range(0, len(ss))]    
    return ss

def objective_func_reg(TempModelNum, Y, Sparsities_Run,
                       Data_Run, BCT_models, BCT_Run,
                       Negative_Run, Weight_Run,
                       Connectivity_Run, data):
    '''

    Define the objective function for the Bayesian optimization.  This consists
    of the number indicating which model to test, a count variable to help
    control which subjects are tested, a random permutation of the indices of the
    subjects, the predictor variables and the actual y outcomes, the number of
    subjects to include in each iteration

    Parameters
    ----------
        TempModelNum: idx of the analysis being run
        Y: Y variable that will be predicted
        Sparsities_Run: List of threshold used
        Data_Run: Data used for creating the space
        BCT_models: Dictionary containing the list of models used
        BCT_Run: List containing the order in which the BCT models were run
        CommunityIDs: Information about the Yeo network Ids
        data1: Motion Regression functional connectivity data of the subjects
               that were not used to create the space
        data2: Global Signal Regression data for the subjects that gridsearch.cv_results['mean_test_score']were not used
               to create the space
        ClassOrRegress: Define if it is a classification or regression problem
        (0: classification; 1 regression)

    Returns
    -------
        score: Returns the MAE of the predictions
    '''
    # load the correct connectivity for this pipeline    
    if Connectivity_Run[TempModelNum] == "correlation":
        Connectivity = "correlation"
    elif Connectivity_Run[TempModelNum] == "covariance":
        Connectivity = "covariance"
    elif Connectivity_Run[TempModelNum] == "partial correlation":
        Connectivity = "partial correlation"
    # load the correct neg_values_option for this pipeline
    if Negative_Run[TempModelNum] == "abs":
        Neg    = {"neg_opt": neg_abs}
    elif Negative_Run[TempModelNum] == "keep":        
        Neg    = {"neg_opt": neg_keep}
    elif Negative_Run[TempModelNum] == "zero":        
        Neg    = {"neg_opt": neg_zero}
    # load the correct weight_option for this pipeline
    if Weight_Run[TempModelNum] == "binarize":
        weight = "binarize"
    else:
        weight = "normalize"
    # load the correct preprocessing for this pipeline
    if Data_Run[TempModelNum] == 'noGSR':
        prep   = {"preprocessing": fork_noGSR}
    elif Data_Run[TempModelNum] == 'GSR':        
        prep   = {"preprocessing": fork_GSR}
    else:
        ValueError('This type of pre-processing is not supported')
    
    TotalSubjects = len(list(data.values())[0])
    TotalRegions  = 52

    TempThreshold = Sparsities_Run[TempModelNum]
    BCT_Num = BCT_Run[TempModelNum]

    TempResults = np.zeros([TotalSubjects, TotalRegions])
    for SubNum in range(0, TotalSubjects):
        sub = data["ts"][SubNum]                                        # extract subject
        sub = prep["preprocessing"](sub)                                # apply preprocessing
        f   = get_1_connectivity(sub, Connectivity)                     # calculate connectivity
        tmp = Neg["neg_opt"](f)                                         # address negative values
        tmp = bct.threshold_proportional(tmp, TempThreshold, copy=True)   # thresholding - prooning weak connections
        x = bct.weight_conversion(tmp, weight, copy = True)                        # binarization - normalization
        if (BCT_Num == 'local efficiency' and (weight == "binarize")):
            ss = BCT_models[BCT_Num](x,1)
        elif (BCT_Num == 'local efficiency' and (weight == "normalize")):
            ss = bct.efficiency_wei(x,1)
        elif BCT_Num == 'modularity (louvain)':
            ss, _ = BCT_models[BCT_Num](x, seed=2)
        elif BCT_Num == 'modularity (probtune)':
            ss, _ = BCT_models[BCT_Num](x, seed=2)
        elif BCT_Num == 'betweennness centrality' and ((weight == "normalize")):
            x = bct.weight_conversion(x,'lengths', copy = True)
            ss = bct.betweenness_wei(x)
        else:
            ss = BCT_models[BCT_Num](x)
    
        #For each subject for each approach keep the 52 regional values.
        TempResults[SubNum, :] = ss

    X_train, X_test, y_train, y_test = train_test_split(TempResults, Y.ravel(),
        test_size=.3, random_state=0)
    model = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Note: the scores were divided by 10 in order to keep the values close
    # to 0 for avoiding problems with the Bayesian Optimisation
    scores = - mean_absolute_error(y_test, pred)/10
    return scores, TempResults

# Multiple Output Regrassion
def search_ehaustive_reg(TempModelNum, age, Sparsities_Run,
                       Data_Run, BCT_models, BCT_Run,
                       Negative_Run, Weight_Run,
                       Connectivity_Run, data):
 
    # load the correct connectivity for this pipeline    
    if Connectivity_Run[TempModelNum] == "correlation":
        Connectivity = "correlation"
    elif Connectivity_Run[TempModelNum] == "covariance":
        Connectivity = "covariance"
    elif Connectivity_Run[TempModelNum] == "partial correlation":
        Connectivity = "partial correlation"
    # load the correct neg_values_option for this pipeline
    if Negative_Run[TempModelNum] == "abs":
        Neg    = {"neg_opt": neg_abs}
    elif Negative_Run[TempModelNum] == "keep":        
        Neg    = {"neg_opt": neg_keep}
    elif Negative_Run[TempModelNum] == "zero":        
        Neg    = {"neg_opt": neg_zero}
    # load the correct weight_option for this pipeline
    if Weight_Run[TempModelNum] == "binarize":
        weight = "binarize"
    else:
        weight = "normalize"
    # load the correct preprocessing for this pipeline
    if Data_Run[TempModelNum] == 'noGSR':
        prep   = {"preprocessing": fork_noGSR}
    elif Data_Run[TempModelNum] == 'GSR':        
        prep   = {"preprocessing": fork_GSR}
    
    TotalSubjects = len(list(data.values())[0])
    TotalRegions  = 52

    TempThreshold = Sparsities_Run[TempModelNum]
    BCT_Num = BCT_Run[TempModelNum]

    TempResults = np.zeros([TotalSubjects, TotalRegions])
    for SubNum in range(0, TotalSubjects):
        sub = data["ts"][SubNum]                                        # extract subject
        sub = prep["preprocessing"](sub)                                # apply preprocessing
        f = get_1_connectivity(sub, Connectivity)                       # calculate connectivity
        tmp = Neg["neg_opt"](f)                                         # address negative values
        tmp = bct.threshold_proportional(tmp, TempThreshold, copy=True) # thresholding - prooning weak connections
        x = bct.weight_conversion(tmp, weight, copy = True)             # binarization - normalization
        if (BCT_Num == 'local efficiency' and (weight == "binarize")):
            ss = BCT_models[BCT_Num](x,1)
        elif (BCT_Num == 'local efficiency' and (weight == "normalize")):
            ss = bct.efficiency_wei(x,1)
        elif (BCT_Num == 'global efficiency' and (weight == "normalize")):
            ge = bct.efficiency_wei(x)
            ss = np.zeros((52,))
            ss[:] = [ge for i in range(0, len(ss))]    
        else:
            ge = BCT_models[BCT_Num](x)
            ss = np.zeros((52,))
            ss[:] = [ge for i in range(0, len(ss))]        
    
        #For each subject for each approach keep the 52 regional values.
        TempResults[SubNum, :] = ss

    return TempResults

# plotting spline for desired pipeline
def calculate_spline_and_plot(data, pipe_choices, storage, pipeline_n, n_participants=301, k=1, intervals=[28, 31, 37], plot_index=41, outputs=True):
    
    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline_n])

    # Sort the data and select the first n_participants
    sort_idx = np.argsort(x)
    x = x[sort_idx][:n_participants]
    y = y[sort_idx][:n_participants]

    regional_r2 = []
    for i in range(y.shape[1]):
        spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=k)
        y_pred = spline_model(x)
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)

    # Plotting
    if outputs:
        xs = np.linspace(x.min(), x.max(), 1000)
        spline_model = LSQUnivariateSpline(x, y[:, plot_index], t=intervals, k=k)
        y_pred = spline_model(x)
        r2 = r2_score(y[:, plot_index], y_pred)

        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-whitegrid')
        plt.plot(x, y[:, plot_index], 'ro', ms=8, label='Data', alpha=0.5)
        plt.plot(xs, spline_model(xs), 'g-', lw=3, label='Spline Fit')
        for interval in intervals:
            plt.axvline(x=interval, color='b', linestyle='--', alpha=0.7, label='Intervals' if interval == intervals[0] else '')
        plt.legend(loc='upper left', fontsize=12)
        plt.xlabel('Gestational age', fontsize=14)
        plt.ylabel('Graph Measure value', fontsize=14)
        plt.title(pipe_choices[pipeline_n], fontsize=16)
        plt.tick_params(labelsize=12)
        plt.text(0.89, 0.92, f'R² = {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, color='k',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        plt.tight_layout()
        plt.show()

    # Calculate derivative
    spline_model = LSQUnivariateSpline(x, y[:, plot_index], t=intervals, k=k)
    spline_derivative = Derivative(spline_model)
    intervals = np.hstack([intervals, x.max()])
    interval_slopes = [spline_derivative(interval) for interval in intervals]

    # Interpret the direction of the relationship within each interval
    slope_signs = {}
    for i, slope in enumerate(interval_slopes):
        interval_start = intervals[i - 1] if i > 0 else x.min()
        interval_end = intervals[i]
        interval_key = f"{round(interval_start)}-{round(interval_end)}"
        if slope > 0:
            print(f"Positive relationship in interval {interval_key}") if outputs else None
            slope_signs[interval_key] = 'positive'
        elif slope < 0:
            print(f"Negative relationship in interval {interval_key}") if outputs else None
            slope_signs[interval_key] = 'negative'
        else:
            print(f"No relationship in interval {interval_key}") if outputs else None
            slope_signs[interval_key] = 'neutral'

    # Calculate R-squared
    y_pred = spline_model(x)
    r2 = r2_score(y[:, plot_index], y_pred)
    print("Overall R-squared:", r2)

    return regional_r2, slope_signs

# Get default FORKs
def get_FORKs():    

    BCT_models       = {
        'local efficiency': bct.efficiency_bin,              # segregation measure
        'global efficiency': bct.efficiency_bin,             # integration measure
        }

    graph_measures   = ['local efficiency', 'global efficiency']
    weight_options   = ["binarize", "normalize"]
    neg_options      = [ "abs", "zero", "keep"]
    thresholds       = [0.65, 0.6, 0.55, 0.5, 0.55, 0.4, 0.35, 0.3, 0.25, 0.2, 0.150, 0.1, 0.05]
    connectivities   = ["covariance", "correlation", "partial correlation"]

    return BCT_models, neg_options, thresholds, weight_options, graph_measures, connectivities
def print_FORKs():

    GSR              = ["GSR, no-GSR"]   
    graph_measures   = ['local efficiency', 'global efficiency']
    weight_options   = ["binarize", "normalize"]
    neg_options      = [ "abs", "zero", "keep"]
    thresholds       = [0.65, 0.6, 0.55, 0.5, 0.55, 0.4, 0.35, 0.3, 0.25, 0.2, 0.150, 0.1, 0.05]
    connectivities   = ["covariance", "correlation", "partial correlation"]

    print("Global Signal Regression:", GSR)
    print("Graph measures:", graph_measures)
    print("Weights preprocessing:", weight_options)
    print("Addressing negative values:", neg_options)
    print("Thresholds:", thresholds)
    print("Connectivities:", connectivities)

    return 

# Show pipeline results
def show_results(data, pipe_choices, storage, pipeline_n, region = 42):
    regional_r2 = []
    ROIs = list(data["ts"][0].keys())

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
        spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=1)

        # Calculate the R-squared value
        y_pred = spline_model(x)
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)
        if r2 > best_ROI:
            best_ROI = r2
            best_ROI_idx = i

    #  -------------- Plotting the spline model for ROI[i] --------------
    # Generate the x values for visualization
    xs = np.linspace(x.min(), x.max(), 1000)
    if region == "best":
        region = best_ROI_idx
    # Run one more iteration of the model to get the final y_pred
    spline_model = LSQUnivariateSpline(x, y[:, region], t=intervals, k=1)
    # Calculate the R-squared value
    y_pred = spline_model(x)
    r2 = r2_score(y[:, region], y_pred)
    # Plot the data points, spline fit, and intervals
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-whitegrid')
    plt.plot(x, y[:, region], 'ro', ms=8, label='Data', alpha=0.5)
    plt.plot(xs, spline_model(xs), 'g-', lw=3, label='Spline Fit')
    for interval in intervals:
        plt.axvline(x=interval, color='b', linestyle='--', alpha=0.7, label='Intervals' if interval == intervals[0] else '')
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel('Gestational age', fontsize=14)
    plt.ylabel('Graph Measure value', fontsize=14)
    plt.title(pipe_choices[pipeline_n], fontsize=16)
    plt.tick_params(labelsize=12)
    plt.text(0.89, 0.92, f'R² = {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, color='k',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.plot(np.sort(regional_r2))
    plt.show()
    print(f"The average R2 for pipeline {pipeline_n} is: {np.mean(regional_r2)}")
    print(f"The highest R2 is:{best_ROI} and was found in {best_ROI_idx}: {ROIs[best_ROI_idx]}")


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

"""def end_of_processing(sound_path):
    pygame.mixer.init()
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()
"""

""" Micha efficient function
import bct
import time
import numpy as np
from numba import jit
from matplotlib import pyplot as plt

def efficiency_wei(Gw, local=True):

   # The local efficiency is the global efficiency computed on the
   # neighborhood of the node, and is related to the clustering coefficient.


    def invert(W, copy=True):
        if copy:
            W = W.copy()
        E = np.where(W)
        W[E] = 1. / W[E]
        return W
    
    def cuberoot(x):
        return np.sign(x) * np.abs(x)**(1 / 3)

    @jit(nopython=True)
    def distance_inv_wei(G):
        n = len(G)
        D = np.full((n, n), np.inf)
        np.fill_diagonal(D, 0)

        for u in range(n):
            # distance permanence (true is temporary)
            S = np.ones((n,), dtype=np.bool_)
            G1 = G.copy()
            V = np.array([u], dtype=np.int64)
            while True:
                S[V] = 0  # distance u->V is now permanent
                G1[:, V] = 0  # no in-edges as already shortest
                
                for v in V:
                    W = np.where(G1[v, :])[0]  # neighbors of smallest nodes
                    max_len = n
                    td = np.empty((2, max_len))
                    len_W = len(W)
                    td[0, :len_W] = D[u, W]
                    td[1, :len_W] = D[u, v] + G1[v, W]
                    for idx in range(len_W):
                        D[u, W[idx]] = min(td[0, idx], td[1, idx])

                if D[u, S].size == 0:  # all nodes reached
                    break
                minD = np.min(D[u, S])
                if np.isinf(minD):  # some nodes cannot be reached
                    break
                V = np.where(D[u, :] == minD)[0]

        np.fill_diagonal(D, 1)
        D = 1 / D
        np.fill_diagonal(D, 0)
        return D

    n = len(Gw)
    Gl = invert(Gw, copy=True)  # connection length matrix
    A = np.array((Gw != 0), dtype=int)
   
    #local efficiency algorithm described by Wang et al 2016, recommended
    if local:
        E = np.zeros((n,))
        for u in range(n):
            V, = np.where(np.logical_or(Gw[u, :], Gw[:, u].T))
            sw = cuberoot(Gw[u, V]) + cuberoot(Gw[V, u].T)
            e = distance_inv_wei(cuberoot(Gl)[np.ix_(V, V)])
            se = e+e.T
            
            numer = np.sum(np.outer(sw.T, sw) * se) / 2
            if numer != 0:
                # symmetrized adjacency vector
                sa = A[u, V] + A[V, u].T
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                # print numer,denom
                E[u] = numer / denom  # local efficiency
    else:
        ValueError("Removed other options for readability")

    return E

#*****************************************************************************************#
# This script calculates and compares the speed for nodal efficiency calculations with    #
#   1) a custom implementation compiled into machine code with jit                        #
#   2) the normal bct toolbox                                                             #
#                                                                                         #
# The jit implementation is most useful when dealing with large matrices or even loops,   #
# as it is compiled only once.                                                            #
#                                                                                         #
# For a 100x100 matrix the speedup on my machine is 10x, for larger matrices even higher! #
#*****************************************************************************************#

W = np.random.rand(100,100)

start = time.time()
efficiency_jit = efficiency_wei(W, local=True)
end = time.time()
print(f"Jit implementation took {end-start} seconds.")

start = time.time()
efficiency_bct = bct.efficiency_wei(W, local=True)
end = time.time()
print(f"BCT implementation took {end-start} seconds.")

plt.title("Efficiency")
plt.plot(efficiency_jit, lw=4, label="jit")
plt.plot(efficiency_bct, lw=2, label="bct")
plt.show()
# %%


"""