# %% PIPELIINE TESTING
import os
import bct
#import pygame
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
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

    myclass = ConnectivityMeasure(kind = connect_met, )
    connectivity = np.squeeze(myclass.fit_transform(temp, ))

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

# Run FORKing path - 3 sgregation, 1 integration
def new_mv(d):

    BCT_models       = {
        'local efficiency': bct.efficiency_bin,                     # segregation measure
        'modularity (louvain)': bct.modularity_louvain_und_sign,    # segregation measure
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
    ResultsIndVar = np.zeros(((n_w * n_n * n_t * n_c * n_b * n_g), 1275))               #TODO: soft code it (sub_num(51) * 25(?))
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
def analysis_space(BCT_Num, BCT_models, x, weight):
    if (BCT_Num == 'local efficiency' and (weight == "binarize")):
        ss = BCT_models[BCT_Num](x,1)
    elif (BCT_Num == 'local efficiency' and (weight == "normalize")):
        ss = bct.efficiency_wei(x,1)
    elif BCT_Num == 'modularity (louvain)':
        ss, _ = BCT_models[BCT_Num](x, seed=2)
    elif BCT_Num == 'modularity (probtune)':
        ss, _ = BCT_models[BCT_Num](x, seed=2)
    elif BCT_Num == 'betweennness centrality' and ((weight == "normalize")):
        x = bct.weight_conversion(x,'lengths')
        ss = bct.betweenness_wei(x)
    else:
        ss = BCT_models[BCT_Num](x)
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
        f = get_1_connectivity(sub, Connectivity)                     # calculate connectivity
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

    age_list = list(map(lambda x: [x], age))
    tmp_list = [list(y) for y in TempResults]
    X_train, X_test, y_train, y_test = train_test_split(age_list, tmp_list,
        test_size=.3, random_state=0)
    #model = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
    model_linear = MultiOutputRegressor(LinearRegression())
    model_tree =  MultiOutputRegressor(RandomForestRegressor())
    model_linear.fit(X_train, y_train)
    model_tree.fit(X_train, y_train)

    pred_linear = model_linear.predict(X_test)
    pred_tree = model_tree.predict(X_test)

    # Note: the scores were divided by 10 in order to keep the values close
    # to 0 for avoiding problems with the Bayesian Optimisation
    scores_tree = - mean_absolute_error(y_test, pred_tree)/10
    scores_linear = - mean_absolute_error(y_test, pred_linear)/10

    return scores_tree, scores_linear, TempResults

# Get default FORKs
def get_FORKs():
    
    BCT_models       = {
        'local efficiency': bct.efficiency_bin,                     # segregation measure
        'modularity (louvain)': bct.modularity_louvain_und_sign,    # segregation measure
        'modularity (probtune)': bct.modularity_probtune_und_sign,  # segregation measure
        'betweennness centrality': bct.betweenness_bin,             # integration measure
        }

    graph_measures   = ['local efficiency','modularity (louvain)','modularity (probtune)']#'global efficiency': nx.global_efficiency  
    weight_options   = ["binarize", "normalize"]
    neg_options      = [ "abs", "zero", "keep"]
    thresholds       = [0.4, 0.3, 0.25, 0.2, 0.175, 0.150, 0.125, 0.1]
    connectivities   = ["covariance", "correlation", "partial correlation"]

    return BCT_models, neg_options, thresholds, weight_options, graph_measures, connectivities
def print_FORKs():

    GSR              = ["GSR, no-GSR"]   
    graph_measures   = ['local efficiency','modularity (louvain)','modularity (probtune)', 'betweennness centrality']#'global efficiency': nx.global_efficiency  
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