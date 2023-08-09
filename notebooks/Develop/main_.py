# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from nilearn.datasets import MNI152_FILE_PATH
from pipeline import get_data, pair_age, split_data
from pipeline import get_3_connectivity
from pipeline import new_mv
from pipeline import get_FORKs, print_FORKs
import pickle

# Prevent warnings about future nilearn package updates
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# %% ------------------- DATA HANDLING ---------------------
# defining relevant paths
signal_path = '/Users/amnesia/Desktop/Master_Thesis/root_dir/end_processing_signal/handy-introduction-022-glbml-21786.mp3'
path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/pipeline_timeseries"
age_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/combined.tsv"
output_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/outputs"


# %%
# Using custom functions to import incompleate data
data_total = get_data(path = path)

# Pair participants to their gestational age
data = pair_age(data_total, age_path, clean=True)       # clean parameter removes uncomplete data

# Split the data into 3 sections
# n° of subjects to build the space: 51
# n° of subjects to assess the prediction of each pipeline: 199
# n° of subjects for post analysis validation: 51
data_space, data_predict, data_lockbox = split_data(data = data,
                                                    SUBJECT_COUNT_SPACE = 51,
                                                    SUBJECT_COUNT_PREDICT = 199,  
                                                    SUBJECT_COUNT_LOCKBOX = 51)

# Get and Display forks we utilize in this multiverse
BCT_models, neg_options, thresholds, weight_options, graph_measures, connectivities = get_FORKs()
print_FORKs()


# %% ----------------- DATA VISUALIZATION -------------------
#TODO: DISPLAY THE DEMOGRAPHICS OF THE DATA

# Print the 3 different connectivities for selected participant
# Infant with lowest Gestational Age: 109
# Infant with highest Gestational Age: 85
sub_idx = np.random.randint(0,300)
connectivities = get_3_connectivity(data = data, plot = True, sub_idx = sub_idx)

# %% ----------------- CREATE THE SPACE -------------------
# This process is time consuming takes approximately 10h to be completed
# Pickled data is provided -> for time efficient exploration of this project

#ModelsResults = new_mv(data_space)
#pickle.dump( ModelsResults, open(str(output_path + "/" + "ModelsResults.p"), "wb" ) )
ModelsResults = pickle.load(open(str(output_path + "/" + "ModelsResults.p"), "rb" ) )
print(ModelsResults.keys())



# %% ------------------------------------------------------------------------------------
# ##                                PLOTTING THE SPACE
# ## ------------------------------------------------------------------------------------

from sklearn import manifold, datasets
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from functools import partial
from time import time
import pickle

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib
from umap import UMAP
import phate
from sklearn.decomposition import PCA

# Load the previous results
Results = ModelsResults['ResultsIndVar']
BCT_Run = ModelsResults['BCT']
Sparsities_Run = ModelsResults['Sparsities']
Data_Run = ModelsResults['Data']
preprocessings = ['GSR', 'noGSR']
Negative_Run = ModelsResults["Negatives"]           
Connectivity_Run = ModelsResults["Connectivity"]    
Weight_Run = ModelsResults["Weights"]               

# %% Scale the data prior to dimensionality reduction
scaler = StandardScaler()
X = scaler.fit_transform(Results.T)
X = X.T
n_neighbors = 20
n_components = 2  # number of components requested. In this case for a 2D space.

# Define different dimensionality reduction techniques
methods = OrderedDict()
"""

#TODO: find why LLE does not work
#LLE = partial(manifold.LocallyLinearEmbedding,
#              n_neighbors, n_components, eigen_solver='dense')
#methods['LLE'] = LLE(method='standard', random_state=0)


methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                           n_neighbors=n_neighbors, random_state=0)
methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
                                 random_state=0)
methods['UMAP'] = UMAP(random_state=40, n_components=2, n_neighbors=200,
                       min_dist=.8)

methods['PHATE'] = phate.PHATE()
methods['PCA'] = PCA(n_components=2)


markers      = ["v", "s", "o", "*", "D", "1", "x", "p", "H", "+", "|", "_", "3", "^", "4", "<", "X"]    # Graph Measures
colourmaps   = {"noGSR": "Oranges", "GSR": "Purples"}                                                   # Preprocessings
hatches      = {"partial correlation": "--", "correlation": "||", "covariance": "**"}                   # Connectivities
sizes        = {"normalize": 80, "binarize": 250}  
widths       = {"abs": 10, "keep": 0, "zero":  5}                                                                              # Binarize or Weighted                                                                                       #

BCT             = np.array(list(BCT_Run.items()))[:, 1]
Sparsities      = np.array(list(Sparsities_Run.items()))[:, 1]
Data            = np.array(list(Data_Run.items()))[:, 1]
Negatives        = np.array(list(Negative_Run.items()))[:, 1]
Connectivities   = np.array(list(Connectivity_Run.items()))[:, 1]
Weights          = np.array(list(Weight_Run.items()))[:, 1]           


# Reduced dimensions
data_reduced = {}
gsDE, axs = plt.subplots(3, 2, figsize=(16, 16), constrained_layout=True)
axs = axs.ravel()

# Perform embedding and plot the results (including info about the approach in the color/intensity and shape).
for idx_method, (label, method) in enumerate(methods.items()):
    Y = method.fit_transform(X)
    # Save the results
    data_reduced[label] = Y
    Lines = {}
    HatchPatch = {}
    for preprocessing in preprocessings:
        BCTTemp = BCT[Data == preprocessing]
        SparsitiesTemp = Sparsities[Data == preprocessing]
        NegativesTemp = Negatives[Data == preprocessing]
        ConnectivitiesTemp = Connectivities[Data == preprocessing]
        WeightsTemp = Weights[Data == preprocessing]
        
        YTemp = Y[Data == preprocessing, :]
        for negatives in neg_options:
            for idx_conn, connect in enumerate(connectivities):
                for weight in weight_options:
                    for idx_bct, bct_model in enumerate(BCT_models):
                        axs[idx_method].scatter(YTemp[:, 0][(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                                YTemp[:, 1][(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                                c=SparsitiesTemp[(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                                marker=markers[idx_bct],
                                                hatch=hatches[connect],
                                                alpha=0.5,
                                                linewidth= widths[negatives],
                                                cmap=colourmaps[preprocessing], s=sizes[weight])
                        Lines[idx_bct] = mlines.Line2D([], [], color='black', linestyle='None',
                                                    marker=markers[idx_bct], markersize=10,
                                                    label=bct_model)
                HatchPatch[idx_conn] = mpatches.Patch(facecolor=[0.1, 0.1, 0.1],
                                                    hatch = hatches[connect],
                                                    label = connect,
                                                    alpha = 0.1)
    # For visualisation purposes show the y and x labels only ons specific plots
    if idx_method % 2 == 0:
        axs[idx_method].set_ylabel('Dimension 1', fontsize=20)
    if (idx_method == 4) or (idx_method == 5):
        axs[idx_method].set_xlabel('Dimension 2', fontsize=20)

    axs[idx_method].set_title("%s " % (label), fontsize=20, fontweight="bold")
    axs[idx_method].axis('tight')
    axs[idx_method].tick_params(labelsize=15)

OrangePatch = mpatches.Patch(color='orange', label='no - Global Signal Regression^')
PurplePatch = mpatches.Patch(color='purple', label='Global Signal Regression')

OrangePatch = mpatches.Patch(color='orange', label='motion regression')
PurplePatch = mpatches.Patch(color=[85 / 255, 3 / 255, 152 / 255], label='global signal regression')

IntensityPatch1 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.4',
                                 alpha=1)
IntensityPatch2 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.2',
                                 alpha=0.4)
IntensityPatch3 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.1',
                                 alpha=0.1)

SizeLines1 = mlines.Line2D([], [], color='black', linestyle='None',
                        marker="*",
                        markersize=10,
                        label="weight normalization")

SizeLines2 = mlines.Line2D([], [], color='black', linestyle='None',
                        marker="*",
                        markersize=20,
                        label="weight binarization")

LineWidth1 = mlines.Line2D([], [], color='black', linestyle='None',
                            marker="_", markersize=20, linewidth=1,
                            label="Negative values: Keep")

LineWidth2 = mlines.Line2D([], [], color='black', linestyle='None',
                            marker="_", markersize=30, linewidth=20,
                            label="Negative values: to Zeros")

LineWidth3 = mlines.Line2D([], [], color='black', linestyle='None',
                            marker="_", markersize=40, linewidth=40,
                            label="Negative values: to Absolute")

BlankLine = mlines.Line2D([], [], linestyle='None')
gsDE.legend(handles=[OrangePatch, PurplePatch, BlankLine, IntensityPatch1,
                     IntensityPatch2, IntensityPatch3, BlankLine,
                     Lines[0], Lines[1], Lines[2], BlankLine,
                     HatchPatch[0], HatchPatch[1], HatchPatch[2], BlankLine,
                     SizeLines1, SizeLines2, BlankLine,
                     LineWidth1, LineWidth2, LineWidth3], fontsize=15,
            frameon=False, bbox_to_anchor=(1.25, .7))

#save plots locally
gsDE.savefig(str(output_path + "/" + 'DifferentEmbeddings.png'), dpi=300, bbox_inches='tight')
gsDE.savefig(str(output_path + "/" +  'DifferentEmbeddings.svg'), format="svg", bbox_inches='tight')
gsDE.show()





#%% ------------------------- MDS -------------------------
methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=10,
                              random_state=21, metric=True)

# Do the same as above but for MDS
Y = methods['MDS'].fit_transform(X)
data_reduced['MDS'] = Y

figMDS = plt.figure(constrained_layout=False, figsize=(21, 15))
gsMDS = figMDS.add_gridspec(nrows=15, ncols=20)
axs = figMDS.add_subplot(gsMDS[:, 0:15])
idx_method = 0

Lines = {}
HatchPatch = {}
for preprocessing in preprocessings:
    BCTTemp = BCT[Data == preprocessing]
    SparsitiesTemp = Sparsities[Data == preprocessing]
    NegativesTemp = Negatives[Data == preprocessing]
    ConnectivitiesTemp = Connectivities[Data == preprocessing]
    WeightsTemp = Weights[Data == preprocessing]

    YTemp = Y[Data == preprocessing, :]
    for negatives in neg_options:
        for idx_conn, connect in enumerate(connectivities):
            for weight in weight_options:
                for idx_bct, bct_model in enumerate(BCT_models):
                    axs.scatter(YTemp[:, 0][(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                            YTemp[:, 1][(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                            c=SparsitiesTemp[(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                            marker=markers[idx_bct],
                                            hatch=hatches[connect],
                                            alpha=0.5,
                                            linewidth= widths[negatives],
                                            cmap=colourmaps[preprocessing], s=sizes[weight])
                    Lines[idx_bct] = mlines.Line2D([], [], color='black', linestyle='None',
                                                marker=markers[idx_bct], markersize=10,
                                                label=bct_model)
            HatchPatch[idx_conn] = mpatches.Patch(facecolor=[0.1, 0.1, 0.1],
                                                    hatch = hatches[connect],
                                                    label = connect,
                                                    alpha = 0.1)
        axs.spines['top'].set_linewidth(1.5)
        axs.spines['right'].set_linewidth(1.5)
        axs.spines['bottom'].set_linewidth(1.5)
        axs.spines['left'].set_linewidth(1.5)
        axs.set_xlabel('Dimension 2', fontsize=20, fontweight="bold")
        axs.set_ylabel('Dimension 1', fontsize=20, fontweight="bold")
        axs.tick_params(labelsize=15)

axs.set_title('Multi-dimensional Scaling', fontsize=25, fontweight="bold")


OrangePatch = mpatches.Patch(color='orange', label='no - Global Signal Regression^')
PurplePatch = mpatches.Patch(color='purple', label='Global Signal Regression')

OrangePatch = mpatches.Patch(color='orange', label='motion regression')
PurplePatch = mpatches.Patch(color=[85 / 255, 3 / 255, 152 / 255], label='global signal regression')

IntensityPatch1 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.4',
                                 alpha=1)
IntensityPatch2 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.2',
                                 alpha=0.4)
IntensityPatch3 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.1',
                                 alpha=0.1)

SizeLines1 = mlines.Line2D([], [], color='black', linestyle='None',
                        marker="*",
                        markersize=10,
                        label="weight normalization")

SizeLines2 = mlines.Line2D([], [], color='black', linestyle='None',
                        marker="*",
                        markersize=20,
                        label="weight binarization")

LineWidth1 = mlines.Line2D([], [], color='black', linestyle='None',
                            marker="_", markersize=10,
                            label="Negative values: Keep")

LineWidth2 = mlines.Line2D([], [], color='black', linestyle='None',
                            marker="_", markersize=25,
                            label="Negative values: to Zeros")

LineWidth3 = mlines.Line2D([], [], color='black', linestyle='None',
                            marker="_", markersize=40,
                            label="Negative values: to Absolute")

BlankLine = mlines.Line2D([], [], linestyle='None')
figMDS.legend(handles=[OrangePatch, PurplePatch, BlankLine, IntensityPatch1,
                     IntensityPatch2, IntensityPatch3, BlankLine,
                     Lines[0], Lines[1], Lines[2], BlankLine,
                     HatchPatch[0], HatchPatch[1], HatchPatch[2], BlankLine,
                     SizeLines1, SizeLines2, BlankLine,
                     LineWidth1, LineWidth2, LineWidth3], fontsize=15,
              frameon=False, bbox_to_anchor=(1.4, 0.8), bbox_transform=axs.transAxes)

figMDS.savefig(str(output_path + "/" + 'MDSSpace.png'), dpi=300)
figMDS.savefig(str(output_path + "/" +  'MDSSpace.svg'), format="svg")
figMDS.show()
pickle.dump(data_reduced, open(str(output_path + "/" + "embeddings_.p"), "wb" ) )

# %% ------------------------------------------------------------------------------------
# ##                             EXHAUSTIVE SEARCH
# ## ------------------------------------------------------------------------------------

 Uncomment later - not used

from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


from pipeline import objective_func_reg
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Load embedding results. This cell is only necessary if you are running this
# part of the analysis separatly.
ModelEmbeddings = pickle.load(open(str(output_path + "/" + "embeddings_.p"), "rb"))
# TODO: find why model embeddings are missing PCA
key = 'MDS'
ModelEmbedding = ModelEmbeddings[key]
PredictedAcc   = np.zeros((len(Data_Run)))
multiverse_section2 = np.zeros((len(Data_Run)))
AgesPrediction = np.asanyarray(data_predict["b_age"])

for i in tqdm(range(len(Data_Run))):
    tempPredAcc, pipeline_result = objective_func_reg(i, AgesPrediction, Sparsities_Run, Data_Run, BCT_models, BCT_Run,
                                    Negative_Run, Weight_Run, Connectivity_Run, data_predict)
    PredictedAcc[i] = tempPredAcc
    multiverse_section2[i] = pipeline_result

# Display how predicted accuracy is distributed across the low-dimensional space
plt.scatter(ModelEmbedding[0: PredictedAcc.shape[0], 0],
            ModelEmbedding[0: PredictedAcc.shape[0], 1],
            c=PredictedAcc, cmap='bwr')
plt.colorbar()

# Dump accuracies
# todo change pickle.dump if runned again
pickle.dump(PredictedAcc, open(str(output_path + "/" + 'predictedAcc_' + key + '.pckl'), 'wb'))
pickle.dump(multiverse_section2, open(str(output_path + "/" + 'multiverse_section2_' + key + '.pckl'), 'wb'))


# %%

from pipeline import objective_func_reg, neg_abs, neg_keep, neg_zero, get_1_connectivity, fork_GSR, fork_noGSR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

from tqdm import tqdm
import bct
import warnings

warnings.filterwarnings("ignore")

ModelEmbeddings = pickle.load(open(str(output_path + "/" + "embeddings_.p"), "rb"))
key = 'MDS'
ModelEmbedding = ModelEmbeddings[key]
PredictedAcc   = np.zeros((len(Data_Run)))
multiverse_section2 = list(np.zeros((len(Data_Run))))
AgesPrediction = np.asanyarray(data_space["b_age"])

def objective_func_reg(TempModelNum, Y, Sparsities_Run,
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
        tmp = bct.weight_conversion(tmp, weight)                        # binarization - normalization
        x = bct.threshold_proportional(tmp, TempThreshold, copy=True)   # thresholding - prooning weak connections
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
i = 0
tempPredAcc, pipeline_result = objective_func_reg(i, AgesPrediction, Sparsities_Run, Data_Run, BCT_models, BCT_Run,
                                Negative_Run, Weight_Run, Connectivity_Run, data_space)
PredictedAcc[i] = tempPredAcc
multiverse_section2[i] = pipeline_result
"""
# %%
from pipeline import objective_func_reg, neg_abs, neg_keep, neg_zero, get_1_connectivity, fork_GSR, fork_noGSR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

from tqdm import tqdm
import bct
import warnings

warnings.filterwarnings("ignore")

ModelEmbeddings = pickle.load(open(str(output_path + "/" + "embeddings_.p"), "rb"))
key = 'MDS'
ModelEmbedding = ModelEmbeddings[key]
PredictedAcc   = np.zeros((len(Data_Run)))
multiverse_section2 = list(np.zeros((len(Data_Run))))
AgesPrediction = np.asanyarray(data_space["b_age"])

def objective_func_reg(TempModelNum, age, Sparsities_Run,
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
    model_linear.fit(X_train.reshape(-1, 1), y_train)
    model_tree.fit(X_train.reshape(-1, 1), y_train)

    pred_linear = model_tree.predict(X_test.reshape(-1, 1))
    pred_tree = model_tree.predict(X_test.reshape(-1, 1))

    # Note: the scores were divided by 10 in order to keep the values close
    # to 0 for avoiding problems with the Bayesian Optimisation
    scores_tree = - mean_absolute_error(y_test, pred_tree)/10
    scores_linear = - mean_absolute_error(y_test, pred_linear)/10

    return scores_tree, scores_linear, TempResults
i = 0
scores_tree, scores_linear, pipeline_result = objective_func_reg(i, AgesPrediction, Sparsities_Run, Data_Run, BCT_models, BCT_Run,
                                Negative_Run, Weight_Run, Connectivity_Run, data_space)
PredictedAcc[i] = scores_linear
multiverse_section2[i] = pipeline_result
# %%
