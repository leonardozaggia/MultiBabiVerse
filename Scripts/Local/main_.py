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
data_space, data_predict, data_lockbox = split_data(data = data,
                                                    SUBJECT_COUNT_SPACE = 51,     # n° of subjects to build the space: 51
                                                    SUBJECT_COUNT_PREDICT = 199,  # n° of subjects to assess the prediction of each pipeline: 199
                                                    SUBJECT_COUNT_LOCKBOX = 51)   # n° of subjects for post analysis validation: 51

# Get and Display forks we utilize in this multiverse
BCT_models, neg_options, thresholds, weight_options, graph_measures, connectivities = get_FORKs()
print_FORKs()


# %% ----------------- DATA VISUALIZATION -------------------
#TODO: DISPLAY THE DEMOGRAPHICS OF THE DATA
# Set up the plot style
plt.figure(figsize=(8, 6))
plt.style.use('seaborn-whitegrid')
plt.hist(np.asanyarray(data["b_age"]), bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Age distribution', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(labelsize=12)
plt.legend(['Infants'], fontsize=12)
plt.tight_layout()
plt.show()

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
#TODO: find why LLE does not work
LLE = partial(manifold.LocallyLinearEmbedding,
             n_neighbors=n_neighbors, n_components=n_components, eigen_solver='dense')
methods['LLE'] = LLE(method='standard', random_state=0)

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

OrangePatch = mpatches.Patch(color='orange', label='no - Global Signal Regression^')
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
                     Lines[0], Lines[1], Lines[2], Lines[3], BlankLine,
                     HatchPatch[0], HatchPatch[1], HatchPatch[2], BlankLine,
                     SizeLines1, SizeLines2, BlankLine,
                     LineWidth1, LineWidth2, LineWidth3], fontsize=15,
            frameon=False, bbox_to_anchor=(1.25, .7))

#save plots locally
gsDE.savefig(str(output_path + "/" + 'DifferentEmbeddings.png'), dpi=300, bbox_inches='tight')
gsDE.savefig(str(output_path + "/" +  'DifferentEmbeddings.svg'), format="svg", bbox_inches='tight')
gsDE.show()


methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=10,
                              random_state=21, metric=True)
Y = methods['MDS'].fit_transform(X)
data_reduced['MDS'] = Y


#%% ------------------------- Single Plot -------------------------
key = 't-SNE'
title_dict = {"MDS": "Multi-dimensional Scaling",
              "t-SNE": "t-Distributed Stochastic Neighbor Embedding",
              "SE": "Spectral Embedding",
              "UMAP": "Uniform Manifold Approximation and Projection", 
              "PHATE": "Potential of Heat-diffusion for Affinity-based Transition Embedding", 
              "PCA": "Principal Component Analysis",
              "LLE": "Locally Linear Embedding"}


# Do the same as above but for chosen embedding
Y = data_reduced[key]

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

axs.set_title( title_dict[key], fontsize=25, fontweight="bold")


OrangePatch = mpatches.Patch(color='orange', label='no - Global Signal Regression')
PurplePatch = mpatches.Patch(color='purple', label='Global Signal Regression')

OrangePatch = mpatches.Patch(color='orange', label='no - global signal regression')
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
                     Lines[0], Lines[1], Lines[2], Lines[3], BlankLine,
                     HatchPatch[0], HatchPatch[1], HatchPatch[2], BlankLine,
                     SizeLines1, SizeLines2, BlankLine,
                     LineWidth1, LineWidth2, LineWidth3], fontsize=15,
              frameon=False, bbox_to_anchor=(1.4, 0.8), bbox_transform=axs.transAxes)

figMDS.savefig(str(output_path + "/" + key + 'Space.png'), dpi=300)
figMDS.savefig(str(output_path + "/" +  key + 'Space.svg'), format="svg")
figMDS.show()
#pickle.dump(data_reduced, open(str(output_path + "/" + "embeddings_.p"), "wb" ) )
"""
"""
# %% ------------------------------------------------------------------------------------
# ##                             EXHAUSTIVE SEARCH
# ## ------------------------------------------------------------------------------------
"""
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
"""
# Extremely time consuming step -> 140h estimated for a local run -> due to the partial correlation estimation
# Provided scripts to perform "tha Job" on the cluster -> 1.5h for cluster run
# Achieving efficiency through parallel pipeline analysis.
# Export file to the local output directory to continue the analysis

"""
from pipeline import objective_func_reg   # -> imitation of Daflon approach -> predict age
from pipeline import search_ehaustive_reg
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

key = 'MDS'
ModelEmbeddings = pickle.load(open(str(output_path + "/" + "embeddings_.p"), "rb"))
ModelEmbedding = ModelEmbeddings[key]
linear_acc = np.zeros((len(Data_Run)))
forest_acc = np.zeros((len(Data_Run)))
multiverse_section2 = list(np.zeros((len(Data_Run))))
AgesPrediction = np.asanyarray(data_space["b_age"])

for i in tqdm(range(len(Data_Run))):
    scores_tree, scores_linear, pipeline_result = search_ehaustive_reg(i, AgesPrediction, Sparsities_Run, Data_Run, BCT_models, BCT_Run,
                                    Negative_Run, Weight_Run, Connectivity_Run, data_predict)
    linear_acc[i] = scores_linear
    forest_acc[i] = scores_tree
    multiverse_section2[i] = pipeline_result

# Dump accuracies
# todo change pickle.dump if runned again
pickle.dump(linear_acc, open(str(output_path + "/" + 'predictedAcc_linear' + key + '.pckl'), 'wb'))
pickle.dump(forest_acc, open(str(output_path + "/" + 'predictedAcc_forest' + key + '.pckl'), 'wb'))
pickle.dump(multiverse_section2, open(str(output_path + "/" + 'exhaustive_search_results' + key + '.pckl'), 'wb'))
"""
# %% ------------------------------------------------------------------------------------
# ##                        EXHAUSTIVE SEARCH EXPLORATION
# ## ------------------------------------------------------------------------------------
# within pipeline x, how accurate can the prediction of graph-measure y in area z be?
pipelines = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))
ModelEmbeddings = pickle.load(open(str(output_path + "/" + "embeddings_.p"), "rb"))
# read all the ROIs:
areas = [key for key in data_predict["ts"][0].keys()]
# select one area to plot:
area_z = 17
# select the embedding to plot:
key = 'MDS'
ModelEmbedding = ModelEmbeddings[key]

# importing the prediction accuracies for each region:
elastic = pickle.load(open(str(output_path + "/" + 'ElasticNet_corr_acc.p'), 'rb'))
ridge = pickle.load(open(str(output_path + "/" + 'Ridge_corr_acc.p'), 'rb'))
lasso = pickle.load(open(str(output_path + "/" + 'Lasso_corr_acc.p'), 'rb'))
linear = pickle.load(open(str(output_path + "/" + 'linear_corr_acc.p'), 'rb'))
forest = pickle.load(open(str(output_path + "/" + 'forest_corr_acc.p'), 'rb'))

MCM_r = np.array(ridge)[:,area_z]
MCM_e = np.array(elastic)[:,area_z]
MCM_l = np.array(lasso)[:,area_z]
MCM_li = np.array(linear)[:,area_z]
MCM_f = np.array(forest)[:,area_z]

# plotting the results
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0,0].set_title('Region ' + str(areas[area_z]), fontweight='bold', fontsize=40, bbox={'facecolor': 'white', 'edgecolor': 'black', 'pad': 10}, loc = 'left')
axs[0,0].axis('off')
axs[0, 1].scatter(ModelEmbedding[0: MCM_e.shape[0], 0],
            ModelEmbedding[0: MCM_e.shape[0], 1],
            c=MCM_e, cmap='bwr')
axs[0, 1].set_title('ElasticNet Regression')
axs[0, 1].axis('off')
axs[0, 2].scatter(ModelEmbedding[0: MCM_l.shape[0], 0],
            ModelEmbedding[0: MCM_l.shape[0], 1],
            c=MCM_l, cmap='bwr')
axs[0, 2].set_title('Lasso Regression')
axs[0, 2].axis('off')
axs[1, 0].scatter(ModelEmbedding[0: MCM_li.shape[0], 0],
            ModelEmbedding[0: MCM_li.shape[0], 1],
            c=MCM_li, cmap='bwr')
axs[1, 0].set_title('Linear Regression')
axs[1, 0].axis('off')
axs[1, 1].scatter(ModelEmbedding[0: MCM_f.shape[0], 0],
            ModelEmbedding[0: MCM_f.shape[0], 1],
            c=MCM_f, cmap='bwr')
axs[1, 1].set_title('Forest Regression')
axs[1, 1].axis('off')
axs[1, 2].axis('off')
axs[1, 2].scatter(ModelEmbedding[0: MCM_r.shape[0], 0],
            ModelEmbedding[0: MCM_r.shape[0], 1],
            c=MCM_r, cmap='bwr')
axs[1, 2].set_title('Ridge Regression')


# save the figure locally
plt.savefig(str(output_path + "/" + 'regression_models.png'))
plt.show()

#%%
key = 'MDS'
pipelines_gm = pipelines["199_subjects_1152_pipelines"]
pipelines_gm = pipelines["pipeline_choices"]
accs = np.mean(np.array(linear), axis = 1)
ModelEmbedding = ModelEmbeddings[key]

plt.scatter(ModelEmbedding[0: accs.shape[0], 0],
            ModelEmbedding[0: accs.shape[0], 1],
            c=accs, cmap='bwr')
plt.colorbar()

# %% ------------------------------------------------------------------------------------
# ##                        EXHAUSTIVE SEARCH EXPLORATION
# ## ------------------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

storage = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["199_subjects_1152_pipelines"]
ROIs = list(data_predict["ts"][0].keys())
pipeline_n = 998
regional_r2 = []

# Load your data and set up your variables
x = np.asanyarray(data_predict["b_age"])
y = np.asanyarray(storage[pipeline_n])

# Sort the data
sort_idx = np.argsort(x)
x = x[sort_idx]
y = y[sort_idx]

# Define the intervals and spline model
intervals = [30, 35, 38]
for i, ROI in enumerate(ROIs):
    spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=1)

    # Calculate the R-squared value
    y_pred = spline_model(x)
    r2 = r2_score(y[:, i], y_pred)
    regional_r2.append(r2)

#  -------------- Plotting the spline model for ROI[i] --------------
# Generate the x values for visualization
xs = np.linspace(x.min(), x.max(), 1000)
i = 40
# Plot the data points, spline fit, and intervals
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-whitegrid')
plt.plot(x, y[:, i], 'ro', ms=8, label='Data', alpha=0.5)
plt.plot(xs, spline_model(xs), 'g-', lw=3, label='Spline Fit')
for interval in intervals:
    plt.axvline(x=interval, color='b', linestyle='--', alpha=0.7, label='Intervals' if interval == intervals[0] else '')
plt.legend(loc='upper left', fontsize=12)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.title('Spline Regression with Intervals', fontsize=16)
plt.tick_params(labelsize=12)
plt.tight_layout()
plt.show()


# %%
# Set up the plot style
plt.figure(figsize=(8, 6))
plt.style.use('seaborn-whitegrid')
plt.hist(regional_r2, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('R-squared', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of regional R-squared', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(labelsize=12)
plt.legend(['Distribution of R2'], fontsize=12)
plt.tight_layout()
plt.show()
# %%
