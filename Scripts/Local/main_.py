# %% IMPORTS
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline
from scipy.stats import linregress
from sklearn.metrics import r2_score
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
output_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir_G_&_L/outputs"
pipe_choices = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["pipeline_choices"]


# %%
# Using custom functions to import incompleate data
data_total = get_data(path = path)

# Pair participants to their gestational age
data = pair_age(data_total, age_path, clean=True)       # clean parameter removes uncomplete data

# Get and Display forks we utilize in this multiverse
BCT_models, neg_options, thresholds, weight_options, graph_measures, connectivities = get_FORKs()

print_FORKs()


# %% ----------------- DATA VISUALIZATION -------------------
#TODO: DISPLAY THE DEMOGRAPHICS OF THE DATA
# Set up the plot style
Boundaries = [28, 31, 37]
plt.figure(figsize=(8, 6))
plt.style.use('seaborn-whitegrid')
plt.hist(np.asanyarray(data["b_age"]), bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.legend(['Infants'], fontsize=12)
for interval in Boundaries:
    plt.axvline(x=interval, color='b', linestyle='--', alpha=0.7)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Age distribution', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(labelsize=12)
plt.tight_layout()
plt.show()

# Print relevant information about the data
print("The lowest gestational age is: " + str(np.min(data["b_age"])))
print("The highest gestational age is: " + str(np.max(data["b_age"])))
print("The mean gestational age is: " + str(np.mean(data["b_age"])))
print("The standard deviation of the gestational age is: " + str(np.std(data["b_age"])))
print()



# Number of participants within each interval
data_b_age = np.array(data["b_age"])
intervals = [24, 28, 31, 37, 43]
preterm = [" Extremely Preterm ", " Very Preterm ", " Preterm ", " Full term "]
for i in range(len(intervals) - 1):
    condition = (data_b_age >= intervals[i]) & (data_b_age < intervals[i + 1])
    print("The number of" + preterm[i] + "(between " + str(intervals[i]) + " and " + str(intervals[i+1]) + ") is: " + str(len(data_b_age[condition])))


# %% Print the 3 different connectivities for selected participant
# Infant with lowest Gestational Age: 109
# Infant with highest Gestational Age: 85
sub_idx = np.random.randint(0,300)
connectivities = get_3_connectivity(data = data, plot = True, sub_idx = sub_idx)

# %% ----------------- CREATE THE SPACE -------------------
# This process is time consuming takes approximately 10h to be completed

"""
This step is performed in the cluster.
The code is in Scripts/Cluster/Global_vs_local__RosaHpc/creating_space.py
In the above script, we calculate the graph measures of all participants for all pipelines.
We then save the similarities between the pipelines in ModelsResults and plot them, 
which will allow us to gain insight into the uniqueness of the pipelines.
"""

# Pickled data is provided -> for time efficient exploration of this project
ModelsResults = pickle.load(open(str(output_path + "/" + "ModelsResults.p"), "rb" ) )
print(ModelsResults.keys())


""" Second similarity metrix to prevent errors with global efficiency:

storage = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["301_subjects_936_pipelines"]
mat_corr = np.zeros((52, 936, 936))

# 52 -> number of regions
# 932 -> number of pipelines

# storage = 936 pipelines
# storage[participants] = 301
# storage[participants][ROI] = 52

for ROI in range(52):
    for pipe in range(936):
        for pipe2 in range(936):
            r = np.corrcoef(storage[pipe][:][:,ROI], storage[pipe2][:][:,ROI])[0,1]
            mat_corr[ROI, pipe, pipe2] = r

avg_mat_corr = np.mean(mat_corr, axis=0)

np.fill_diagonal(avg_mat_corr, 0)
"""
second_similarity = pickle.load(open(str(output_path + "/" + "second_similarity.p"), "rb" ) )
second_similarity[np.isnan(second_similarity)] = 0

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
from umap import UMAP
import phate
from sklearn.decomposition import PCA

# Load the previous results
Results = ModelsResults['ResultsIndVar']
#Results = second_similarity
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

colourmaps = {"noGSR": "Oranges", "GSR": "Purples"}                                                   # Preprocessings
markers    = ["v", "s", "o", "*", "D", "1", "x", "p", "H", "+", "|", "_", "3", "^", "4", "<", "X"]    # Graph Measures
hatches    = {"partial correlation": "--", "correlation": "||", "covariance": "**"}                   # Connectivities
widths     = {"abs": 10, "keep": 0, "zero":  5}                                                                              # Binarize or Weighted                                                                                       #
sizes      = {"normalize": 80, "binarize": 250}  

Connectivities = np.array(list(Connectivity_Run.items()))[:, 1]
Sparsities     = np.array(list(Sparsities_Run.items()))[:, 1]
Negatives      = np.array(list(Negative_Run.items()))[:, 1]
Weights        = np.array(list(Weight_Run.items()))[:, 1]           
Data           = np.array(list(Data_Run.items()))[:, 1]
BCT            = np.array(list(BCT_Run.items()))[:, 1]

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
                        ### Distinguish between Quinones' Multiverse and the rest of it (different edgecolor)
                        if weight == "binarize" and connect == "correlation" and negatives == "abs" and preprocessing == "noGSR":
                            axs[idx_method].scatter(YTemp[:, 0][(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                                    YTemp[:, 1][(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                                    c=SparsitiesTemp[(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                                    marker=markers[idx_bct],
                                                    hatch=hatches[connect],
                                                    alpha=0.5,
                                                    linewidth= widths[negatives],
                                                    edgecolor='green', # Show Quinones' pipelines in green
                                                    cmap=colourmaps[preprocessing], s=sizes[weight])
                        else:
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

IntensityPatch1 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.65',
                                 alpha=1)
IntensityPatch2 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.3',
                                 alpha=0.4)
IntensityPatch3 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.05',
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
                     Lines[0], Lines[1], BlankLine,
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


colourmaps = {"global efficiency": "Oranges", "local efficiency": "Purples"}                                                   # Preprocessings
markers    = ["v", "s", "o", "*", "D", "1", "x", "p", "H", "+", "|", "_", "3", "^", "4", "<", "X"]    # Graph Measures
hatches    = {"partial correlation": "--", "correlation": "||", "covariance": "**"}                   # Connectivities
widths     = {"abs": 10, "keep": 0, "zero":  5}                                                                              # Binarize or Weighted                                                                                       #
sizes      = {"normalize": 80, "binarize": 250}  

# Do the same as above but for chosen embedding
Y = data_reduced[key]

figMDS = plt.figure(constrained_layout=False, figsize=(21, 15))
gsMDS = figMDS.add_gridspec(nrows=15, ncols=20)
axs = figMDS.add_subplot(gsMDS[:, 0:15])
idx_method = 0

Lines = {}
HatchPatch = {}
for idx_pp, preprocessing in enumerate(preprocessings):
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
                    # Distinguish between Juan's Multiverse and the rest of it (different edgecolor)
                    if weight == "binarize" and connect == "correlation" and negatives == "abs" and preprocessing == "noGSR":
                        axs.scatter(YTemp[:, 0][(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                                YTemp[:, 1][(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                                c=SparsitiesTemp[(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                                marker=markers[idx_pp],
                                                hatch=hatches[connect],
                                                alpha=0.5,
                                                linewidth= 15,
                                                edgecolor='green', # Show JUAN pipelines in green
                                                cmap=colourmaps[bct_model], s=sizes[weight])
                    else:
                        axs.scatter(YTemp[:, 0][(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                                YTemp[:, 1][(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                                c=SparsitiesTemp[(BCTTemp == bct_model) & (WeightsTemp == weight) & (ConnectivitiesTemp == connect) & (NegativesTemp == negatives)],
                                                marker=markers[idx_pp],
                                                hatch=hatches[connect],
                                                alpha=0.5,
                                                linewidth= widths[negatives],
                                                cmap=colourmaps[bct_model], s=sizes[weight])
                        
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


OrangePatch = mpatches.Patch(color='orange', label='global efficiency')
PurplePatch = mpatches.Patch(color='purple', label='local efficiency')

OrangePatch = mpatches.Patch(color='orange', label='global efficiency')
PurplePatch = mpatches.Patch(color=[85 / 255, 3 / 255, 152 / 255], label='local efficiency')

IntensityPatch1 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.65',
                                 alpha=1)
IntensityPatch2 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.3',
                                 alpha=0.4)
IntensityPatch3 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.05',
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

Lines[0] = mlines.Line2D([], [], color='black', linestyle='None',
                                            marker=markers[0], markersize=10,
                                            label=preprocessings[0])

Lines[1] = mlines.Line2D([], [], color='black', linestyle='None',
                                            marker=markers[1], markersize=10,
                                            label=preprocessings[1])

BlankLine = mlines.Line2D([], [], linestyle='None')
figMDS.legend(handles=[OrangePatch, PurplePatch, BlankLine, IntensityPatch1,
                     IntensityPatch2, IntensityPatch3, BlankLine,
                     Lines[0], Lines[1], BlankLine,
                     HatchPatch[0], HatchPatch[1], HatchPatch[2], BlankLine,
                     SizeLines1, SizeLines2, BlankLine,
                     LineWidth1, LineWidth2, LineWidth3], fontsize=15,
              frameon=False, bbox_to_anchor=(1.4, 0.8), bbox_transform=axs.transAxes)

figMDS.savefig(str(output_path + "/" + key + 'Space.png'), dpi=300)
figMDS.savefig(str(output_path + "/" +  key + 'Space.svg'), format="svg")
figMDS.show()
pickle.dump(data_reduced, open(str(output_path + "/" + "embeddings_.p"), "wb" ) )
"""
"""
# %% ------------------------------------------------------------------------------------
# ##                             EXHAUSTIVE SEARCH
# ## ------------------------------------------------------------------------------------

"""
Operation performed on the cluster due to time constraints.
Use the output from the cluster to explore the results.
"""

# %% ------------------------------------------------------------------------------------
# ##                        EXHAUSTIVE SEARCH EXPLORATION
# ## ------------------------------------------------------------------------------------

storage = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["301_subjects_936_pipelines"]
pipe_choices = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["pipeline_choices"]
ROIs = list(data["ts"][0].keys())
"""
Best pipeline for each Multiverse
Linear Regression = 390
Spline Regression = 207 + 233
"""
pipeline_n = 233
regional_r2 = []

# Load your data and set up your variables
x = np.asanyarray(data["b_age"])
y = np.asanyarray(storage[pipeline_n])

# Sort the data
sort_idx = np.argsort(x)
x = x[sort_idx]
y = y[sort_idx]

# Define the intervals and spline model
intervals = [28, 31, 37]
for i, ROI in enumerate(ROIs):
    spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=1)

    # Calculate the R-squared value
    y_pred = spline_model(x)
    r2 = r2_score(y[:, i], y_pred)
    regional_r2.append(r2)

#  -------------- Plotting the spline model for ROI[i] --------------
# Generate the x values for visualization
xs = np.linspace(x.min(), x.max(), 1000)
i = 41
# Run one more iteration of the model to get the final y_pred
spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=1)
# Calculate the R-squared value
y_pred = spline_model(x)
r2 = r2_score(y[:, i], y_pred)
# Plot the data points, spline fit, and intervals
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-whitegrid')
plt.plot(x, y[:, i], 'ro', ms=8, label='Data', alpha=0.5)
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
plt.legend(['Distribution of R²'], fontsize=12)
plt.tight_layout()
plt.show()


# %% ------------------------------------------------------------------------------------
# ##                        RERUN - ALL SUBJECTS
# ## ------------------------------------------------------------------------------------
# %% K = 1
accs_1 = np.zeros(936)
for pipeline in range(936):
    ROIs = list(data["ts"][0].keys())
    regional_r2 = []

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline])

    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Define the intervals and spline model
    intervals = [28, 31, 37]
    for i, ROI in enumerate(ROIs):
        spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=1)
        # Calculate the R-squared value
        y_pred = spline_model(x)
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)
    if any(value == 1.0 for value in regional_r2):
        accs_1[pipeline] = 0.0
    else:
        accs_1[pipeline] = np.mean(regional_r2)

   
# %% K = 2
accs_2 = np.zeros(936)
for pipeline in range(936):
    ROIs = list(data["ts"][0].keys())
    regional_r2 = []

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline])

    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Define the intervals and spline model
    intervals = [28, 31, 37]
    for i, ROI in enumerate(ROIs):
        spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=2)
        # Calculate the R-squared value
        y_pred = spline_model(x)
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)
    if any(value == 1.0 for value in regional_r2):
        accs_2[pipeline] = 0.0
    else:
        accs_2[pipeline] = np.mean(regional_r2)
# %% K = 3
accs_3 = np.zeros(936)
for pipeline in range(936):
    ROIs = list(data["ts"][0].keys())
    regional_r2 = []

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline])

    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Define the intervals and spline model
    intervals = [28, 31, 37]
    for i, ROI in enumerate(ROIs):
        spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=3)
        # Calculate the R-squared value
        y_pred = spline_model(x)
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)
    if any(value == 1.0 for value in regional_r2):
        accs_3[pipeline] = 0.0
    else:
        accs_3[pipeline] = np.mean(regional_r2)
# %% Linear Regression
accs_0 = np.zeros(936)
for pipeline in range(936):
    ROIs = list(data["ts"][0].keys())
    regional_r2 = []

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline])

    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Define the intervals and spline model
    intervals = [28, 31, 37]
    for i, ROI in enumerate(ROIs):
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(x.reshape(-1, 1), y[:, i])
        # Calculate the R-squared value
        y_pred = reg.predict(x.reshape(-1, 1))
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)
    if any(value == 1.0 for value in regional_r2):
        accs_0[pipeline] = 0.0
    else:
        accs_0[pipeline] = np.mean(regional_r2)
# %% Dictionary with all the results
accs_dict_301 = {"0": accs_0, "1": accs_1, "2": accs_2, "3": accs_3}
pickle.dump(accs_dict_301, open(str(output_path + "/" + "accs_dict_301_fixed.p"), "wb" ) )
# %% ------------------------------------------------------------------------------------
# ##                        RERUN - 150 SUBJECTS
# ## ------------------------------------------------------------------------------------
# %% Accuracies for 150 subjects
# K = 1
accs_1 = np.zeros(936)
for pipeline in range(936):
    ROIs = list(data["ts"][0].keys())
    regional_r2 = []

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline])

    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    x = x[:150]
    y = y[:150]

    # Define the intervals and spline model
    intervals = [28, 31, 37]
    for i, ROI in enumerate(ROIs):
        spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=1)
        # Calculate the R-squared value
        y_pred = spline_model(x)
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)
    if any(value == 1.0 for value in regional_r2):
        accs_1[pipeline] = 0.0
    else:
        accs_1[pipeline] = np.mean(regional_r2)
# K = 2
accs_2 = np.zeros(936)
for pipeline in range(936):
    ROIs = list(data["ts"][0].keys())
    regional_r2 = []

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline])

    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    x = x[:150]
    y = y[:150]

    # Define the intervals and spline model
    intervals = [28, 31, 37]
    for i, ROI in enumerate(ROIs):
        spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=2)
        # Calculate the R-squared value
        y_pred = spline_model(x)
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)
    if any(value == 1.0 for value in regional_r2):
        accs_2[pipeline] = 0.0
    else:
        accs_2[pipeline] = np.mean(regional_r2)
# K = 3
accs_3 = np.zeros(936)
for pipeline in range(936):
    ROIs = list(data["ts"][0].keys())
    regional_r2 = []

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline])

    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    x = x[:150]
    y = y[:150]

    # Define the intervals and spline model
    intervals = [28, 31, 37]
    for i, ROI in enumerate(ROIs):
        spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=3)
        # Calculate the R-squared value
        y_pred = spline_model(x)
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)
    if any(value == 1.0 for value in regional_r2):
        accs_3[pipeline] = 0.0
    else:
        accs_3[pipeline] = np.mean(regional_r2)
# Linear Regression
accs_0 = np.zeros(936)
for pipeline in range(936):
    ROIs = list(data["ts"][0].keys())
    regional_r2 = []

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline])

    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    x = x[:150]
    y = y[:150]

    # Define the intervals and spline model
    intervals = [28, 31, 37]
    for i, ROI in enumerate(ROIs):
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(x.reshape(-1, 1), y[:, i])
        # Calculate the R-squared value
        y_pred = reg.predict(x.reshape(-1, 1))
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)
    if any(value == 1.0 for value in regional_r2):
        accs_0[pipeline] = 0.0
    else:
        accs_0[pipeline] = np.mean(regional_r2)
# Dictionary with all the results
accs_dict_150 = {"0": accs_0, "1": accs_1, "2": accs_2, "3": accs_3}
pickle.dump(accs_dict_150, open(str(output_path + "/" + "accs_dict_150_fixed.p"), "wb" ) )
# %% ------------------------------------------------------------------------------------
# ##                        RERUN - 95 SUBJECTS
# ## ------------------------------------------------------------------------------------
# %% Accuracies for 95 subjects 
# K = 1
accs_1 = np.zeros(936)
for pipeline in range(936):
    ROIs = list(data["ts"][0].keys())
    regional_r2 = []

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline])

    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    x = x[:95]
    y = y[:95]

    # Define the intervals and spline model
    intervals = [28, 31, 37]
    for i, ROI in enumerate(ROIs):
        spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=1)
        # Calculate the R-squared value
        y_pred = spline_model(x)
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)
    if any(value == 1.0 for value in regional_r2):
        accs_1[pipeline] = 0.0
    else:
        accs_1[pipeline] = np.mean(regional_r2)
# K = 2
accs_2 = np.zeros(936)
for pipeline in range(936):
    ROIs = list(data["ts"][0].keys())
    regional_r2 = []

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline])

    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    x = x[:95]
    y = y[:95]

    # Define the intervals and spline model
    intervals = [28, 31, 37]
    for i, ROI in enumerate(ROIs):
        spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=2)
        # Calculate the R-squared value
        y_pred = spline_model(x)
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)
    if any(value == 1.0 for value in regional_r2):
        accs_2[pipeline] = 0.0
    else:
        accs_2[pipeline] = np.mean(regional_r2)
# K = 3
accs_3 = np.zeros(936)
for pipeline in range(936):
    ROIs = list(data["ts"][0].keys())
    regional_r2 = []

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline])

    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    x = x[:95]
    y = y[:95]

    # Define the intervals and spline model
    intervals = [28, 31, 37]
    for i, ROI in enumerate(ROIs):
        spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=3)
        # Calculate the R-squared value
        y_pred = spline_model(x)
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)
    if any(value == 1.0 for value in regional_r2):
        accs_3[pipeline] = 0.0
    else:
        accs_3[pipeline] = np.mean(regional_r2)
# Linear Regression
accs_0 = np.zeros(936)
for pipeline in range(936):
    ROIs = list(data["ts"][0].keys())
    regional_r2 = []

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline])

    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    x = x[:95]
    y = y[:95]

    # Define the intervals and spline model
    intervals = [28, 31, 37]
    for i, ROI in enumerate(ROIs):
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(x.reshape(-1, 1), y[:, i])
        # Calculate the R-squared value
        y_pred = reg.predict(x.reshape(-1, 1))
        r2 = r2_score(y[:, i], y_pred)
        regional_r2.append(r2)
    if any(value == 1.0 for value in regional_r2):
        accs_0[pipeline] = 0.0
    else:
        accs_0[pipeline] = np.mean(regional_r2)
# Dictionary with all the results
accs_dict_95 = {"0": accs_0, "1": accs_1, "2": accs_2, "3": accs_3} 
pickle.dump(accs_dict_95, open(str(output_path + "/" + "accs_dict_95_fixed.p"), "wb" ) )
# %% ------------------------------------------------------------------------------------
# ##                        EXHAUSTIVE SEARCH EXPLORATION
# ## ------------------------------------------------------------------------------------
# %% Plotting R2 for all pipelines - spline k = 1
key = 't-SNE'
ModelEmbedding = data_reduced[key]

# %% ------------------------------------------------------------------------------------
# ##                        PLOT ALTERNATIVE 2
# ## ------------------------------------------------------------------------------------

# Set a custom color palette
sns.set_palette("coolwarm")

# Create a figure with 2x2 subplots and adjust spacing
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.subplots_adjust(hspace=0.2, wspace=0.2)  # Adjust spacing

# Titles for subplots
titles = ['Linear Regression', 'k = 1', 'k = 2', 'k = 3']

# Create a list of k values and corresponding accuracy arrays
k_values = [0, 1, 2, 3]
accs_list = [accs_0, accs_1, accs_2, accs_3]
vmax = np.max(np.hstack([accs_3,accs_2,accs_1,accs_0]))

# Define custom colors for subplots
colors = ['red', 'blue', 'green', 'purple']

# Iterate through subplots and corresponding data
for i, ax in enumerate(axs.flat):
    if i < len(k_values):
        k = k_values[i]
        accs = accs_list[i]

        # Create a scatter plot with custom color
        scatter = ax.scatter(
            ModelEmbedding[0: accs.shape[0], 0],
            ModelEmbedding[0: accs.shape[0], 1],
            c=accs,
            cmap='coolwarm',
            vmax=vmax,
            vmin=0,
        )

        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('R²', fontsize=12)

        # Title with custom color and font size
        ax.set_title(titles[i], fontsize=16, color=colors[i])
        ax.set_xlabel(f'Max R²: {np.max(accs)}', fontsize=12)
        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])

# Save the figure locally
# plt.savefig(str(output_path + "/" + 'regression_models.png'))

# Show the plot
plt.show()

# %% ------------------------------------------------------------------------------------
# ##                        PLOT SINGLE MULTIVERSE
# ## ------------------------------------------------------------------------------------
"""
Plotting a single multiverse, using linear regression or spline k = 1, 2, 3
"""

# Set a custom color palette
sns.set_palette("coolwarm")

# Choose a specific value of k
k = 1
accs = accs_list[k]  # Assuming you have accs_list defined somewhere

# Create a figure and adjust size
fig, ax = plt.subplots(figsize=(10, 8))

# Create a scatter plot with custom color
scatter = ax.scatter(
    ModelEmbedding[0: accs.shape[0], 0],
    ModelEmbedding[0: accs.shape[0], 1],
    c=accs,
    cmap='coolwarm',
    vmax=vmax,
    vmin=0,
)

# Add a vertical colorbar with the label on top
cbar = plt.colorbar(scatter, shrink = 0.7, orientation='vertical', pad=0.05)

# Adjust the label position to be on top
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.labelpad = 20

cbar.set_label('R2', fontsize=12)  # Adjust label position with labelpad

# Title with custom color and font size
ax.set_title(f'k = {k}', fontsize=16, color='blue')
ax.set_xlabel(f'Max R²: {np.max(accs)}', fontsize=12)
ax.set_xticks([])  # Remove x-axis ticks
ax.set_yticks([])  # Remove y-axis ticks

plt.savefig(str('/Users/amnesia/Desktop/Master_Thesis' + "/" + 'single_multiverse_tsne.png'))
# Show the plot
plt.show()


# %% ------------------------------------------------------------------------------------
# ##                        PLOT DISTRIBUTION OF R2
# ## ------------------------------------------------------------------------------------
"""
Plotting the distribution of each of the accs arrays
In this way we can visualize and compare how model flexibility affects the distribution of R2
"""
# Set a custom color palette
sns.set_palette("coolwarm")

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(10, 5))

# Title for the plot
ax.set_title('R² Distribution by model flexibility', fontsize=16)

# Create a list of k values and corresponding accuracy arrays
k_values = ["linear", 1, 2, 3]
legend = ['Linear Regression', 'Spline k = 1', 'Spline k = 2', 'Spline k = 3']
accs_list = [accs_0, accs_1, accs_2, accs_3]
vmax = np.max(accs_3)

# Define custom colors for subplots
colors = ['red', 'blue', 'green', 'purple']

# Iterate through subplots and corresponding data
for i, accs in enumerate(accs_list):
    k = k_values[i]
    # Create a scatter plot with custom color (distplot will be deprecated)
    sns.histplot(accs,ax=ax, color=colors[i], label=f'{legend[i]}')
    # Remove axis labels and ticks
    ax.set_xlim([0, 0.20])

# Create a legend with custom colors and labels
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.2, 0.5), fontsize=12)
# Show the plot
plt.show()

# %% Print the index of the best pipeline from each Multiverse 
print("Linear: " + str(np.argmax(accs_0)))
print("k = 1: " + str(np.argmax(accs_1)))
print("k = 2: " + str(np.argmax(accs_2)))
print("k = 3: " + str(np.argmax(accs_3)))
# Print the forking path for the best pipeline from each Multiverse
print("The best pipeline using linear model is: " + pipe_choices[np.argmax(accs_0)].replace("_", ", "))
print("The best pipeline using polinomial k = 1 is: " + pipe_choices[np.argmax(accs_1)].replace("_", ", "))
print("The best pipeline using polinomial k = 2 is: " + pipe_choices[np.argmax(accs_2)].replace("_", ", "))
print("The best pipeline using polinomial k = 3 is: " + pipe_choices[np.argmax(accs_3)].replace("_", ", "))

# %% ------------------------------------------------------------------------------------
# ##                        Significance Test Linear
# ## ------------------------------------------------------------------------------------
# Linear Model Significance
# Imports


accs_dict_301 = pickle.load(open(str(output_path + "/" + "accs_dict_301.p"), "rb" ) )
accs_dict_150 = pickle.load(open(str(output_path + "/" + "accs_dict_150.p"), "rb" ) )
accs_dict_95 = pickle.load(open(str(output_path + "/" + "accs_dict_95.p"), "rb" ) )

storage = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["301_subjects_936_pipelines"]
pipe_choices = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["pipeline_choices"]
ROIs = list(data["ts"][0].keys())
histogram = []

for pipeline_n in range(len(pipe_choices)):
    print("Chosen pipeline: " + pipe_choices[pipeline_n] + " - ", pipeline_n)
    print()

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline_n])


    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Initializing relevant variables
    tmp = 0
    idx_significance = []

    for i in range(len(ROIs)):

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y[:,i])

        # Check if the association is significant (using a common alpha level of 0.05)
        if p_value < 0.05:
            tmp = 1
            idx_significance.append(i)
            histogram.append(i)

    if tmp == 1:
        print("The association between x and y was statistically significant at least once.")
        for idx in idx_significance:
            print(ROIs[idx], "-", idx)
    else:
        print("There is no significant association between x and y.")

plt.figure(figsize=(8, 6))
plt.style.use('seaborn-whitegrid')
plt.hist(histogram, bins=52, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Significant ROI', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Linear regression', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(labelsize=12)
plt.legend(['Significant associations'], fontsize=12)
plt.tight_layout()
plt.show()


# %% ePT - spline k = 1

histogram = []

for pipeline_n in range(len(pipe_choices)):
    print("Chosen pipeline: " + pipe_choices[pipeline_n] + " - ", pipeline_n)
    print()

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline_n])


    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Initializing relevant variables
    tmp = 0
    idx_significance = []

    for i in range(len(ROIs)):

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x[:7], y[:7,i])

        # Check if the association is significant (using a common alpha level of 0.05)
        if p_value < 0.05:
            tmp = 1
            idx_significance.append(i)
            histogram.append(i)

    if tmp == 1:
        print("The association between x and y was statistically significant at least once.")
        for idx in idx_significance:
            print(ROIs[idx], "-", idx)
    else:
        print("There is no significant association between x and y.")

plt.figure(figsize=(8, 6))
plt.style.use('seaborn-whitegrid')
plt.hist(histogram, bins=52, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Significant ROI', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('ePT - spline k = 1', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(labelsize=12)
plt.legend(['Significant associations'], fontsize=12)
plt.tight_layout()
plt.show()

# %% vPT - spline k = 1
histogram = []

for pipeline_n in range(len(pipe_choices)):
    print("Chosen pipeline: " + pipe_choices[pipeline_n] + " - ", pipeline_n)
    print()

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline_n])


    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Initializing relevant variables
    tmp = 0
    idx_significance = []

    for i in range(len(ROIs)):

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x[8:17], y[8:17,i])

        # Check if the association is significant (using a common alpha level of 0.05)
        if p_value < 0.05:
            tmp = 1
            idx_significance.append(i)
            histogram.append(i)

    if tmp == 1:
        print("The association between x and y was statistically significant at least once.")
        for idx in idx_significance:
            print(ROIs[idx], "-", idx)
    else:
        print("There is no significant association between x and y.")

plt.figure(figsize=(8, 6))
plt.style.use('seaborn-whitegrid')
plt.hist(histogram, bins=52, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Significant ROI', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('vPT - spline k = 1', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(labelsize=12)
plt.legend(['Significant associations'], fontsize=12)
plt.tight_layout()
plt.show()
# %% PT - spline k = 1
histogram = []

for pipeline_n in range(len(pipe_choices)):
    print("Chosen pipeline: " + pipe_choices[pipeline_n] + " - ", pipeline_n)
    print()

    # Load your data and set up your variables
    x = np.asanyarray(data["b_age"])
    y = np.asanyarray(storage[pipeline_n])


    # Sort the data
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Initializing relevant variables
    tmp = 0
    idx_significance = []

    for i in range(len(ROIs)):

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x[18:47], y[18:47,i])

        # Check if the association is significant (using a common alpha level of 0.05)
        if p_value < 0.05:
            tmp = 1
            idx_significance.append(i)
            histogram.append(i)

    if tmp == 1:
        print("The association between x and y was statistically significant at least once.")
        for idx in idx_significance:
            print(ROIs[idx], "-", idx)
    else:
        print("There is no significant association between x and y.")

plt.figure(figsize=(8, 6))
plt.style.use('seaborn-whitegrid')
plt.hist(histogram, bins=52, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Significant ROI', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('PT - spline k = 1', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(labelsize=12)
plt.legend(['Significant associations'], fontsize=12)
plt.tight_layout()
plt.show()

#%% 




# %%
