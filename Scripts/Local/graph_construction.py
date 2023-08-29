# %% IMPORTS
import seaborn as sns
import networkx as nx
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pipeline import get_data, pair_age
from pipeline import get_1_connectivity, get_3_connectivity
from pipeline import neg_corr, fork_GSR
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics.pairwise import cosine_similarity
import bct
import pickle
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %% ------------------------------------------------------------------
# ##                Graph Construction
# ## ------------------------------------------------------------------
#    #    #    #    #    #    #    #    #    #    #    #    #    #    #    

#%% get data one time -> 
output_p = "/Users/amnesia/Desktop/Master_Thesis/root_dir/outputs"
data_p = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/pipeline_timeseries"
age_p = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/combined.tsv"
d = get_data(data_p)
d = pair_age(d, age_p, clean = True)
def ff(d,sub = 109):
    lst = []                                        # create participant list 
    sub = sub                                       # youngest 109; oldest 78, 85
    dic = d["ts"][sub]                              # one participant ROI time series
    values = list(dic.values())                     # get timeseries
    m = np.zeros((2300,52))
    m = np.array(values).reshape((len(dic), 2300))  # properly reshape
    lst.append(m.T)
    print(d["b_age"][sub])
    corr_met = "correlation"
    pollo = ConnectivityMeasure(kind = corr_met)
    c     = pollo.fit_transform(lst)
    return c



# %% BCT try-out + Multiverse with GSR and Graph



def analysis_space(BCT_Num, BCT_models, x, weight):
    if (BCT_Num == 'local efficiency' & (weight == "binarize")):
        ss = BCT_models[BCT_Num](x,1)
    if (BCT_Num == 'local efficiency' & (weight == "normalize")):
        ss = bct.efficiency_wei(x,1)
    elif BCT_Num == 'modularity (louvain)':
        ss, _ = BCT_models[BCT_Num](x, seed=2)
    elif BCT_Num== 'modularity (probtune)':
        ss, _ = BCT_models[BCT_Num](x, seed=2)
    return ss

"""
BCT_try = "local efficiency"
tresh = 0.125
connect = 'covariance'
x = get_3_connectivity(d)
x = bct.threshold_proportional(x['covariance'], tresh, copy = True)
ss = analysis_space(BCT_try, BCT_models, x)
"""

def new_mv(d):

    BCT_models       = {
        'modularity (louvain)': bct.modularity_louvain_und_sign,      # segregation measure
        'local efficiency': bct.efficiency_bin,                        # segregation measure
        #'modularity (probtune)': bct.modularity_probtune_und_sign,      # segregation measure
        #'global efficiency': nx.global_efficiency                       # integration measure
        }
    
    weight_options   = ["binarize", "normalize"]
    neg_options      = ["zero","abs", "keep"]
    thresholds       = [0.4, 0.3, 0.25, 0.2, 0.175, 0.150, 0.125, 0.1]
    thresholds        = [1]
    connectivities   = ["covariance", "correlation", "partial correlation"]
    connectivities   = ["covariance"]

    tot_sub          = len(list(d.values())[0])              # size of our data - subject dimension 
    n_ROIs           = len(list(d["ts"][0].keys()))
    
    data             = deepcopy(d)
    data["Multiverse"] = []                                  # initiate new multiverse key
    pipelines_graph  = []
    pipelines_conn   = []
    pipe_code        = []
    temp_dic         = {}
    GSR              = ["GSR"]#, "noGSR"]

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

                                for idx in range(0,2):
                                    sub  = data["ts"][idx]
                                    if G == "GSR":
                                        sub = fork_GSR(sub)
                                    f    = get_1_connectivity(sub, connectivity)
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

ModelsResults = new_mv(d)
warnings.filterwarnings("default", category=FutureWarning)
pickle.dump( ModelsResults, open(str(output_p + "/" + "ModelsResults.p"), "wb" ) )
# check dimention
# %%
#f = store["Multiverse"][7]['GSR']['local efficiency']["correlation"]["abs"]["0.3"]["normalize"]
i = 45
pipelines_conn = ModelsResults["pipelines_conn"]
# Visualize the network.
f = "pipelines_conn"[700][i]
graph = nx.from_numpy_array(np.squeeze(f))
pos = nx.spring_layout(graph, seed=42)
nx.draw(graph, pos, with_labels=False, node_color='skyblue', node_size=100, font_size=8)
plt.title("Brain Network")

degree = dict(graph.degree())                    # Calculate the degree of each node.
betweenness = nx.betweenness_centrality(graph)   # Calculate betweenness centrality.
clustering_coefficient = nx.average_clustering(graph)    # Calculate clustering coefficient.
global_efficiency = nx.global_efficiency(graph)  # Calculate global efficiency.

# Visualize the distribution of node degrees.
print(f" Graph measure: {str(global_efficiency)} - value: {global_efficiency}")

# %%
#%% continue

# get one instance of connectivity
f = ff(d, 78)
f1= get_1_connectivity    
graph = nx.from_numpy_array(np.squeeze(f))

degree = dict(graph.degree())                    # Calculate the degree of each node.
betweenness = nx.betweenness_centrality(graph)   # Calculate betweenness centrality.
clustering_coefficient = nx.average_clustering(graph)    # Calculate clustering coefficient.
global_efficiency = nx.global_efficiency(graph)  # Calculate global efficiency.

# Visualize the network.
pos = nx.spring_layout(graph, seed=42)
nx.draw(graph, pos, with_labels=False, node_color='skyblue', node_size=100, font_size=8)
plt.title("Brain Network")
plt.show()

# Visualize the distribution of node degrees.
print(f"graph measure value: {global_efficiency}")
#sns.histplot(global_efficiency, bins=20, kde=True)
#plt.xlabel("Node Degree")
#plt.ylabel("Frequency")
#plt.title("Distribution of Node Degrees")
#plt.show()

# Visualize other graph measures as needed.
betweenness = list(betweenness.values())
#sns.histplot(betweenness, bins=20, kde=True)
#plt.xlabel("betweenness")
#plt.ylabel("Frequency")
#plt.title("Distribution of betweenness")
#plt.show()

# Calculate various graph measures using bct.
# global_efficiency = bct.efficiency_wei(graph)
# Calculate other measures as needed.


