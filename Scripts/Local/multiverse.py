# %% IMPORTS
import seaborn as sns
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pipeline import get_data, pair_age, get_connectivity
from pipeline import get_1_connectivity
from nilearn.connectome import ConnectivityMeasure
import bct
import pandas as pd

#%% DATA FETCHING
p = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/pipeline_timeseries"
d = get_data(p)
p = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/combined.tsv"
d = pair_age(d, p, clean = True)
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


# %% ------------------------------------------------------------------
# ##                Tresh Holding + Weight/Binarization / 1 sub
# ## ------------------------------------------------------------------

sub = 1
sub = {key: value[sub] for key, value in d.items()} # get the subject data
f   = get_1_connectivity(sub)

weight_options = ["binarize", "normalize", "lengths"]
thresholds     = [0.4,  0.2, 0.175, 0.09, 0.08,
                 0.06, 0.05,  0.01]
#key = "correlation"

for key in f.keys():                                                # connect. measure FORK
    for tresh in thresholds:                                        # tresholds FORK
        f1 = bct.threshold_proportional(f[key], tresh, copy = True)
        fig, axs = plt.subplots(1,3)
        fig.suptitle(f"connectivity measure: --{key}--")

        for i, weight in enumerate(weight_options):                 # handling weights FORK
            da = bct.weight_conversion(f1, weight)
            axs[i].matshow(np.squeeze(da), cmap='coolwarm')
            axs[i].set_title(f"w: {weight}")
            
plt.show()
        


# %% ------------------------------------------------------------------
# ## Tresh Holding + Weight/Binarization + Negative Correlation / 1 sub
# ## ------------------------------------------------------------------

sub_idx = 0
f = get_1_connectivity(d, p = True, sub_idx = sub_idx)

weight_options   = ["binarize", "normalize"]
neg_corr_options = [ "abs", "zero", "keep"]
thresholds       = [0.4,  0.2, 0.09, 0.08, 0.05,  0.01]
f1               = {}

def neg_corr(option, f):   
    if  option == "abs":
        out = abs(f)
    elif option == "zero": 
        out = f
        out[np.where(f < 0)] = 0
    elif option == "keep":
        out = f
    return out

for key in f.keys():                                                # connect. measure FORK
    for opt in neg_corr_options:
        for tresh in thresholds:                                        # tresholds FORK
            f1 = bct.threshold_proportional(neg_corr(opt, f[key]), tresh, copy = True)
            fig, axs = plt.subplots(1,2)
            fig.suptitle(f"--{opt}-- : --{key}--")

            for i, weight in enumerate(weight_options):                 # handling weights FORK
                if weight == "keep":
                    da = f1
                else: 
                    da = bct.weight_conversion(f1, weight)
                axs[i].matshow(np.squeeze(da), cmap='coolwarm')
                axs[i].set_title(f"w: {weight}")
plt.show()

# %% ------------------------------------------------------------------
# ## Tresh Holding + Weight/Binarization + Negative Correlation / 301 sub
# ## ------------------------------------------------------------------


from copy import deepcopy

tot_sub          = range(0, len(list(d.values())[0]))                       # create iterable with the size of our data
d2               = deepcopy(d)                                      
d2["Multiverse"] = []                                                       # initiate new multiverse key

weight_options   = ["binarize", "normalize", "lengths"]
neg_corr_options = [ "abs", "zero", "keep"]
thresholds       = [0.4, 0.3, 0.25, 0.2, 0.175, 0.150, 0.125, 0.1]
#thresholds       = [0.4,  0.2, 0.175, 0.09, 0.08, 0.06, 0.05,  0.01]
f1               = {}

for sub in tot_sub:
    sub = {key: value[sub] for key, value in d.items()}                     # get the subject data x each entry in the dictionary
    f   = get_1_connectivity(sub)
    for key in f.keys():                                                    # connectivity measure FORK
        f1[key] = {}
        for opt in neg_corr_options:
            f1[key][opt] = {}
            for tresh in thresholds:                                        # tresholds FORK
                f1[key][opt][str(tresh)] = {}
                temp = bct.threshold_proportional(neg_corr(option = opt, f = f[key]), tresh, copy = True)
                for weight in weight_options:                               # handling weights FORK
                    f1[key][opt][str(tresh)][weight] = bct.weight_conversion(temp, weight)
    d2["Multiverse"].append(f1)











#%% handling negative values in a loop

f = get_connectivity(d)     # 301 subjects, 3 connectivities

neg_f            = {}
neg_corr_options = [ "abs", "zero", "keep"]

for opt in neg_corr_options:
    neg_f[opt] = {}
    for key in f.keys():
        neg_f[opt][key] = neg_corr(opt, f[key])

print(neg_f.keys())         # 301 subjects, 3 connectivities, 3 ways to handle neg-correlations