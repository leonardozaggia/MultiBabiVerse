import numpy as np
import matplotlib.pyplot as plt
from nilearn.datasets import MNI152_FILE_PATH
import sys
import bct
from pipeline import get_data, pair_age, split_data
from pipeline import get_3_connectivity
from pipeline import search_ehaustive_reg, set_mv_forking
from pipeline import get_mv
import pickle

# define relevant paths
path = "/dss/work/head3827/MultiBabiVerse/pipeline_timeseries"
age_path = "/dss/work/head3827/MultiBabiVerse/data/combined.tsv"
output_path = "/dss/work/head3827/MultiBabiVerse/outputs"
data_total = get_data(path = path)
data = pair_age(data_total, age_path, clean=True)       # clean parameter removes uncomplete data

# get necessary variables
ModelsResults = set_mv_forking(data)
####
BCT_models = {
    'local efficiency': bct.efficiency_bin,              # segregation measure
    'global efficiency': bct.efficiency_bin,            # integration measure
    }
####
Results = ModelsResults['ResultsIndVar']
BCT_Run = ModelsResults['BCT']
Sparsities_Run = ModelsResults['Sparsities']
Data_Run = ModelsResults['Data']
preprocessings = ['GSR', 'noGSR']
Negative_Run = ModelsResults["Negatives"]           
Connectivity_Run = ModelsResults["Connectivity"]    
Weight_Run = ModelsResults["Weights"]               
#### calculate singolar graph measures
pipe_idx = (int(sys.argv[1])-1)
pipe_g = get_mv(
    pipe_idx,
    Sparsities_Run, 
    Data_Run, 
    BCT_models, 
    BCT_Run, 
    Negative_Run, 
    Weight_Run, 
    Connectivity_Run, 
    data
    )

# creating a folder and saving the single pipelines with 301 participants each
pipes_path = output_path + "/pipes"
pickle.dump( pipe_g, open(str(pipes_path + "/" +  str(pipe_idx) + '.p'), 'wb'))