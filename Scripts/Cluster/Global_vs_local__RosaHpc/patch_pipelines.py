import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pipeline import get_data, pair_age
from pipeline import set_mv_forking
import pickle
import os

# define relevant paths
path = "/dss/work/head3827/MultiBabiVerse/pipeline_timeseries"
age_path = "/dss/work/head3827/MultiBabiVerse/data/combined.tsv"
output_path = "/dss/work/head3827/MultiBabiVerse/outputs"
Exhaustive_path = output_path + "/Exhaustive"
pipes_path = output_path + "/pipes"

# load the data
data_total = get_data(path = path)
data = pair_age(data_total, age_path, clean=True)       # clean parameter removes uncomplete data

# get necessary variables
tot_sub = len(list(data.values())[0])
ModelsResults = set_mv_forking(data)   
####
Results = ModelsResults['ResultsIndVar']
BCT_Run = ModelsResults['BCT']
Sparsities_Run = ModelsResults['Sparsities']
Data_Run = ModelsResults['Data']
preprocessings = ['GSR', 'noGSR']
Negative_Run = ModelsResults["Negatives"]           
Connectivity_Run = ModelsResults["Connectivity"]    
Weight_Run = ModelsResults["Weights"]   
####

# lodad and path the pipelines
def get_ModelsResults():
    pipelines = []
    pipe_code = []
    pipes = os.listdir(pipes_path)
    for i, _ in enumerate(pipes):
        pipe_g = pickle.load(open(str(pipes_path + "/" + str(i) + ".p"), "rb"))
        cos_sim = cosine_similarity(pipe_g, pipe_g)
        ModelsResults["Results"][i, :] = np.mean(pipe_g, axis=0)
        ModelsResults["ResultsIndVar"][i, :] = cos_sim[np.triu_indices(tot_sub, k=1)].T
        
        pipelines.append(pipe_g)
        pipe_code.append("_".join([Data_Run[i], Connectivity_Run[i], Negative_Run[i], str(Sparsities_Run[i]), Weight_Run[i], BCT_Run[i]]))
    
    Exhaustive_dict = {"301_subjects_936_pipelines":pipelines, "pipeline_choices": pipe_code}
    pickle.dump( ModelsResults, open(str(output_path + "/" + "ModelsResults.p"), "wb" ) )
    pickle.dump(Exhaustive_dict, open(str(output_path + "/" + 'exhaustive_search_results.p'), 'wb'))

