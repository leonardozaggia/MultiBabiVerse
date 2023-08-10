import numpy as np
import pickle
import sys
import os

key = str(sys.argv[1])
path = "/gss/work/head3827/root_dir/data/pipeline_timeseries"
age_path = "/gss/work/head3827/root_dir/data/combined.tsv"
output_path = "/gss/work/head3827/root_dir/outputs"
Exhaustive_path = output_path + "/Exhaustive/" + key
PredictAccs_linear_path = Exhaustive_path + "/PredictAccs_linear"
PredictAccs_tree_path = Exhaustive_path + "/PredictAccs_tree"
Multi_path = Exhaustive_path + "/MultiversePipelines"

ModelsResults = pickle.load(open(str(output_path + "/" + "ModelsResults.p"), "rb" ) )
Results = ModelsResults['ResultsIndVar']
BCT_Run = ModelsResults['BCT']
Sparsities_Run = ModelsResults['Sparsities']
Data_Run = ModelsResults['Data']
preprocessings = ['GSR', 'noGSR']
Negative_Run = ModelsResults["Negatives"]           
Connectivity_Run = ModelsResults["Connectivity"]    
Weight_Run = ModelsResults["Weights"]  

pipelines = []
pipe_code = []
files = os.listdir(Multi_path)
for i, file in enumerate(files):
                
    pipelines.append(pickle.load(open(str(Multi_path + "/" + str(i) + ".p"), "rb")))
    pipe_code.append("_".join([Data_Run[i], Connectivity_Run[i], Negative_Run[i], Sparsities_Run[i], Weight_Run[i], BCT_Run[i]]))

Exhaustive_dict = {"199_subjects_1152_pipelines":pipelines, "pipeline_choices": pipe_code}
pickle.dump(Exhaustive_dict, open(str(Exhaustive_path + "/" + 'exhaustive_search_results.p'), 'wb'))