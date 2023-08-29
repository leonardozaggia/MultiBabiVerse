import numpy as np
import pickle
import sys
import os
from threading import Thread

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

def get_pipelines():
    pipelines = []
    pipe_code = []
    files = os.listdir(Multi_path)
    for i, _ in enumerate(files):        
        pipelines.append(pickle.load(open(str(Multi_path + "/" + str(i) + ".p"), "rb"))[i])
        pipe_code.append("_".join([Data_Run[i], Connectivity_Run[i], Negative_Run[i], str(Sparsities_Run[i]), Weight_Run[i], BCT_Run[i]]))
    Exhaustive_dict = {"199_subjects_1152_pipelines":pipelines, "pipeline_choices": pipe_code}
    pickle.dump(Exhaustive_dict, open(str(Exhaustive_path + "/" + 'exhaustive_search_results.p'), 'wb'))

def get_forest_acc():
    forest_acc = []
    files = os.listdir(PredictAccs_linear_path)
    for i, _ in enumerate(files):
        forest_acc.append(pickle.load(open(str(PredictAccs_tree_path + "/" + str(i) + ".p"), "rb"))[i])
    pickle.dump(forest_acc, open(str(Exhaustive_path + "/" + 'forest_acc.p'), 'wb'))    

def get_linear_acc():
    linear_acc = []
    files = os.listdir(PredictAccs_linear_path)
    for i, _ in enumerate(files):
        linear_acc.append(pickle.load(open(str(PredictAccs_linear_path + "/" + str(i) + ".p"), "rb"))[i])
    pickle.dump(linear_acc, open(str(Exhaustive_path + "/" + 'linear_acc.p'), 'wb'))     


t1 = Thread(target = get_pipelines)
t1.start()
t2 = Thread(target = get_linear_acc)
t2.start()
t3 = Thread(target = get_forest_acc)
t3.start()




