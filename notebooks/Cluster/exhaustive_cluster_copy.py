from pipeline import get_data, pair_age, split_data, get_FORKs
import pickle
import numpy as np
path = "/gss/work/head3827/root_dir/data/pipeline_timeseries"
age_path = "/gss/work/head3827/root_dir/data/combined.tsv"
output_path = "/gss/work/head3827/root_dir/outputs"
data_total = get_data(path = path)
data = pair_age(data_total, age_path, clean=True)       # clean parameter removes uncomplete data
data_space, data_predict, data_lockbox = split_data(data = data,
                                                    SUBJECT_COUNT_SPACE = 51,
                                                    SUBJECT_COUNT_PREDICT = 199,  
                                                    SUBJECT_COUNT_LOCKBOX = 51)
BCT_models, neg_options, thresholds, weight_options, graph_measures, connectivities = get_FORKs()

ModelsResults = pickle.load(open(str(output_path + "/" + "ModelsResults.p"), "rb" ) )
Results = ModelsResults['ResultsIndVar']
BCT_Run = ModelsResults['BCT']
Sparsities_Run = ModelsResults['Sparsities']
Data_Run = ModelsResults['Data']
preprocessings = ['GSR', 'noGSR']
Negative_Run = ModelsResults["Negatives"]           
Connectivity_Run = ModelsResults["Connectivity"]    
Weight_Run = ModelsResults["Weights"]  

#----
from pipeline import search_ehaustive_reg
from tqdm import tqdm
import warnings
#from multiprocessing import Pool
#from itertools import product
import sys
warnings.filterwarnings("ignore")

ModelEmbeddings = pickle.load(open(str(output_path + "/" + "embeddings_.p"), "rb"))
key = 'MDS'
ModelEmbedding = ModelEmbeddings[key]
PredictedAcc_l = np.zeros((len(Data_Run)))
PredictedAcc_t = np.zeros((len(Data_Run)))
multiverse_section2 = list(np.zeros((len(Data_Run))))
AgesPrediction = np.asanyarray(data_predict["b_age"])


#pool = Pool()
i = (int(sys.argv[1])-1)
tempPredAcc_t, tempPredAcc_l, pipeline_result = search_ehaustive_reg(i, AgesPrediction, Sparsities_Run, Data_Run, BCT_models, BCT_Run, Negative_Run, Weight_Run, Connectivity_Run, data_predict)
PredictedAcc_l[i] = tempPredAcc_l
PredictedAcc_t[i] = tempPredAcc_t
multiverse_section2[i] = pipeline_result

#plt.scatter(ModelEmbedding[0: PredictedAcc.shape[0], 0],
#            ModelEmbedding[0: PredictedAcc.shape[0], 1],
#            c=PredictedAcc, cmap='bwr')
#plt.colorbar()

# Dump accuracies
# todo change pickle.dump if runned again
PredictAccs_linear_path = "/gss/work/head3827/root_dir/outputs/Exhaustive/MDS/PredictAccs_linear"
PredictAccs_tree_path = "/gss/work/head3827/root_dir/outputs/Exhaustive/MDS/PredictAccs_tree"
Multi_path = "/gss/work/head3827/root_dir/outputs/Exhaustive/MDS/MultiversePipelines"

pickle.dump(PredictedAcc_l, open(str(PredictAccs_linear_path + "/" + str(i) + '.p'), 'wb'))
pickle.dump(PredictedAcc_t, open(str(PredictAccs_tree_path + "/" + str(i) + '.p'), 'wb'))
pickle.dump(multiverse_section2, open(str(Multi_path + "/" + str(i) + '.p'), 'wb'))

