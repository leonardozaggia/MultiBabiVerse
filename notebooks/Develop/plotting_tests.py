import pickle
import numpy as np

# %% ------------------- PATH HANDLING ---------------------
signal_path = '/Users/amnesia/Desktop/Master_Thesis/root_dir/end_processing_signal/handy-introduction-022-glbml-21786.mp3'
path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/pipeline_timeseries"
age_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/combined.tsv"
output_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/outputs"

# %% ------------------- DATA IMPORTS ---------------------
ModelEmbeddings = pickle.load(open(str(output_path + "/" + "embeddings_.p"), "rb"))
ModelsResults = pickle.load(open(str(output_path + "/" + "ModelsResults.p"), "rb" ) )
Results = ModelsResults['ResultsIndVar']
BCT_Run = ModelsResults['BCT']
Sparsities_Run = ModelsResults['Sparsities']
Data_Run = ModelsResults['Data']
preprocessings = ['GSR', 'noGSR']
Negative_Run = ModelsResults["Negatives"]           
Connectivity_Run = ModelsResults["Connectivity"]    
Weight_Run = ModelsResults["Weights"]               
# %% ------------------- DATA PLOTTING ---------------------
key = 'MDS'
ModelEmbedding = ModelEmbeddings[key]
linear_acc = np.zeros((len(Data_Run)))
forest_acc = np.zeros((len(Data_Run)))
multiverse_section2 = list(np.zeros((len(Data_Run))))
AgesPrediction = np.asanyarray(data_space["b_age"])


res = pickle.load(open(str("/Users/amnesia/Desktop/Master_Thesis/root_dir/outputs/exhaustive_search_results.p"), "rb"))
linear_acc = np.asanyarray(pickle.load(open(str("/Users/amnesia/Desktop/Master_Thesis/root_dir/outputs/linear_acc.p"), "rb")))
forest_acc = np.asanyarray(pickle.load(open(str("/Users/amnesia/Desktop/Master_Thesis/root_dir/outputs/forest_acc.p"), "rb")))
plt.scatter(ModelEmbedding[0: linear_acc.shape[0], 0],
            ModelEmbedding[0: linear_acc.shape[0], 1],
            c=-np.log(np.abs(linear_acc)), cmap='bwr')
plt.colorbar()
