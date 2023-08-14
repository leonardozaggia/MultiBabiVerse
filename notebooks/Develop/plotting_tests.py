# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pipeline import get_data, pair_age, split_data
# %% ------------------- PATH HANDLING ---------------------
signal_path = '/Users/amnesia/Desktop/Master_Thesis/root_dir/end_processing_signal/handy-introduction-022-glbml-21786.mp3'
path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/pipeline_timeseries"
age_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/combined.tsv"
output_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/outputs"
# %% ------------------ DATA HANDLING ----------------------
data_total = get_data(path = path)
data = pair_age(data_total, age_path, clean=True)       # clean parameter removes uncomplete data
data_space, data_predict, data_lockbox = split_data(data = data,
                                                    SUBJECT_COUNT_SPACE = 51,
                                                    SUBJECT_COUNT_PREDICT = 199,  
                                                    SUBJECT_COUNT_LOCKBOX = 51)

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
plt.show()

# create a mesh to plot in
x_min, x_max = ModelEmbedding[:, 0].min() - 1, ModelEmbedding[:, 0].max() + 1
y_min, y_max = ModelEmbedding[:, 1].min() - 1, ModelEmbedding[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
plt.plot(xx, yy)
plt.show()

# Plot also the training points
plt.scatter(ModelEmbedding[0: linear_acc.shape[0], 0], ModelEmbedding[0: linear_acc.shape[0], 1], c=-np.log(np.abs(linear_acc)), cmap='bwr')
plt.colorbar()
plt.title('Linear Regression')
plt.show()
plt.scatter(ModelEmbedding[0: forest_acc.shape[0], 0], ModelEmbedding[0: forest_acc.shape[0], 1], c=-np.log(np.abs(forest_acc)), cmap='bwr')
plt.colorbar()
plt.title('Random Forest')
plt.show()


# %% Plotting correlational exhaustive search
"""
With the exhaustive search, we can plot the correlation between the predicted graph measure model and the real ones.
Upon visual inspection, we can see that the linear regression model leads to higher correlation values.
Nevertheless, both methods seems to identify regions of pipelines which are better able to capture the age-network association.

VARIABLES:
    - ModelEmbedding: MDS embedding of the pipelines
    - linear_corr_acc: average correlation between the predicted graph measure and the real one for each pipeline 
    - forest_corr_acc: average correlation between the predicted graph measure and the real one for each pipeline
"""

linear_corr_acc = np.asanyarray(pickle.load(open(str(output_path + "/" + "linear_corr_acc.p"), "rb")))
forest_corr_acc = np.asanyarray(pickle.load(open(str(output_path + "/" + "forest_corr_acc.p"), "rb")))

plt.scatter(ModelEmbedding[0: linear_corr_acc.shape[0], 0], ModelEmbedding[0: linear_corr_acc.shape[0], 1], c=linear_corr_acc, cmap='bwr')
plt.colorbar()
plt.title('LINEAR Regression')
plt.axis('off')
plt.show()

plt.scatter(ModelEmbedding[0: forest_corr_acc.shape[0], 0], ModelEmbedding[0: forest_corr_acc.shape[0], 1], c=forest_corr_acc, cmap='bwr')
plt.colorbar()
plt.title('FOREST Regression')
# add description to the plot
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.xticks([])
plt.yticks([])
plt.show()

# %% run following and explore
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from pipeline import get_data, pair_age, split_data
import sys
import pickle
from sklearn.model_selection import train_test_split

i = 1
age = AgesPrediction = np.asanyarray(data_predict["b_age"])
TempResults = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["199_subjects_1152_pipelines"]
age_list = list(map(lambda x: [x], age))
tmp_list = [list(y) for y in TempResults[i]]
X_train, X_test, y_train, y_test = train_test_split(age_list, tmp_list,
    test_size=.3, random_state=0)
model_linear = MultiOutputRegressor(LinearRegression())
model_linear.fit(X_train, y_train)
pred_linear = model_linear.predict(X_test)

#%% explore pred_linear
import matplotlib.pyplot as plt
plt.scatter(y_test, pred_linear)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

pred_linear[:, 0].shape
#scores_linear = np.mean(np.triu(np.corrcoef(np.array(y_test), pred_linear)))

# %% create a correlation matrix from these two vectors
import numpy as np

c = 1
vec1 = pred_linear[:,c].reshape(-1,1)
vec2 = np.array(y_test)[:,c].reshape(-1,1)
corr = np.corrcoef(vec1, vec2)

print(corr)


# %%
