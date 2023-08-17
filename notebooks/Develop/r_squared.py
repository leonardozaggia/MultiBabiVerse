# %% imports
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from pipeline import get_data, pair_age, split_data, get_FORKs
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys
import pickle
from sklearn.model_selection import train_test_split

signal_path = '/Users/amnesia/Desktop/Master_Thesis/root_dir/end_processing_signal/handy-introduction-022-glbml-21786.mp3'
path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/pipeline_timeseries"
age_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/combined.tsv"
output_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/outputs"

pipelines = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))
ModelsResults = pickle.load(open(str(output_path + "/" + "ModelsResults.p"), "rb" ) )
ModelEmbeddings = pickle.load(open(str(output_path + "/" + "embeddings_.p"), "rb"))
BCT_models, neg_options, thresholds, weight_options, graph_measures, connectivities = get_FORKs()

data_total = get_data(path = path)
data = pair_age(data_total, age_path, clean=True)    
data_space, data_predict, data_lockbox = split_data(data = data,
                                                    SUBJECT_COUNT_SPACE = 51,
                                                    SUBJECT_COUNT_PREDICT = 199,  
                                                    SUBJECT_COUNT_LOCKBOX = 51)

#%% first try
i = 500
age = AgesPrediction = np.asanyarray(data_predict["b_age"])
TempResults = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["199_subjects_1152_pipelines"]
age_list = list(map(lambda x: [x], age))
tmp_list = [list(y) for y in TempResults[i]]
X_train, X_test, y_train, y_test = train_test_split(age_list, tmp_list,
    test_size=.3, random_state=0)
model_linear = MultiOutputRegressor(LinearRegression())
model_linear.fit(X_train, y_train)
pred_linear = model_linear.predict(X_test)

# R^2 calculation
ROI = 2
for degree in [1, 2, 3]:
    model = MultiOutputRegressor(make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-2)))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    true_y = np.array(y_test)[:, ROI]  # Extract the target values for the ROI
    pred_y = np.array(pred)[:, ROI]  # Extract the predicted values for the ROI
    
    r2 = r2_score(true_y, pred_y)  # Calculate R-squared
    print(f"Degree {degree} - R-squared:", r2)

#%% explore pred_linear
import matplotlib.pyplot as plt
plt.scatter(y_test, pred_linear)

#scores_linear = np.mean(np.triu(np.corrcoef(np.array(y_test), pred_linear)))

#%% Alternative approach
"""
We are now going to run the model for each region and each pipeline.
From age we will predict region1, region2, and so on so forth independently.
Lets see an example for one single pipeline
"""
storage = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["199_subjects_1152_pipelines"]
pipeline_n = 444
y = np.asanyarray(data_predict["b_age"])
x = np.asanyarray(storage[pipeline_n])
print("Number of age entries " + str(y.shape))
print("Number of entries x pipeline " + str(x.shape))
histo_data = []

ROIs = list(data_predict["ts"][0].keys())
for i, ROI in enumerate(ROIs):
    model = LinearRegression()
    model.fit(y.reshape(-1, 1), x[:, i])
    pred = model.predict(y.reshape(-1, 1))
    print("R2 for ROI " + str(ROI) + " is " + str(r2_score(x[:, i], pred)))
    histo_data.append(r2_score(x[:, i], pred))

plt.hist(histo_data, bins = 20)
plt.show()

# %% Non linear approach 

storage = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["199_subjects_1152_pipelines"]
pipeline_n = 444
y = np.asanyarray(data_predict["b_age"])
x = np.asanyarray(storage[pipeline_n])
print("Number of age entries " + str(y.shape))
print("Number of entries x pipeline " + str(x.shape))
regional_r2 = []

ROIs = list(data_predict["ts"][0].keys())
for i, ROI in enumerate(ROIs):
    model = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-2))
    model.fit(y.reshape(-1, 1), x[:, i])
    pred = model.predict(y.reshape(-1, 1))
    print("R2 for ROI " + str(ROI) + " is " + str(r2_score(x[:, i], pred)))
    regional_r2.append(r2_score(x[:, i], pred))

# %% Spline approach
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Load your data and set up your variables
storage = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["199_subjects_1152_pipelines"]
pipeline_n = 444
y = np.asanyarray(data_predict["b_age"])
x = np.asanyarray(storage[pipeline_n])

ROIs = list(data_predict["ts"][0].keys())
regional_r2 = []

for i, ROI in enumerate(ROIs):
    # Define piecewise intervals
    intervals = [20, 40, 60, 80, 100]  # Adjust these intervals as needed
    
    # Split the data into intervals and fit separate models for each interval
    interval_predictions = []
    for j in range(len(intervals) - 1):
        mask = (y >= intervals[j]) & (y < intervals[j + 1])
        y_interval = y[mask].reshape(-1, 1)
        x_interval = x[mask, i]
        
        model = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-2))
        model.fit(y_interval, x_interval)
        pred_interval = model.predict(y_interval)
        interval_predictions.append(pred_interval)
    
    pred = np.concatenate(interval_predictions)
    
    print("R2 for ROI " + str(ROI) + " is " + str(r2_score(x[:, i], pred)))
    regional_r2.append(r2_score(x[:, i], pred))


# %%
