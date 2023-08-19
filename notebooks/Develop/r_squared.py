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
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn.model_selection import train_test_split

signal_path = '/Users/amnesia/Desktop/Master_Thesis/root_dir/end_processing_signal/handy-introduction-022-glbml-21786.mp3'
path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/pipeline_timeseries"
age_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/combined.tsv"
output_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/outputs"

storage = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["199_subjects_1152_pipelines"]
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
    
    r2 = r2_score(X_test, pred_y)  # Calculate R-squared
    print(f"Degree {degree} - R-squared:", r2)

#%% explore pred_linear
import matplotlib.pyplot as plt
plt.scatter(y_test, pred_linear)

#scores_linear = np.mean(np.triu(np.corrcoef(np.array(y_test), pred_linear)))

#%% -------------- Alternative approach - Test/Train split --------------
"""
We are now going to run the model for each region and each pipeline.
From age we will predict region1, region2, and so on so forth independently.
Lets see an example for one single pipeline
"""
import matplotlib.pyplot as plt
pipeline_n = 678
x = np.asanyarray(data_predict["b_age"])
y = np.asanyarray(storage[pipeline_n])
print("Number of age entries " + str(y.shape))
print("Number of entries x pipeline " + str(x.shape))
histo_data = []
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)

ROIs = list(data_predict["ts"][0].keys())
for i, ROI in enumerate(ROIs):
    model = LinearRegression()
    model.fit(X_train.reshape(-1, 1), y_train[:, i])
    pred = model.predict(X_test.reshape(-1, 1))
    print("R2 for ROI " + str(ROI) + " is " + str(r2_score(y_test[:, i], pred)))
    histo_data.append(r2_score(y_test[:, i], pred))

plt.hist(histo_data, bins = 20)
plt.show()

#%% -------------- Alternative approach - NO Test/Train split -------------- 
"""
We are now going to run the model for each region and each pipeline.
From age we will predict region1, region2, and so on so forth independently.
Lets see an example for one single pipeline
"""
pipeline_n = 678
x = np.asanyarray(data_predict["b_age"])
y = np.asanyarray(storage[pipeline_n])
print("Number of age entries " + str(y.shape))
print("Number of entries x pipeline " + str(x.shape))
histo_data = []

ROIs = list(data_predict["ts"][0].keys())
for i, ROI in enumerate(ROIs):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y[:, i])
    pred = model.predict(x.reshape(-1, 1))
    print("R2 for ROI " + str(ROI) + " is " + str(r2_score(y[:, i], pred)))
    histo_data.append(r2_score(y[:, i], pred))

plt.hist(histo_data, bins = 20)

# %% -------------- Non-linear approach - NO Test/Train split -------------- 

pipeline_n = 444
x = np.asanyarray(data_predict["b_age"])
y = np.asanyarray(storage[pipeline_n])
print("Number of age entries " + str(y.shape))
print("Number of entries x pipeline " + str(x.shape))
regional_r2 = []

ROIs = list(data_predict["ts"][0].keys())
for i, ROI in enumerate(ROIs):
    model = make_pipeline(PolynomialFeatures(3), Ridge(alpha=1e-2))
    model.fit(x.reshape(-1, 1), y[:, i])
    pred = model.predict(x.reshape(-1, 1))
    print("R2 for ROI " + str(ROI) + " is " + str(r2_score(y[:, i], pred)))
    regional_r2.append(r2_score(y[:, i], pred))

plt.hist(regional_r2, bins = 20)
# TODO: add color legend to differenciate linear and non linear R2
plt.show()
# %% UNCONSTRAINED SPLINE APPROACH
# Load your data and set up your variables
pipeline_n = 444
x = np.asanyarray(data_predict["b_age"])
y = np.asanyarray(storage[pipeline_n])
ROIs = list(data_predict["ts"][0].keys())
regional_r2 = []

# Define piecewise intervals
intervals = [20, 30, 35, 38, 42.5]  # Theoretical base adjustment of the bins
                                    # Alternatives
                                    #    - np.percentile(y, [25, 50, 75])
                                    #    - estimate the bins from the data 

intervals_dict = {class_id: np.zeros((len(ROIs))) for class_id in range(1,len(intervals))}
for i, ROI in enumerate(ROIs):
    # Split the data into intervals and fit separate models for each interval
    interval_predictions = []
    cnt = 0
    for j in range(len(intervals) - 1):
        cnt += 1
        mask = (x >= intervals[j]) & (x < intervals[j + 1])
        x_interval = x[mask].reshape(-1, 1)
        y_interval = y[mask, i]
        
        model = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-2))
        model.fit(x_interval, y_interval)
        pred_interval = model.predict(x_interval)
        interval_predictions.append(pred_interval)
        intervals_dict[cnt][i] = r2_score(y_interval, pred_interval)

# Plot the results with 5 subplots
fig, axes = plt.subplots(4, 1, figsize=(8, 12))
cnt = 0
for j in range(len(intervals) - 1):
    cnt += 1
    axes[j].hist(intervals_dict[cnt], label="Data", bins=20)
    axes[j].set_title(f"Age interval: {intervals[j]}-{intervals[j + 1]}")
    axes[j].set_ylabel("Count")
    axes[j].legend()    

# set main image title
fig.suptitle("R-squared distribution across ROIs", fontsize=16)


# %% -------------- Spline approach - with constraints --------------
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as plt

# Load your data and set up your variables
pipeline_n = 444
x = np.asanyarray(data_predict["b_age"])
y = np.asanyarray(storage[pipeline_n])
regional_r2 = []

# Sort the data
sort_idx = np.argsort(x)
x = x[sort_idx]
y = y[sort_idx]

# Define the intervals and spline model
intervals = [30, 35, 38]
for i, ROI in enumerate(ROIs):
    spline_model = LSQUnivariateSpline(x, y[:, i], t=intervals, k=1)

    # Calculate the R-squared value
    y_pred = spline_model(x)
    r2 = r2_score(y[:, i], y_pred)
    regional_r2.append(r2)

#  -------------- Plotting the spline model for ROI[i] --------------
# Generate the x values for visualization
xs = np.linspace(x.min(), x.max(), 1000)

# Plot the data points, spline fit, and intervals
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-whitegrid')
plt.plot(x, y[:, i], 'ro', ms=8, label='Data', alpha=0.5)
plt.plot(xs, spline_model(xs), 'g-', lw=3, label='Spline Fit')
for interval in intervals:
    plt.axvline(x=interval, color='b', linestyle='--', alpha=0.7, label='Intervals' if interval == intervals[0] else '')
plt.legend(loc='upper left', fontsize=12)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.title('Spline Regression with Intervals', fontsize=16)
plt.tick_params(labelsize=12)
plt.tight_layout()
plt.show()


# %%
# Set up the plot style
plt.figure(figsize=(8, 6))
plt.style.use('seaborn-whitegrid')
plt.hist(regional_r2, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('R-squared', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of regional R-squared', fontsize=16)

# Adding a grid
plt.grid(True, linestyle='--', alpha=0.7)

# Adjusting tick label size
plt.tick_params(labelsize=12)

# Adding a legend (optional)
plt.legend(['Distribution of R2'], fontsize=12)

# Tight layout for better appearance
plt.tight_layout()

# Show the plot
plt.show()
# %%
