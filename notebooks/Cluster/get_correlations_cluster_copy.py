import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from pipeline import get_data, pair_age, split_data
import sys
import pickle
from sklearn.model_selection import train_test_split

sl = []
st = []
key = "MDS"
path = "/gss/work/head3827/root_dir/data/pipeline_timeseries"
age_path = "/gss/work/head3827/root_dir/data/combined.tsv"
output_path = "/gss/work/head3827/root_dir/outputs"
Exhaustive_path = output_path + "/Exhaustive/" + key

PredictAccs_linear_path = Exhaustive_path + "/PredictAccs_linear"
PredictAccs_tree_path = Exhaustive_path + "/PredictAccs_tree"
PredictAccs_Ridge_path = Exhaustive_path + "/PredictAccs_Ridge"
PredictAccs_Lasso_path = Exhaustive_path + "/PredictAccs_Lasso"
PredictAccs_ElasticNet_path = Exhaustive_path + "/PredictAccs_ElasticNet"

Multi_path = Exhaustive_path + "/MultiversePipelines"
path = "/gss/work/head3827/root_dir/data/pipeline_timeseries"
age_path = "/gss/work/head3827/root_dir/data/combined.tsv"
output_path = "/gss/work/head3827/root_dir/outputs"
data_total = get_data(path = path)
data = pair_age(data_total, age_path, clean=True)       # clean parameter removes uncomplete data
data_space, data_predict, data_lockbox = split_data(data = data,
                                                    SUBJECT_COUNT_SPACE = 51,
                                                    SUBJECT_COUNT_PREDICT = 199,  
                                                    SUBJECT_COUNT_LOCKBOX = 51)


i = (int(sys.argv[1])-1)
n_ROIs = len(list(data_predict["ts"][0].keys()))
age = AgesPrediction = np.asanyarray(data_predict["b_age"])
TempResults = pickle.load(open(str(Exhaustive_path + "/" + 'exhaustive_search_results.p'), 'rb'))["199_subjects_1152_pipelines"]
age_list = list(map(lambda x: [x], age))
tmp_list = [list(y) for y in TempResults[i]]

X_train, X_test, y_train, y_test = train_test_split(age_list, tmp_list,
    test_size=.3, random_state=0)

model_linear = MultiOutputRegressor(LinearRegression())
model_tree =  MultiOutputRegressor(RandomForestRegressor())
model_Ridge = MultiOutputRegressor(Ridge())
model_Lasso = MultiOutputRegressor(Lasso())
model_ElasticNet = MultiOutputRegressor(ElasticNet())

model_linear.fit(X_train, y_train)
model_tree.fit(X_train, y_train)
model_Ridge.fit(X_train, y_train)
model_Lasso.fit(X_train, y_train)
model_ElasticNet.fit(X_train, y_train)

pred_linear = model_linear.predict(X_test)
pred_tree = model_tree.predict(X_test)
pred_Ridge = model_Ridge.predict(X_test)
pred_Lasso = model_Lasso.predict(X_test)
pred_ElasticNet = model_ElasticNet.predict(X_test)

scores_tree = np.diagonal(np.corrcoef(pred_tree, np.array(y_test), rowvar=False)[:n_ROIs, n_ROIs:])
scores_linear = np.diagonal(np.corrcoef(pred_linear, np.array(y_test), rowvar=False)[:n_ROIs, n_ROIs:])
scores_Ridge = np.diagonal(np.corrcoef(pred_Ridge, np.array(y_test), rowvar=False)[:n_ROIs, n_ROIs:])
scores_Lasso = np.diagonal(np.corrcoef(pred_Lasso, np.array(y_test), rowvar=False)[:n_ROIs, n_ROIs:])
scores_ElasticNet = np.diagonal(np.corrcoef(pred_ElasticNet, np.array(y_test), rowvar=False)[:n_ROIs, n_ROIs:])

pickle.dump(scores_linear, open(str(PredictAccs_linear_path + "/" + str(i) + '.p'), 'wb'))
pickle.dump(scores_tree, open(str(PredictAccs_tree_path + "/" + str(i) + '.p'), 'wb'))
pickle.dump(scores_Ridge, open(str(PredictAccs_Ridge_path + "/" + str(i) + '.p'), 'wb'))
pickle.dump(scores_Lasso, open(str(PredictAccs_Lasso_path + "/" + str(i) + '.p'), 'wb'))
pickle.dump(scores_ElasticNet, open(str(PredictAccs_ElasticNet_path + "/" + str(i) + '.p'), 'wb'))