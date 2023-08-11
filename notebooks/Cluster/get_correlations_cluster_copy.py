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
age = AgesPrediction = np.asanyarray(data_predict["b_age"])
TempResults = pickle.load(open(str(Exhaustive_path + "/" + 'exhaustive_search_results.p'), 'rb'))["199_subjects_1152_pipelines"]
age_list = list(map(lambda x: [x], age))
tmp_list = [list(y) for y in TempResults[i]]

X_train, X_test, y_train, y_test = train_test_split(age_list, tmp_list,
    test_size=.3, random_state=0)

model_linear = MultiOutputRegressor(LinearRegression())
model_tree =  MultiOutputRegressor(RandomForestRegressor())
model_linear.fit(X_train, y_train)
model_tree.fit(X_train, y_train)

pred_linear = model_linear.predict(X_test)
pred_tree = model_tree.predict(X_test)

scores_tree = np.mean(np.triu(np.corrcoef(np.array(y_test), pred_tree)))
scores_linear = np.mean(np.triu(np.corrcoef(np.array(y_test), pred_linear)))

pickle.dump(scores_linear, open(str(PredictAccs_linear_path + "/" + str(i) + '.p'), 'wb'))
pickle.dump(scores_tree, open(str(PredictAccs_tree_path + "/" + str(i) + '.p'), 'wb'))