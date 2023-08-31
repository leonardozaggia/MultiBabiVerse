import numpy as np
import pickle
import sys
import os
import multiprocessing as mp
from threading import Thread

key = str(sys.argv[1])
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

def get_forest_acc():
    forest_acc = []
    files = os.listdir(PredictAccs_linear_path)
    for i, _ in enumerate(files):
        forest_acc.append(pickle.load(open(str(PredictAccs_tree_path + "/" + str(i) + ".p"), "rb")))
    pickle.dump(forest_acc, open(str(Exhaustive_path + "/" + 'forest_corr_acc.p'), 'wb'))    

def get_linear_acc():
    linear_acc = []
    files = os.listdir(PredictAccs_linear_path)
    for i, _ in enumerate(files):
        linear_acc.append(pickle.load(open(str(PredictAccs_linear_path + "/" + str(i) + ".p"), "rb")))
    pickle.dump(linear_acc, open(str(Exhaustive_path + "/" + 'linear_corr_acc.p'), 'wb'))     

def get_Ridge_acc():
    Ridge_acc = []
    files = os.listdir(PredictAccs_linear_path)
    for i, _ in enumerate(files):
        Ridge_acc.append(pickle.load(open(str(PredictAccs_Ridge_path + "/" + str(i) + ".p"), "rb")))
    pickle.dump(Ridge_acc, open(str(Exhaustive_path + "/" + 'Ridge_corr_acc.p'), 'wb'))

def get_Lasso_acc():
    Lasso_acc = []
    files = os.listdir(PredictAccs_linear_path)
    for i, _ in enumerate(files):
        Lasso_acc.append(pickle.load(open(str(PredictAccs_Lasso_path + "/" + str(i) + ".p"), "rb")))
    pickle.dump(Lasso_acc, open(str(Exhaustive_path + "/" + 'Lasso_corr_acc.p'), 'wb'))

def get_ElasticNet_acc():
    ElasticNet_acc = []
    files = os.listdir(PredictAccs_linear_path)
    for i, _ in enumerate(files):
        ElasticNet_acc.append(pickle.load(open(str(PredictAccs_ElasticNet_path + "/" + str(i) + ".p"), "rb")))
    pickle.dump(ElasticNet_acc, open(str(Exhaustive_path + "/" + 'ElasticNet_corr_acc.p'), 'wb'))

t2 = Thread(target = get_linear_acc)
t2.start()
t3 = Thread(target = get_forest_acc)
t3.start()
t4 = Thread(target = get_Ridge_acc)
t4.start()
t5 = Thread(target = get_Lasso_acc)
t5.start()
t6 = Thread(target = get_ElasticNet_acc)
t6.start()