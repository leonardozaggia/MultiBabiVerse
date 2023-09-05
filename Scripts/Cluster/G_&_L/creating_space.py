import numpy as np
import matplotlib.pyplot as plt
from nilearn.datasets import MNI152_FILE_PATH
from pipeline import get_data, pair_age, split_data
from pipeline import get_3_connectivity
from pipeline import new_mv
from pipeline import get_FORKs, print_FORKs
import pickle

path = "/gss/work/head3827/root_dir/data/pipeline_timeseries"
age_path = "/gss/work/head3827/root_dir/data/combined.tsv"
output_path = "/gss/work/head3827/root_dir/outputs"
data_total = get_data(path = path)
data = pair_age(data_total, age_path, clean=True)       # clean parameter removes uncomplete data


ModelsResults = new_mv(data)
pickle.dump( ModelsResults, open(str(output_path + "/" + "ModelsResults.p"), "wb" ) )