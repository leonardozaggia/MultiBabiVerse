import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pipeline import get_data, pair_age
from pipeline import set_mv_forking
import pickle
import os

# define relevant paths
path = "/dss/work/head3827/MultiBabiVerse/pipeline_timeseries"
age_path = "/dss/work/head3827/MultiBabiVerse/data/combined.tsv"
output_path = "/dss/work/head3827/MultiBabiVerse/outputs"

pipes_path = output_path + "/pipes"

# lodad and path the pipelines
storage = pickle.load(open(str(output_path + "/" + 'exhaustive_search_results.p'), 'rb'))["301_subjects_936_pipelines"]
mat_corr = np.zeros((52, 936, 936))

# 52 -> number of regions
# 932 -> number of pipelines

# storage = 936 pipelines
# storage[participants] = 301
# storage[participants][ROI] = 52

for ROI in range(52):
    for pipe in range(936):
        for pipe2 in range(936):
            r = np.corrcoef(storage[pipe][:][:,ROI], storage[pipe2][:][:,ROI])[0,1]
            mat_corr[ROI, pipe, pipe2] = r

avg_mat_corr = np.mean(mat_corr, axis=0)

np.fill_diagonal(avg_mat_corr, 0)

pickle.dump(avg_mat_corr, open(str(output_path + "/" + "ModelsResults.p"), "wb" ) )


