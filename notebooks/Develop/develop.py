# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
import nibabel as nib
from nilearn.datasets import MNI152_FILE_PATH
import pandas as pd
from pipeline import get_data, pair_age
from pipeline import get_1_connectivity
from pipeline import run_mv, get_FORKs
from pipeline import fork_GSR

# To prevent the printing of a number of future warnings
# from nilearn package -> 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %% ------------------------------------------------------------------------------------
# ## TESTING CREATED FUNCTIONS
# ## ------------------------------------------------------------------------------------


# %% ------------------- DATA HANDLING -------------------
# defining relevant paths
ring_bell_path = '/Users/amnesia/Desktop/Master_Thesis/root_dir/end_processing_signal/handy-introduction-022-glbml-21786.mp3'
path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/pipeline_timeseries"
age_path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/combined.tsv"
# import time series and ID
data1 = get_data(path = path)
# check the lenghts of entries 
# x participants -> x time series
print([f"The length of {key} is {len(data1[key])}" for key in data1.keys()])
# Read TSV file and pair the gestational age to each participant
data = pair_age(data1, age_path, clean=True)

# convert to dataFrame for better visualization
df = pd.DataFrame(data)
print(df)

# %% Creating a Functional connectivity sample
corr_mets = {"covariance", "correlation", "partial correlation"}
sub_idx = np.random.randint(0,300)
sub = data["ts"][sub_idx]
con_dic = get_1_connectivity(sub, "covariance", p = True, sub_idx = sub_idx)            # Specify sub idx if wanted

# %% ------------------- MULTIVERSE RUN -------------------
neg_options, thresholds, weight_options = get_FORKs()

store = run_mv(data, neg_options, thresholds, weight_options)

warnings.filterwarnings("default", category=FutureWarning)
#store["Multiverse"][130]['GSR']['local efficiency']["correlation"]["abs"]["0.3"]["normalize"]

# %% ------------------------------------------------------------------------------------
# ## DEVELOP HARD
# ## ------------------------------------------------------------------------------------

one_time_Series = fork_GSR(data, 0)

# %%
import pickle
output_p = "/Users/amnesia/Desktop/Master_Thesis/root_dir/outputs"
ModelsResults = pickle.load(open(str(output_p + "ModelsResults.p"), "rb" ) )
print(ModelsResults.keys())

#Uncomment to release the real power of this scritp
#play_monumental_sound(ring_bell_path)
# TODO: import pikle package so to save and load variables without having to run the data every single time
# TODO: add filtering to the multiverse
# TODO: research how to implement the new sparsities
# TODO: graph measures calculation - segregation - integration
### TODO: calculate the similarities measures and build the 2 dimenional space
# TODO: YEOID research, ask Juan which atlas he used
# TODO: decrease n of thresholds >0.1

# TODO-> better way of representing the multiverse
    # - create 2 multiverse plots with different decisions
    # - create a function that prints the multiverse with selected options

# TODO:  rerun the multiverse 
    # - correct number of participants
    # - with the implemented graph measures