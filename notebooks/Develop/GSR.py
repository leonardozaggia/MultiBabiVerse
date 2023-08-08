#%% Imports
import nilearn as ni
import nibabel as nib
from nilearn.signal import clean
from nilearn import image
import pandas as pd
import numpy as np

# %% Exploring the nilearn funcitonalities
#fmri_img = image.load_img()



# -------------------------------------------------------------
# ONE PARTICIPANT JOB
# -------------------------------------------------------------




# %% Reading ROIs - reorganize them into dictionaries

# Read the CSV file
df = pd.read_csv('/Users/amnesia/Desktop/Master_Thesis/root_dir/data/rel3_dhcp_fmri_pipeline/sub-CC00069XX12/ses-26300/time_series/time_s_LR.csv')

# Extract the ROI names from the first row
roi_names = df.columns.tolist()

# Extract the time series data for each ROI
time_series = {}
for roi in roi_names:
    time_series[roi] = np.array(df[roi].tolist())

# %% Proceeding with the GSR

# Averaging across ROI
global_signal = df.mean(axis=1).values

# Regressing the global signal from ROI
gsr_df = pd.DataFrame()
for roi in df.columns:
    roi_data = df[roi].values
    roi_data = clean(roi_data.reshape(-1, 1), detrend=True, standardize=False, confounds=global_signal.reshape(-1, 1))
    gsr_df.loc[:,roi] = np.squeeze(roi_data)


# %%
import matplotlib.pyplot as p
p.plot(gsr_df.L_A1+59, color = "r")
p.plot(df.L_A1)
p.show()

# %%
