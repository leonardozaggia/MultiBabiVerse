# %% Import cell
import matplotlib
import matplotlib.pyplot as plt
from nilearn import plotting
import nibabel as nib
import numpy as np

# %% Loading the paths of the NIFTI files from transformation files
p    = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/rel3_dhcp_anat_pipeline_"
p1   = (p +"/sub-CC00069XX12/ses-26300/xfm/sub-CC00069XX12_ses-26300_from-serag40wk_to-T2w_mode-image.nii.gz")
bb1  = nib.load(p1)
print(bb1.shape)
print(bb1.header.get_zooms())
print(bb1.header.get_xyzt_units())

p2   = (p +"/sub-CC00070XX05/ses-26700/xfm/sub-CC00070XX05_ses-26700_from-T2w_to-serag38wk_mode-image.nii.gz")
bb2  = nib.load(p2)
print(bb2.shape)
print(bb2.header.get_zooms())
print(bb2.header.get_xyzt_units())

# %% Printing and checking the sizes
print(bb1.shape)
print(bb2.shape)

# %% GIFTI files -> transformations
g1   = (p +"/sub-CC00070XX05/ses-26700/xfm/sub-CC00070XX05_ses-26700_hemi-left_from-native_to-dhcpSym40_dens-32k_mode-sphere.surf.gii")
g1   = nib.load(g1)
# 'GiftiImage' object has no attribute 'shape'
g1

# %% Structural data for check-up
s1   = (p+"/sub-CC00070XX05/ses-26700/anat/sub-CC00070XX05_ses-26700_desc-biasfield_T2w.nii.gz")
st1  = nib.load(s1)
st1.shape # in fact this variable has the correct shape

# %% Functional
f1 = ("/Users/amnesia/Desktop/Master_Thesis/dHCP_sampleData/scans/100000_5_func/NIFTI/sub-CC00313XX08_ses-100000_task-rest_bold.nii.gz")
f1 = nib.load(f1)
f1.shape

# %% reading subjects gestational age
import csv
p = "/Users/amnesia/Desktop/Master_Thesis/dHCP_sampleData/target_variables.csv"
subAge_w_ID = []
summy       = []
with open(p, "r") as file:
    csvreader = csv.reader(file)
    for raw in csvreader:
        subAge_w_ID.append([raw[0],raw[2]])
        summy.append(raw[2])

# Printing variable matching ID with gestational age
# print(subAge_w_ID)
# summy2 = [pollo[1] for pollo in subAge_w_ID[3:]]
summy = np.array(summy[3:])
print(summy.shape)

# %% demographics
demographics = np.delete(summy, np.where(summy == ''))
demographics = demographics.astype(float)
mymean       = np.mean(demographics)
lowest       = min(demographics)
highest      = max(demographics)
SDbruh       = np.std(demographics)

print(f"The mean age is {mymean}")
print(f"The oldest age is {highest}")
print(f"The youngest age is {lowest}")
print(f"The median age is {np.median(demographics)}")
print(f"The SD is {SDbruh}")

# %% some more exploration/ quantifications
sorty = np.sort(demographics)
print(sorty.shape)
sorty[sorty>40].shape

# %% bits of plotting distributions
plt.hist(demographics, bins=100)
plt.show()

# %% Exploring the nilearn funcitonalities
from pipeline import get_data

path = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/pipeline_timeseries"
data = get_data(path=path)
# %%
