# %% CONNECTIVITY SCRIPT
from pipeline import get_data, pair_age
from pipeline import neg_corr
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import matplotlib.pyplot as plt

#%% data import

p = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/pipeline_timeseries"
d = get_data(p)

p = "/Users/amnesia/Desktop/Master_Thesis/root_dir/data/combined.tsv"
d = pair_age(d, p, clean = True)


# %% ------------------------------------------------------------------
# ##                One participant connectivity
# ## ------------------------------------------------------------------
#TODO: create a function for single participant connectivity



lst = []                                        # create participant list 
sub = 109                                       # youngest 109; oldest 78, 85
dic = d["ts"][sub]                              # one participant ROI time series
values = list(dic.values())                     # get timeseries
m = np.zeros((2300,52))
m = np.array(values).reshape((len(dic), 2300))  # properly reshape
lst.append(m.T)

#print(len(lst))
#print(m.shape)
# np.where(d["b_age"]==np.max(d["b_age"][:]))   # find young/old - est subject
print(d["b_age"][sub])

corr_met = "correlation"
pollo = ConnectivityMeasure(kind = corr_met)
f     = pollo.fit_transform(lst)

vmax  = np.max(f)
vmin = np.min(f)

plt.imshow(np.squeeze(f), vmin = vmin, vmax = vmax)
plt.colorbar()
plt.show()



# %% ------------------------------------------------------------------
# ##                Dealing with negative correlations
# ## ------------------------------------------------------------------
# function that addresses the negative correlations


# Preallocation of relevant variables
options     = ["abs", "zero", "keep"]
corr_met    = "correlation"


fig, axs = plt.subplots(1,3)
fig.suptitle("Negative correlation step")

for i, opt in enumerate(options):
    # definining one instance
    lst = []
    lst.append(m.T)
    # calculating f-connectivity
    temp = ConnectivityMeasure(kind = corr_met)
    f    = temp.fit_transform(lst)
    f    = neg_corr(opt, f)
    # plotting the results
    axs[i].matshow(np.squeeze(f), cmap='coolwarm')
    axs[i].set_title(opt)


# %% ------------------------------------------------------------------
# ##                Loop through participants
# ## ------------------------------------------------------------------
a = []
f = {}
subs = 109
# corr_mets = {"covariance", "correlation", "partial correlation", "tangent", "precision"}
corr_mets = {"covariance", "correlation", "partial correlation"}

#len(list(d.values())[0]) #-> number of participants
for sub,n in enumerate(list(d.values())[0]):                       # getting the subject dimension
    m = np.zeros((2300,52))
    m = np.array(list(d["ts"][sub].values()))                      # getting the ts data of a participant in the format requested by nilearn
    a.append(m.T)                                                  # appending subjects

for corr_met in corr_mets:                                         # multiverse of connectivity
    temp = ConnectivityMeasure(kind = corr_met)
    f[corr_met] = temp.fit_transform(a)

print(len(a))
print(m.shape)
fig, axs = plt.subplots(1,3)
fig.suptitle('3 functional connectivities')

for i, corr_met in enumerate(f.keys()):
    vmax  = np.max(f[corr_met][subs])
    vmin  = np.min(f[corr_met][subs])
    axs[i].matshow(f[corr_met][subs], vmin = vmin, vmax = vmax, cmap='coolwarm')
    axs[i].set_title(corr_met)
    plt.colorbar()
    
plt.show()


# %% Function

