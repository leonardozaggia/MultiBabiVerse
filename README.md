# MultiBabiVerse

MultiBabiVerse contains the code used to explore how prematurity affects early brain development using a multiverse approach. The analysis relies on 301 infant resting‑state fMRI datasets and evaluates more than a thousand preprocessing pipelines to test the stability of different network measures.

## Repository structure
- `Scripts/Local/` – Python modules to run the analysis on a local workstation. `main_.py` provides an example workflow based on the functions defined in `pipeline.py`.
- `Scripts/Cluster/` – Variants of the same pipeline adapted for high performance computing (HPC) environments.
- `multiverse_analysis_classification.py` and `multiverse_analysis_regression.py` – Jupyter notebooks and scripts used during development to replicate parts of the analysis.
- `requirements.txt` – List of Python packages required to execute the code.

## Installation
Clone the repository and install the dependencies with
```bash
pip install -r requirements.txt
```
Python 3.8 or newer is recommended.

## Data
The raw fMRI time‑series and demographic files are not distributed with this repository. Update the path variables at the top of the scripts in `Scripts/Local` or `Scripts/Cluster` to point to your local copies of these data.

## Running the analysis
For a local run execute
```bash
python Scripts/Local/main_.py
```
For large scale experiments on a cluster modify the paths inside the scripts under `Scripts/Cluster/` and submit them using your scheduler of choice.

## License
This project is released under the University of Oldenburg License. See the corresponding license file for details.

## Author
Leonardo Zaggia – [@leonardo_zaggia](https://twitter.com/leonardo_zaggia)
