# NAcInhib
Codebase accompanying the manuscript: "Accumbal calcium-permeable AMPARs orchestrate social attachment"

# Author(s)
Mostafa M. El-Kalliny, William M. Sheeran, Liza E. Brusman, Olivia E. Neilly, Kelly E. Winther, Michael A. Kelberman, Zoe R. Donaldson*

This dataset contains a combination of code used to
support the findings in the manuscript 
"Accumbal calcium-permeable AMPARs orchestrate social attachment"

Please see the Methods section of the manuscript for a
detailed description of how these data were generated. 

Questions can be sent to corresponding author:
zoe.donaldson@colorado.edu

Or to the primary author who produced this dataset:
mostafa.el-kalliny@colorado.edu
mostafa.elkalliny@gmail.com

# Outline
Files (which will become publicly available via Dryad)
are designed to be directly loaded into the Jupyter notebooks here
and to be sufficient for running all analyses and generating a rough
draft of figures.

Notes: 
'Stranger' and 'Novel' are interchangeable terms throughout,
as are 'Assembly' and 'Ensemble'.

# Used by gria2_seq.ipynb:
gria_sequencing.csv
Contains Gria subunit sequencing results. Each row is an individual
cell. Obtained from Brusman et al, as cited. 

# Used by pharmacology.ipynb:
final_IEM_data.xlsx
Results of pharmacology-only experiment examining the effect of bilateral
accumbal ACSF versus IEM-1460 administration on bond formation

# Used by cell_encoding.ipynb:
cell_encoding_total.pkl
Variety of data consolidated for analysis of behavior during imaging, 
neuronal encoding via auROC, encoding across days by individual 
neurons, and spatial clustering of individual neurons. All variables 
are loaded up front, then used by individual functions accordingly.

# Used by population_encoding.ipynb:
population_encoding_total.pkl
Variety of data consolidated for neuronal encoding via 
ensembles and population analyses using PCA, decoding, 
cosine similarity. 

# Used by ephys_analysis.ipynb:
/ephys contains a #_day2_final.pkl with different # 
for each animal, containing all preprocessed 
electrophysiological data.
It is loaded by load_session_data and used as the basis 
for the remainder of the notebook.

ephys_snippet.pkl contains the snippet of data used by
the load_and_plot_snippet function. 

# Used by multiple notebooks
general_utils.py
Several core functions, including auROC 
calculations. 

assemblies.py 
Core code for PCA/ICA methods described in Detecting cell 
assemblies in large neuronal populations, 
Lopes-dos-Santos et al (2013).
https://doi.org/10.1016/j.jneumeth.2013.04.010
Implementation originally written by Vitor Lopez dos Santos, and
modified with permission.
