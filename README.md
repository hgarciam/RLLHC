# RLLHC
Repository containing the code related to RL applied to optics correction in the LHC and HL-LHC

Running: python3 RUN.py <dataset> <numer of jobs (run*)> <method [predictor/surrogate]> <model [Ridge/RandomForest]>

RUN.py = runs the script
RLLHC.py = contains the RLLHC class which contains all the information about data loading and models.

## TODO

- Feature selection based on feature importance (look for phase advances that are relevant)
- Score per output [arcs/triplet]