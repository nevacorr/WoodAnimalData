import pandas as pd
from data_cleaning_and_exploration import clean_data

remove_brain_measures = 1
make_tgs_binary = 0

# Load ferret data with multiple run information
ferret_orig = pd.read_csv('data/Ferret CatWalk EpoTH IDs 60-74 Run Statistics with Brain Morphology.csv')

# Clean data
ferret = clean_data(ferret_orig, remove_brain_measures, make_tgs_binary)

# Class coding: 0: Control, 1: Vehicle, 2: Epo, 3: TH

most_important_features = ['PhaseDispersions_LH->RH_Mean', 'LH_Swing_(s)_Mean', 'StepSequence_CA_(%)',
                           'PhaseDispersions_LF->LH_CStat_Mean', 'StepSequence_AB_(%)', 'RH_SingleStance_(s)_Mean',
                           'LH_PrintWidth_cm_Mean', 'Run_Maximum_Variation_(%)']

brain_columns = ['total volume (cm^3)', 'cerebrum+brainstem (cm^3)', 'cerebellum (cm^3)', '% cerebellum',
                         'Summed White Matter GFAP (um)',
                         'CC Thickness (um)', 'Overall Sulci Sum', 'Overall Gyri Sum']

mystop=1