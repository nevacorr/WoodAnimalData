import pandas as pd
from data_cleaning_and_exploration import clean_data
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

remove_brain_measures = 0
make_tgs_binary = 0

# Load ferret data with multiple run information
ferret_orig = pd.read_csv('data/Ferret CatWalk EpoTH IDs 60-74 Run Statistics with Brain Morphology.csv')

# Clean data
ferret = clean_data(ferret_orig, remove_brain_measures, make_tgs_binary)

ferret = ferret.groupby(['ID']).mean()

# Class coding: 0: Control, 1: Vehicle, 2: Epo, 3: TH
# give every subject group a name
controlcode = 0  # Control = 0
vehiclecode = 1  # Vehicle = 1
epocode = 2  # Epo = 2
thcode = 3  # TH = 3

# Code group as number instead of string
ferret['Group'] = ferret['Group'].replace(controlcode, 'Control')
ferret['Group'] = ferret['Group'].replace(vehiclecode, 'Vehicle')
ferret['Group'] = ferret['Group'].replace(epocode, 'Epo')
ferret['Group'] = ferret['Group'].replace(thcode, 'TH')

cw_measures = ['PhaseDispersions_LH->RH_Mean', 'LH_Swing_(s)_Mean', 'StepSequence_CA_(%)',
                           'PhaseDispersions_LF->LH_CStat_Mean', 'StepSequence_AB_(%)', 'RH_SingleStance_(s)_Mean',
                           'LH_PrintWidth_(cm)_Mean', 'Run_Maximum_Variation_(%)']

brain_columns = ['total volume (cm^3)', 'cerebrum+brainstem (cm^3)', 'cerebellum (cm^3)', '% cerebellum',
                         'Summed White Matter GFAP (um)',
                         'CC Thickness (um)', 'Overall Sulci Sum', 'Overall Gyri Sum', 'total gross score', 'Pathology Score']

# Loop over catwalk measures
for meas in cw_measures:
    # Create a new figure for each column in B with a 2x4 grid of subplots
    fig, axes = plt.subplots(2, 5, figsize=(16, 12), sharey=True)
    fig.suptitle(f'{meas}', fontsize=14)

    # Flatten the axes array to make indexing easier
    axes = axes.flatten()

    # Loop over brain measures
    for i, brain_feature in enumerate(brain_columns):
        ax = axes[i]

        # Calculate correlation coefficient
        r_value, p_value = pearsonr(ferret[brain_feature], ferret[meas])

        # Scatter plot with color based on 'color_column'
        scatter = sns.scatterplot(x=ferret[brain_feature], y=ferret[meas], hue=ferret['Group'], ax=ax)

        # Capture handles and labels from the first scatter plot for the legend
        if i != 4:
            ax.get_legend().remove()
        else:
            sns.move_legend(ax, "upper right", bbox_to_anchor=(1.45, 1))

        # Scatter plot with regression line
        sns.regplot(x=ferret[brain_feature], y=ferret[meas], ax=ax, ci=None, scatter=False, color='gray')
        ax.set_title(f'vs\n{brain_feature}\nr={r_value:.2f} p={p_value:.3f}', fontsize=10)
        ax.set_xlabel(brain_feature)
        ax.set_ylabel(meas)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(right=0.92)  # Adjust to fit the title and legend
    plt.show(block=False)

mystop=1