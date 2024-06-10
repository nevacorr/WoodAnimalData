import numpy as np

# Function to shuffle the series nshuffle times
def shuffle_series(series, nshuffles):
    unique_series = []
    while len(unique_series) < nshuffles:
        # Permute the series using numpy permutation
        permuted_series = np.random.permutation(series)

        permuted_series_list = permuted_series.tolist()
        # Check if the shuffled series is not already in the list
        #flag initialization
        flag = 0

        #check whether series already exists in the list
        for elem in unique_series:
            if permuted_series_list == elem:
                flag = 1

        #if series does not exist in the list, append to list of unique series
        if flag==0:
            unique_series.append(permuted_series_list)
            mystop=1
    return unique_series
