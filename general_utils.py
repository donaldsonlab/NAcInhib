# In general_utils.py
__all__ = ['get_transient_timestamps_mod', 'calculate_auROC', 'significant_modulation_proportion', 'generate_labels_from_cv', 'calculate_auROC_with_permutation']

import numpy as np
import scipy.signal
from scipy.signal import argrelextrema
from sklearn.metrics import roc_auc_score
from random import randint
import random
import pdb


def get_transient_timestamps_mod(
    neural_data, thresh_type="zscore_localMax", std_thresh=3, localMaxNumPoints=15
):
    """
    Converts an array of continuous time series (e.g., traces or S)
    into lists of timestamps where activity exceeds some threshold.

    :parameters
    ---
    neural_data: (neuron, time) array
        Neural time series, (e.g., C or S).

    thresh_type: str
        Type of thresholding ("zscore" or "zscore_localMax" or "scipy_peak").

    std_thresh: float
        Number of standard deviations above the mean to define threshold.

    :returns
    ---
    event_times: list of length neuron
        Each entry in the list contains the timestamps of a neuron's
        activity.

    event_mags: list of length neuron
        Event magnitudes.

    bool_arr: ndarray of bool
        Boolean array indicating whether a value exceeds the threshold.
    """
    # Compute thresholds (z-scores) for each neuron.
    neural_data = np.asarray(neural_data, dtype=np.float32)
    stds = np.std(neural_data, axis=1)
    means = np.mean(neural_data, axis=1)
    thresh = means + std_thresh * stds

    # Initialize event times, magnitudes, and boolean array.
    event_times = []
    event_mags = []
    bool_arr = np.zeros_like(neural_data, dtype=bool)

    for index, (neuron, t) in enumerate(zip(neural_data, thresh)):
        event_indices = []
        
        if thresh_type == "zscore":
            event_indices = np.where(neuron > t)[0]
        
        elif thresh_type == "zscore_localMax":
            for i in range(1, len(neuron) - 1):
                if (
                    neuron[i] > t
                    and all(0 <= j < len(neuron) and neuron[i] > neuron[j] for j in range(i - localMaxNumPoints, i))
                    and all(0 <= j < len(neuron) and neuron[i] > neuron[j] for j in range(i + 1, i + localMaxNumPoints + 1))
                ):
                    event_indices.append(i)
        elif thresh_type == "scipy_peak":
            local_max = scipy.signal.argrelextrema(neuron, np.greater, order=10)[0]
            event_indices = local_max          
        
        event_times.append(np.array(event_indices))
        event_mags.append(neuron[event_indices])
        bool_arr[index, event_indices] = True

    return event_times, event_mags, bool_arr


def calculate_auROC(C_final, behavior_events):
    """
    Calculates the area under the Receiver Operating Characteristic curve (auROC)
    for each neuron's ability to predict the occurrence of a specific behavior.

    Args:
    C_final (np.ndarray): The matrix of neural activity (neurons x time).
    behavior_events (np.ndarray): Boolean array indicating the presence (True) or
                                  absence (False) of the behavior at each time point.

    Returns:
    np.ndarray: Array of auROC values for each neuron.
    """
    auROCs = np.array([roc_auc_score(behavior_events, C_final[neuron, :]) for neuron in range(C_final.shape[0])])
    return auROCs



def calculate_auROC_with_permutation(C_final, labels, num_permutations=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    actual_auROC = roc_auc_score(labels, C_final)
    shuffled_auROCs = []

    for _ in range(num_permutations):
        shift = randint(0, C_final.shape[0] - 1)  # Assuming C_final is a 1D array for a single neuron
        shuffled_C_final = np.roll(C_final, shift)
        shuffled_auROC = roc_auc_score(labels, shuffled_C_final)
        shuffled_auROCs.append(shuffled_auROC)

    p_value = sum(1 for roc in shuffled_auROCs if roc >= actual_auROC) / num_permutations
    return actual_auROC, p_value


def calculate_threshold_auROC_with_permutation(C_final, labels, num_permutations=500, threshold=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Calculate the threshold value as mean + (threshold * std)
    threshold_value = np.mean(C_final) + (threshold * np.std(C_final))

    # Create a mask to only consider time points above the threshold
    high_activation_mask = C_final > threshold_value
    filtered_C_final = C_final[high_activation_mask]
    filtered_labels = labels[high_activation_mask]

    # Ensure there are enough data points to calculate AUROC
    if len(filtered_C_final) > 1 and len(np.unique(filtered_labels)) > 1:
        actual_auROC = roc_auc_score(filtered_labels, filtered_C_final)
        shuffled_auROCs = []

        for _ in range(num_permutations):
            # Shuffle the entire C_final data and apply the same threshold-based filter
            shuffled_full_C_final = np.roll(C_final, randint(0, len(C_final) - 1))
            shuffled_filtered_C_final = shuffled_full_C_final[high_activation_mask]

            # Only calculate AUROC if there are valid points
            if len(shuffled_filtered_C_final) > 1 and len(np.unique(filtered_labels)) > 1:
                shuffled_auROC = roc_auc_score(filtered_labels, shuffled_filtered_C_final)
                shuffled_auROCs.append(shuffled_auROC)

        # Calculate p-value based on the permutation distribution
        p_value = sum(1 for roc in shuffled_auROCs if roc >= actual_auROC) / num_permutations
    else:
        print("Insufficient data points for meaningful AUROC calculation. Returning neutral values.")
        actual_auROC = 0.5
        p_value = 1.0

    return actual_auROC, p_value




def significant_modulation_proportion(C_final, labels, num_permutations=1000, significance_level=0.05, seed=42, threshold=False):
   """Calculate proportion of neurons (or assemblies) significantly positively and negatively modulated.
   
   Args:
       C_final (np.ndarray): Neural activity data (neurons x timepoints).
       labels (np.ndarray): Binary labels for each timepoint.
       significance_level (float): Threshold for significance.

   Returns:
       tuple: (proportion_positive, proportion_negative, positive_indices, negative_indices, auroc_values, p_values)
   """
   significant_positive_count = 0
   significant_negative_count = 0
   positive_indices = []
   negative_indices = []
   auroc_values = []
   p_values_list = []

   for i, neuron_data in enumerate(C_final):  # Assuming C_final is 2D
       if threshold==False:
            actual_auROC, p_values = calculate_auROC_with_permutation(neuron_data, labels, 
                                                                num_permutations = num_permutations, 
                                                                seed=seed)
       else:
            actual_auROC, p_values = calculate_threshold_auROC_with_permutation(neuron_data, labels, 
                                                                num_permutations = num_permutations, 
                                                                seed=seed, threshold=2)           
       
       
       auroc_values.append(actual_auROC)
       p_values_list.append(p_values)

       if actual_auROC > 0.5 and p_values <= significance_level:
           significant_positive_count += 1
           positive_indices.append(i)
       elif actual_auROC < 0.5 and p_values >= (1 - significance_level):
           significant_negative_count += 1
           negative_indices.append(i)

   total_neurons = C_final.shape[0]
   proportion_positive = significant_positive_count / total_neurons if total_neurons > 0 else 0
   proportion_negative = significant_negative_count / total_neurons if total_neurons > 0 else 0

   return proportion_positive, proportion_negative, positive_indices, negative_indices, auroc_values, p_values_list


def generate_labels_from_cv_filtered(cv, total_frames):
    labels = np.zeros(total_frames, dtype=int)
    for _, row in cv.iterrows():
        start = int(row['scopeFrameStart'] / 2)  # Adjust for downsampling
        end = int(row['scopeFrameEnd'] / 2)
        labels[start:end] = 1
    return labels


def generate_labels_from_cv_filtered_npx(cv, total_frames, bin_size=0.1, sampling_rate=30000):
    labels = np.zeros(total_frames, dtype=int)
    for i, row in cv.iterrows():
        # Convert timestamps to bin indices using the correct time scale
        start = int(row['indexStart'] / (sampling_rate * bin_size))
        end = int(row['indexEnd'] / (sampling_rate * bin_size))
        
        # Print debug information for each event
        #print(f"Event {i}: Start {row['indexStart']} -> Bin {start}, End {row['indexEnd']} -> Bin {end}, Total frames: {total_frames}")
        
        # Ensure indices are valid
        if start < total_frames and end <= total_frames:
            labels[start:end] = 1
        else:
            print(f"Warning: Event {i} indices out of bounds (Start: {start}, End: {end}, Total frames: {total_frames})")
    return labels
