'''
	Codes for PCA/ICA methods described in Detecting cell assemblies in large neuronal populations, Lopes-dos-Santos et al (2013).
											https://doi.org/10.1016/j.jneumeth.2013.04.010
	This implementation was written in Feb 2019 by Vitor Lopez dos Santos. 
    Modified with permission by Mostafa El-Kalliny (mostafa.elkalliny@gmail.com and mostafa.el-kalliny@colorado.edu)
'''

from general_utils import get_transient_timestamps_mod, calculate_auROC
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy import stats
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pdb
import os
from matplotlib.colors import Normalize, LinearSegmentedColormap

#### Custom imports start here ####
import util
from itertools import zip_longest
import warnings
from tqdm import tqdm


from CellReg import CellRegObj, trim_map, rearrange_neurons, get_cellreg_path
from scipy.ndimage import gaussian_filter1d


__author__ = "Vítor Lopes dos Santos_MElkallinyEdit"
__version__ = "2025.1"


def toyExample(assemblies, nneurons=10, nbins=1000, rate=1.):
    np.random.seed()

    actmat = np.random.poisson(rate, nneurons * nbins).reshape(nneurons, nbins)
    assemblies.actbins = [None] * len(assemblies.membership)
    for (ai, members) in enumerate(assemblies.membership):
        members = np.array(members)
        nact = int(nbins * assemblies.actrate[ai])
        actstrength_ = rate * assemblies.actstrength[ai]

        actbins = np.argsort(np.random.rand(nbins))[0:nact]

        actmat[members.reshape(-1, 1), actbins] = \
            np.ones((len(members), nact)) + actstrength_

        assemblies.actbins[ai] = np.sort(actbins)

    return actmat


class toyassemblies:

    def __init__(self, membership, actrate, actstrength):
        self.membership = membership
        self.actrate = actrate
        self.actstrength = actstrength


def marcenkopastur(significance):
    nbins = significance.nbins
    nneurons = significance.nneurons
    tracywidom = significance.tracywidom

    # calculates statistical threshold from Marcenko-Pastur distribution
    q = float(nbins) / float(nneurons)  # note that silent neurons are counted too
    lambdaMax = pow((1 + np.sqrt(1 / q)), 2)
    lambdaMax += tracywidom * pow(nneurons, -2. / 3)  # Tracy-Widom correction

    return lambdaMax


def getlambdacontrol(zactmat_):
    significance_ = PCA()
    significance_.fit(zactmat_.T)
    lambdamax_ = np.max(significance_.explained_variance_)

    return lambdamax_


def binshuffling(zactmat, significance):
    np.random.seed()

    lambdamax_ = np.zeros(significance.nshu)
    for shui in range(significance.nshu):
        zactmat_ = np.copy(zactmat)
        for (neuroni, activity) in enumerate(zactmat_):
            randomorder = np.argsort(np.random.rand(significance.nbins))
            zactmat_[neuroni, :] = activity[randomorder]
        lambdamax_[shui] = getlambdacontrol(zactmat_)

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def circshuffling(zactmat, significance):
    np.random.seed()

    lambdamax_ = np.zeros(significance.nshu)
    for shui in tqdm(range(significance.nshu)):
        zactmat_ = np.copy(zactmat)
        for (neuroni, activity) in enumerate(zactmat_):
            cut = int(np.random.randint(significance.nbins * 2))
            zactmat_[neuroni, :] = np.roll(activity, cut)
        lambdamax_[shui] = getlambdacontrol(zactmat_)

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def runSignificance(zactmat, significance):
    if significance.nullhyp == 'mp':
        lambdaMax = marcenkopastur(significance)
    elif significance.nullhyp == 'bin':
        lambdaMax = binshuffling(zactmat, significance)
    elif significance.nullhyp == 'circ':
        lambdaMax = circshuffling(zactmat, significance)
    else:
        print('ERROR !')
        print('    nyll hypothesis method ' + str(nullhyp) + ' not understood')
        significance.nassemblies = np.nan

    nassemblies = np.sum(significance.explained_variance_ > lambdaMax)
    significance.nassemblies = nassemblies

    return significance


def extractPatterns(actmat, significance, method, random_seed=42):
    nassemblies = significance.nassemblies

    if method == 'pca':
        idxs = np.argsort(-significance.explained_variance_)[0:nassemblies]
        patterns = significance.components_[idxs, :]
    elif method == 'ica':
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=nassemblies, random_state=random_seed)  # Set random seed
        ica.fit(actmat.T)
        patterns = ica.components_
    else:
        print('ERROR !')
        print('    assembly extraction method ' + str(method) + ' not understood')
        patterns = np.nan

    if patterns is not np.nan:
        patterns = patterns.reshape(nassemblies, -1)

        # sets norm of assembly vectors to 1
        norms = np.linalg.norm(patterns, axis=1)
        patterns /= np.matlib.repmat(norms, np.size(patterns, 1), 1).T

    return patterns



def runPatterns(zactmat, method='ica', nullhyp='circ', nshu=1000,
                percentile=99, tracywidom=False):
    '''
    INPUTS

        zactmat:     activity matrix - numpy array (neurons, time bins)
                        should already be z-scored

        nullhyp:    defines how to generate statistical threshold for assembly detection.
                        'bin' - bin shuffling, will shuffle time bins of each neuron independently
                        'circ' - circular shuffling, will shift time bins of each neuron independently
                                                            obs: maintains (virtually) autocorrelations
                        'mp' - Marcenko-Pastur distribution - analytical threshold

        nshu:       defines how many shuffling controls will be done (n/a if nullhyp is 'mp')

        percentile: defines which percentile to be used use when shuffling methods are employed.
                                                                    (n/a if nullhyp is 'mp')

        tracywidow: determines if Tracy-Widom is used. See Peyrache et al 2010.
                                                (n/a if nullhyp is NOT 'mp')

    OUTPUTS

        patterns:     co-activation patterns (assemblies) - numpy array (assemblies, neurons)
        significance: object containing general information about significance tests
        zactmat:      returns zactmat

    '''

    nneurons = np.size(zactmat, 0)
    nbins = np.size(zactmat, 1)

    silentneurons = np.var(zactmat, axis=1) == 0
    if any(silentneurons):
        warnings.warn(f'Silent neurons detected: '
                      f'{np.where(silentneurons)[0].tolist()}')
    actmat_didspike = zactmat[~silentneurons, :]

    # Check if actmat_didspike is empty
    if actmat_didspike.size == 0:
        #pdb.set_trace()
        print("actmat_didspike is empty after removing silent neurons. Skipping assembly detection.")
        return None, None, zactmat

    # running significance (estimating number of assemblies)
    significance = PCA()
    significance.fit(actmat_didspike.T)
    significance.nneurons = nneurons
    significance.nbins = nbins
    significance.nshu = nshu
    significance.percentile = percentile
    significance.tracywidom = tracywidom
    significance.nullhyp = nullhyp
    significance = runSignificance(actmat_didspike, significance)
    if np.isnan(significance.nassemblies):
        return None, None, zactmat

    if significance.nassemblies < 1:
        print('WARNING !')
        print('    no assembly detected!')
        patterns = []
    else:
        # extracting co-activation patterns
        patterns_ = extractPatterns(actmat_didspike, significance, method)
        if patterns_ is np.nan:
            return None, None, zactmat

        # putting eventual silent neurons back (their assembly weights are defined as zero)
        patterns = np.zeros((np.size(patterns_, 0), nneurons))
        patterns[:, ~silentneurons] = patterns_

    return patterns, significance, zactmat

def computeAssemblyActivity(patterns, zactmat, zerodiag=True):
    nassemblies = len(patterns)
    nbins = np.size(zactmat, 1)

    assemblyAct = np.zeros((nassemblies, nbins))
    for (assemblyi, pattern) in enumerate(patterns):
        projMat = np.outer(pattern, pattern)
        projMat -= zerodiag * np.diag(np.diag(projMat))
        for bini in range(nbins):
            assemblyAct[assemblyi, bini] = \
                np.dot(np.dot(zactmat[:, bini], projMat), zactmat[:, bini])

    return assemblyAct


def find_assemblies(
    neural_data, method='ica', nullhyp='mp',
    n_shuffles=1000, percentile=99, tracywidow=False,
    compute_activity=True, use_bool=False, plot=True, plot_each_pattern=False,
    save_single_pattern=False, which_pattern_toSave=0,
    save_dir='none', filename='none',
    important_neurons_mode='raw', important_neurons_n=10
):
    """
    Gets patterns and assembly activations in one go.

    :parameters
    ---
    neural_data: (neuron, time) array
        Neural activity (e.g., S).

    method: str
        'ica' or 'pca'. 'ica' is recommended.

    nullhyp: str
        defines how to generate statistical threshold for assembly detection.
            'bin' - bin shuffling, will shuffle time bins of each neuron independently
            'circ' - circular shuffling, will shift time bins of each neuron independently
                     obs: maintains (virtually) autocorrelations
             'mp' - Marcenko-Pastur distribution - analytical threshold

    nshu: float
        defines how many shuffling controls will be done (n/a if nullhyp is 'mp').

    percentile: float
        defines which percentile to be used when shuffling methods are employed.
        (n/a if nullhyp is 'mp').

    tracywidow: bool
        determines if Tracy-Widom is used. See Peyrache et al 2010.
        (n/a if nullhyp is NOT 'mp').

    important_neurons_mode: str
        Mode for identifying important neurons ('raw', 'percentile', 'stdev').

    important_neurons_n: float
        Number or percentile of important neurons to extract.
    """
    # Preprocessing for activity matrix
    spiking, _, bool_arr = get_transient_timestamps_mod(
        neural_data, thresh_type="zscore", std_thresh=3, localMaxNumPoints=15
    )
    if use_bool:
        # Convert boolean array to integer array for imputation
        int_arr = bool_arr.astype(int)
        int_arr = stats.zscore(int_arr, axis=1)
        # Replace NaNs with a specific value, e.g., 0
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        imputed_int_arr = imp.fit_transform(int_arr.T).T
        actmat = imputed_int_arr
    else:
        actmat = stats.zscore(neural_data, axis=1)
        # Replace NaNs
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        actmat = imp.fit_transform(actmat.T).T

    # Run pattern extraction
    patterns, significance, z_data = runPatterns(
        actmat, method=method, nullhyp=nullhyp, nshu=n_shuffles,
        percentile=percentile, tracywidom=tracywidow
    )
    if patterns is None:
        return None, None, None

    # Identify important neurons
    important_neurons = get_important_neurons(patterns, mode=important_neurons_mode, n=important_neurons_n)

    # Generate distinct colors for each pattern
    if len(patterns) > 0:
        colorsAll = [util.distinct_colors(neural_data.shape[0]) for _ in range(len(patterns))]
    else:
        spiking = []
        colorsAll = []

    # Compute assembly activations
    activations = computeAssemblyActivity(patterns, actmat) if compute_activity else None

    # Plot assemblies if requested
    fig, axs = None, None
    if plot:
        # Dynamically select the plotting function
        if 'spike' in filename.lower():
            plotting_function = plot_assemblies_individual_ephys
        else:
            plotting_function = plot_assemblies_individual

        # Plotting logic
        if plot_each_pattern:
            plotting_function(
                activations, actmat, patterns, save=save_single_pattern,
                whichPattern=which_pattern_toSave, save_dir=save_dir, filename=filename
            )
        else:
            colors = util.distinct_colors(neural_data.shape[0])
            fig, axs = plot_assemblies(activations, actmat, colors=colors)

    # Build output dictionary
    assembly_dict = {
        'patterns': patterns,
        'significance': significance,
        'z_data': z_data,
        'orig_data': neural_data,
        'activations': activations,
        'sorted_spiking': spiking,
        'sorted_colors': colorsAll,
        'important_neurons': important_neurons
    }

    return assembly_dict, fig, axs




def membership_sort(patterns, neural_data, sort_duplicates=True):
    """
    Sorts neurons by their contributions to each pattern.

    :param patterns:
    :param neural_data:
    :return:
    """
    high_weights = get_important_neurons(patterns, mode='stdev')
    colors = util.distinct_colors(patterns.shape[0])

    do_not_sort, sorted_data, sorted_colors = [], [], []
    for color, pattern in zip(colors, high_weights):
        for neuron in pattern:
            if neuron not in do_not_sort:
                sorted_data.append(neural_data[neuron])
                sorted_colors.append(color)

                if not sort_duplicates:
                    do_not_sort.append(neuron)

    return sorted_data, sorted_colors


def preprocess_multiple_sessions(S_list, smooth_factor=0,
                                 neurons=None, use_bool=True,
                                 z_method='global'):
    # Store original data.
    data = {'orig_S_list': S_list.copy()}

    # Keep certain neurons here. If None, keep all.
    if neurons is not None:
        S_list = [S[neurons] for S in S_list]

    # Get event timestamps.
    spike_times, rates, bool_arr_list, new_S = [], [], [], []
    for S in S_list:
        # Handle missing data.
        S = np.asarray(S, dtype=float)
        imp = SimpleImputer(missing_values=np.nan, strategy='constant',
                            fill_value=0)
        S = imp.fit_transform(S.T).T

        # Get spiking timestamps.
        temp_s, temp_r, temp_bool = \
            get_transient_timestamps(S, thresh_type='eps',
                                     do_zscore=False)
        spike_times.append(temp_s)
        rates.append(temp_r)
        bool_arr_list.append(temp_bool)
        new_S.append(S)
    S_list = new_S

    # Smooth if desired.
    if smooth_factor > 0:
        S_list = [util.smooth_array(S, smooth_factor)
                  for S in S_list]
        bool_arr_list = [util.smooth_array(spikes, smooth_factor)
                         for spikes in bool_arr_list]

    # Make sure to z-score. Either globally or locally.
    # If global, take into account activity from all sessions that got
    # passed through this function. If local, just z-score within
    # session.
    if z_method == 'global':
        S_list = util.zscore_list(S_list)
        bool_arr_list = util.zscore_list(bool_arr_list)

    elif z_method == 'local':
        S_list = [stats.zscore(S, axis=1) for S in S_list]
        bool_arr_list = [stats.zscore(spikes, axis=1) for spikes in bool_arr_list]

    data['S'] = S_list
    data['spike_times'] = spike_times
    data['spike_rates'] = rates
    data['bool_arrs'] = bool_arr_list

    if use_bool:
        data['processed'] = bool_arr_list
    else:
        data['processed'] = S_list

    return data

def lapsed_activation(act_list, nullhyp='circ', n_shuffles=1000,
                      percentile=99):
    """
    Computes activity of ensembles based on data from another day.

    :parameters
    ---
    S_list: list of (neurons, time) arrays. The first entry will be
    considered the template AND all arrays must be sorted by row
    (neuron) in the same order.
        Neural activity from all sessions.

        method: str
        'ica' or 'pca'. 'ica' is recommended.

    nullhyp: str
        defines how to generate statistical threshold for assembly detection.
            'bin' - bin shuffling, will shuffle time bins of each neuron independently
            'circ' - circular shuffling, will shift time bins of each neuron independently
                     obs: maintains (virtually) autocorrelations
             'mp' - Marcenko-Pastur distribution - analytical threshold

    n_shuffles: float
        defines how many shuffling controls will be done (n/a if nullhyp is 'mp')

    percentile: float
        defines which percentile to be used use when shuffling methods are employed.
        (n/a if nullhyp is 'mp')
    """
    # Get patterns.
    patterns, significance, _= runPatterns(act_list[0],
                                           nullhyp=nullhyp,
                                           nshu=n_shuffles,
                                           percentile=percentile)

    if significance.nassemblies < 1:
        raise ValueError('No assemblies detected.')

    # Find assembly activations for the template session then the lapsed ones.
    activations = []
    for actmat in act_list:
        # Get activations.
        activations.append(computeAssemblyActivity(patterns, actmat))

    assemblies = {'activations': activations,
                  'patterns': patterns,
                  'significance': significance}

    return assemblies




def plot_assemblies(assembly_act, spiking, do_zscore=True, colors=None):
    """
    Plots assembly activations with S overlaid.

    :parameters
    ---
    assembly_act: list of (patterns, time) arrays
        Assembly activations.

    spiking: (sessions,) list of (neurons,) lists
        The inner lists should contain timestamps of spiking activity (e.g., from S).

    do_zscore: bool
        Flag to z-score assembly_act.

    colors: (sessions,) list of (neurons,) lists
        The inner lists should contain colors for each neuron.

    """

    # Handles cases where you only want to plot one session's assembly.
    if not isinstance(assembly_act, list):
        assembly_act = [assembly_act]

    # If colors are not specified, use defaults.
    if colors is None:
        colors = util.distinct_colors(assembly_act[0])

    # spiking should already be a list. Let's also check that it's a list
    # that's the same size as assembly_act. If not, it's probably a list
    # of a single session so package it into a list.
    if len(spiking) != len(assembly_act):
        spiking = [spiking]
        colors = [colors]

    # Get color for each assembly.
    uniq_colors = util.ordered_unique(colors[0])

    # Build the figure.
    n_sessions = len(assembly_act)
    fig, axes = plt.subplots(n_sessions, 1)
    if n_sessions == 1:
        axes = [axes]       # For iteration purposes.

    # For each session, plot each assembly.
    for n, (ax, act, spikes, c) in \
            enumerate(zip_longest(axes, assembly_act, spiking, colors,
                                  fillvalue='k')):
        if do_zscore:
            act = stats.zscore(act, axis=1)

        # Plot assembly activation.
        for activation, assembly_color in zip(act, uniq_colors):
            ax.plot(activation, color=assembly_color, alpha=0.7)
        ax2 = ax.twinx()
        ax2.invert_yaxis()

        # Plot S.
        ax2.eventplot(spikes, colors=c)
        ax.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        ax.set_ylabel('Ensemble activation [a.u.]')
        ax.set_xlabel('Time [frame]')
        ax2.set_ylabel('Neurons grouped by ensembles', rotation=-90)
        ax2.set_yticks([0, len(spikes)])

    return fig, axes

from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

def plot_assemblies_individual(assembly_act, spiking, patterns, do_zscore=True, colors=None, save=False, 
                               whichPattern=None, save_dir=None, filename=None, plotType='calcium'):
    """
    Plots each assembly activation with S on individual plots.

    :parameters
    ---
    assembly_act: list of (patterns, time) arrays
        Assembly activations.

    spiking: (sessions,) list of (neurons,) lists or 2D numpy array (neurons, time)
        The inner lists should contain timestamps of spiking activity (e.g., from S) or the activity matrix.

    patterns: array
        The patterns detected from the activity matrix.

    do_zscore: bool
        Flag to z-score assembly_act.

    colors: (sessions,) list of (neurons,) lists
        The inner lists should contain colors for each neuron.

    save: bool
        If True, save the specified pattern plot.

    whichPattern: int
        Index of the pattern to save.

    save_dir: str
        Directory to save the plot.

    filename: str
        Base filename to use for saving the plot.

    plotType: str
        Type of plot, 'spiking' or 'calcium'.
    """

    # Handles cases where you only want to plot one session's assembly.
    if not isinstance(assembly_act, list):
        assembly_act = [assembly_act]

    # If colors are not specified, use defaults.
    if colors is None:
        colors = [util.distinct_colors(assembly_act[0].shape[0]) for _ in range(len(patterns))]

    # spiking should already be a list. Let's also check that it's a list
    # that's the same size as assembly_act. If not, it's probably a list
    # of a single session so package it into a list.
    if len(spiking) != len(assembly_act):
        spiking = [spiking]
        colors = [colors]

    # Custom colormap from white to dark blue
    cmap = LinearSegmentedColormap.from_list('custom_blue', ['white', 'blue'])

    # For each session, plot each assembly on a separate figure.
    n_sessions = len(assembly_act)
    for session_idx in range(n_sessions):
        act = assembly_act[session_idx]
        spikes = spiking[session_idx]

        if do_zscore:
            act = stats.zscore(act, axis=1)

        for assembly_idx, activation in enumerate(act):
            if save and whichPattern is not None and assembly_idx != whichPattern:
                continue

            fig, ax = plt.subplots(2, 1, sharex=True)

            # Plot assembly activation.
            ax[0].plot(activation, color='black', alpha=0.7)
            ax[0].set_ylabel('Ensemble activation [a.u.]')

            # Identify neurons involved in this pattern
            pattern_weights = patterns[assembly_idx]
            threshold = np.mean(np.abs(pattern_weights)) + 2 * np.std(np.abs(pattern_weights))
            involved_neurons = np.where(np.abs(pattern_weights) > threshold)[0]

            if plotType == 'spiking':
                spikes_involved = [spikes[i] for i in involved_neurons]
                # Plot S for the involved neurons
                ax[1].eventplot(spikes_involved, colors='black')
            elif plotType == 'calcium':
                actmat = spikes
                min_val = np.min(actmat[involved_neurons, :])
                max_val = np.max(actmat[involved_neurons, :]) / 4
                norm = Normalize(vmin=min_val, vmax=max_val)
                im = ax[1].imshow(actmat[involved_neurons, :], aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
                #cbar = fig.colorbar(im, ax=ax[1], orientation='vertical')
                #cbar.set_label('Activity')

            ax[1].set_ylabel('Neurons')
            ax[1].set_xlabel('Time [frame]')
            ax[1].set_yticks([0, len(involved_neurons)])

            plt.tight_layout()

            if save and whichPattern is not None and assembly_idx == whichPattern:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with PdfPages(os.path.join(save_dir, f"{filename}.pdf")) as pdf:
                    fig.savefig(pdf, format='pdf')
                fig.savefig(os.path.join(save_dir, f"{filename}.png"))
                fig.savefig(os.path.join(save_dir, f"{filename}.svg"))
            plt.show()

def plot_assemblies_individual_ephys(
    assembly_act, spiking, patterns, 
    colors=None, save=False,
    whichPattern=None, save_dir=None, filename=None,
    xlim=(24000,26000)  # Add optional xlim parameter
):
    """
    Plots each assembly activation with z-scored spike rate data.
    Includes proper tick labels for interactive exploration.
    Optional xlim parameter can be added after finding desired window.
    """
    if not isinstance(assembly_act, list):
        assembly_act = [assembly_act]

    # Create custom green colormap with fast transition to green
    target_color = '#0DAC4B'
    import matplotlib.colors as mcolors
    target_rgb = mcolors.hex2color(target_color)
    
    positions = [0, 0.1, 0.3, 1.0]
    colors = ['#ffffff',  # pure white
             mcolors.rgb2hex((0.8, 0.95, 0.8)),  # very light green
             target_color,  # target green
             mcolors.rgb2hex((target_rgb[0]*0.5, target_rgb[1]*0.5, target_rgb[2]*0.5))]  # darker green
    cmap = LinearSegmentedColormap.from_list('custom_green', list(zip(positions, colors)))

    n_sessions = len(assembly_act)
    for session_idx in range(n_sessions):
        act = assembly_act[session_idx]
        spikes = spiking[session_idx] if isinstance(spiking, list) else spiking

        if spikes.ndim == 1:
            spikes = np.expand_dims(spikes, axis=0)

        for assembly_idx, activation in enumerate(act):
            if save and whichPattern is not None and assembly_idx != whichPattern:
                continue

            # Create figure
            fig = plt.figure(figsize=(12, 8))
            gs_main = plt.GridSpec(2, 1, height_ratios=[1, 2], figure=fig)
            
            # Top plot: Assembly activation
            ax_act = fig.add_subplot(gs_main[0])
            ax_act.plot(activation, color='black', alpha=0.7, linewidth=1)
            ax_act.set_ylabel('Assembly\nActivation')
            if xlim:
                ax_act.set_xlim(xlim)
            
            # Show x-axis ticks on top plot too for navigation
            ax_act.tick_params(labelbottom=True)
            
            # Bottom plot: Spike raster
            ax_spikes = fig.add_subplot(gs_main[1], sharex=ax_act)
            
            # Get neurons involved in this assembly
            pattern_weights = patterns[assembly_idx]
            threshold = np.mean(np.abs(pattern_weights)) + 1.5 * np.std(np.abs(pattern_weights))
            involved_neurons = np.where(np.abs(pattern_weights) > threshold)[0]

            if len(involved_neurons) == 0:
                print(f"Warning: No neurons passed threshold for assembly {assembly_idx}")
                involved_neurons = np.arange(len(pattern_weights))

            # Sort neurons by their pattern weights
            sorted_indices = involved_neurons[np.argsort(-np.abs(pattern_weights[involved_neurons]))]
            spike_data = spikes[sorted_indices]

            # Set limits for positive values only
            vmax = np.percentile(spike_data[spike_data > 0], 95)
            norm = Normalize(vmin=0, vmax=vmax)
            
            # Plot with bin numbers for easy reference
            im = ax_spikes.imshow(spike_data, 
                                aspect='auto', 
                                cmap=cmap, 
                                norm=norm,
                                interpolation='nearest')
            
            ax_spikes.set_ylabel('Neuron Index')
            ax_spikes.set_xlabel('Time (bins)')
            
            # Add colorbar to the right of both subplots
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Z-scored Spike Rate')
            
            # Adjust the main subplot area to make room for colorbar
            plt.subplots_adjust(right=0.9)
            
            # Print instructions for user
            if xlim is None:
                print("\nZoom to desired window, then check the x-axis limits.")
                print("You can add these limits when calling the function using xlim=(start, end)")

            if save and (whichPattern is None or assembly_idx == whichPattern):
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                base_path = os.path.join(save_dir, f"{filename}_assembly_{assembly_idx+1}")
                fig.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
                fig.savefig(f"{base_path}.pdf", bbox_inches='tight')
                fig.savefig(f"{base_path}.svg", bbox_inches='tight')
                
            plt.show()

def get_important_neurons(patterns, mode='raw', n=10, stdevthresh = 2):
    """
    Gets the most highly contributing neurons from each pattern.

    :parameters
    ---
    patterns: (patterns, neurons) array
        Weights for each neuron.

    mode: 'raw', 'percentile', or 'stdev'
        Determines whether to interpret n as a percentile, a raw number, or standard deviation threshold.

    n: float
        Percentile or number of neurons to extract from pattern weightings.

    :return
    ---
    inds: (patterns,) list of arrays
        Neuron indices.
    """
    inds = []

    if mode == 'percentile':
        n = int((100 - n) * patterns.shape[1] / 100)

    if mode == 'raw':
        for pattern in np.abs(patterns):
            if n > len(pattern):
                n = len(pattern)
            inds.append(np.argpartition(pattern, -n)[-n:])

    elif mode == 'stdev':
        for pattern in patterns:
            mean = np.mean(np.abs(pattern))
            std_dev = np.std(np.abs(pattern))
            threshold = mean + stdevthresh * std_dev
            important_indices = np.where(np.abs(pattern) > threshold)[0]
            inds.append(important_indices)

    return inds


if __name__ == '__main__':

    # Make toy datasets.
    toy = toyassemblies(membership=[[0, 1, 2, 3]],
                        actrate=[0.05],
                        actstrength=[10])
    act1 = stats.zscore(toyExample(toy, nbins=500), axis=1)

    toy = toyassemblies(membership=[[6, 7, 8, 9]],
                        actrate=[0.05],
                        actstrength=[10])
    act2 = stats.zscore(toyExample(toy, nbins=500), axis=1)
    acts = [act1, act2]

    toy = toyassemblies(membership=[[2, 3, 4, 5]],
                        actrate=[0.05],
                        actstrength=[10])
    act3 = stats.zscore(toyExample(toy, nbins=500), axis=1)
    acts = [act1, act2, act3]

    # Get patterns from first dataset.
    patterns = runPatterns(act1)[0]

    # Get activation strengths from all datasets.
    assemblyActs = []
    for act in acts:
        assemblyActs.append(computeAssemblyActivity(patterns, act))

    fig, axs = plt.subplots(len(acts), 2, sharey='col')
    for act, assemblyAct, ax in zip(acts,
                                    assemblyActs,
                                    axs):
        # Spikes and ensemble activation.
        ax[0].plot(assemblyAct.T, color='b', alpha=0.3)
        ax[0].set_ylabel('Activation strength')
        spike_ax = ax[0].twinx()
        spks = spike_ax.imshow(act, cmap='Reds')
        spike_ax.axis('tight')
        ax[0].set_zorder(spike_ax.get_zorder() + 1)
        ax[0].patch.set_visible(False)

        # Correlation matrix.
        r = ax[1].imshow(np.corrcoef(act))
        fig.colorbar(spks, ax=spike_ax)
        fig.colorbar(r, ax=ax[1])

    plt.tight_layout()
    axs[0,1].set_title('Correlations')
    plt.show()
