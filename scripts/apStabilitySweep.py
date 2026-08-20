###############################################
# Original author: Calvin Hinkle
# Last modified date: 8/20/2026 
# Description: 
# Load pulses from a npy array,
# preprocess the pulses using a DWT and min/max
# normalization, and run a stability test
# for parameters used for AP clustering.
# Returns/saves: 
# parameter search grid as pandas df 
# including information on cluster structure, convergence
# and stability
#
# AP solutions can be evaluated using (not here):
# clustering stability
# cluter size distribution (max, min, mean, median)
# physical association with laser, multiplicity, energy, etc
# manual interpretibilty 
# single, pileup, etc
# downstream SVM performance 

# we should select based on (not here):
# convergance
# stability (high mean ARI + low ARI variance)
# minimal number of singleton clusters
# physical association with laser, multiplicity, energy, etc --> future script
# SVM performance
#
# This script is intended to be run after apParamSweep.py 
# and the results are analyzed to find a small (<<30) parameter space

###############################################

import argparse
import numpy as np
import pywt
import pandas as pd
from datetime import datetime
import os
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from sklearn.cluster import AffinityPropagation
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import adjusted_rand_score as ari

########################################################################################################################################

PRE_PULSE_LEN = int(2**7)
PULSE_LEN = int(2**11 - PRE_PULSE_LEN)

########################################################################################################################################

def _arg_config():
    parser = argparse.ArgumentParser(
        description='Load pulses from a npy array, pre-process, and run a parameter search for AP clustering'
    )
    parser.add_argument(
        '--pulse_file',
        help='File where the selected pulses are stored. Input data must follow from pulses saved using getNPulses.py',
        required=True
    )
    parser.add_argument(
        '--save_directory',
        help='Directory where AP parameter search results will be stored',
        default='/data_fast/chinkle/26_ml/apResults/',
        required=False
    )
    parser.add_argument(
        '--threads',
        help='Max threads to launch. Max 48 on CRONOS',
        default=20,
        required=False,
        type=int
    )
    parser.add_argument(
        '--prefs',
        nargs='+',
        type=float,
        required=True,
        help='Preference values to test. These will be tested with each damping value.'
             'Must provide at least 1 value.'
             'Provide values seperated by spaces. Ex: -27.1 -26.34 -14.3'
    )  
    parser.add_argument(
        '--dampings',
        nargs='+',
        type=float,
        required=True,
        help='Damping values to test. These will be tested with each preference value.'
             'Must provide at least 1 value.'
             'Provide values seperated by spaces. Ex: 0.80 0.82 0.84'
    ) 
    parser.add_argument(
        '--max_iter',
        type=int,
        default=500,
        required=False,
        help='Max iterations to run AP'
    )
    parser.add_argument(
        '--converge_iter',
        type=int,
        default=15,
        required=False,
        help='Number of iterations with no change in the number of estimated clusters that stops the convergence'
    )
    parser.add_argument(
        '--num_seeds',
        type=int,
        default=5,
        required=False,
        help='Number of seeds to test model stability. ' 
        'Default: 5'
    )

    return parser

########################################################################################################################################

def preprocess(_pulses, _level=3):
    # Apply DWT
    wavelet='haar'
    coeffs = pywt.wavedec(_pulses, wavelet, level=_level, axis=1)
    cA3, cD3, cD2, cD1 = coeffs # cA3 is smoothed with 1/8 length of normal time

    prepulse_len_dwt = PRE_PULSE_LEN // 8
    #pulse_len_dwt = PULSE_LEN // 8
    
    # Apply offset
    offsets = np.median(cA3[:,:(prepulse_len_dwt//2)], axis=1)
    baseline_pulses = np.subtract(cA3, offsets[:, None])
    
    # Find max or min
    max_min = np.maximum(
        np.max(baseline_pulses, axis=1),
        np.abs(np.min(baseline_pulses, axis=1))
    )[:, None]

    # Normalize to [-1,1]
    max_min[max_min == 0] = 1
    norm_pulses = np.divide(baseline_pulses, max_min)

    return norm_pulses

########################################################################################################################################

def get_similarity_matrix(_pulses):
    S = -pairwise_distances(_pulses, metric='l1')
    N = len(_pulses) #number of waveforms
    p = np.median(S[~np.eye(N, dtype=bool)])
    np.fill_diagonal(S,p)

    return S
    
########################################################################################################################################

def make_save_name(args):
    # Extract info from the pulse file like this:
    # /data_fast/chinkle/26_ml/selectedPulses/day16_chan03_5000_20260729_{potentially other info}.npy

    pulse_file = os.path.splitext(os.path.basename(args.pulse_file))[0]

    today = datetime.now().strftime("%Y%m%d")

    base_name = f"{today}_ap_stability_{pulse_file}"
    save_dir = args.save_directory.rstrip("/")

    save_loc = os.path.join(save_dir, f"{base_name}.pkl")

    # Avoid overwriting existing files
    count = 1
    while os.path.exists(save_loc):
        save_loc = os.path.join(
            save_dir, f"{base_name}_{count}.pkl"
        )

        count += 1

    return save_loc

########################################################################################################################################

def run_ap(params, S, args):
    pref = params["preference"]
    damping = params["damping"]
    seed = params["seed"]

    try:
        ap = AffinityPropagation(
            preference=pref,
            damping=damping,
            affinity='precomputed', #pass S for precomputed, would need to change how S is computed for pref values
            #affinity='euclidean',
            random_state=seed,
            max_iter=args.max_iter,
            convergence_iter=args.converge_iter
        )

        labels = ap.fit_predict(S)

        # Convergence is required before cluster-structure metrics are valid
        converged = ap.n_iter_ < ap.max_iter

        cluster_sizes = np.bincount(labels)

        if converged:
            n_clusters = len(np.unique(labels))
        else:
            n_clusters = np.nan
        return {
            "preference": pref,
            "damping": damping,
            "seed": seed,
            "n_clusters": n_clusters,
            "converged": converged,
            "n_iter": ap.n_iter_,
            "n_singleton_clusters": np.sum(cluster_sizes == 1),
            "fraction_singleton_clusters": np.sum(cluster_sizes == 1) / n_clusters if n_clusters > 0 else np.nan,
            "fraction_singleton_events": np.sum(cluster_sizes == 1) / len(labels) if len(labels) > 0 else np.nan,
            "min_cluster_size": np.min(cluster_sizes) if n_clusters > 0 else np.nan,
            "max_cluster_size": np.max(cluster_sizes) if n_clusters > 0 else np.nan,
            "mean_cluster_size": np.mean(cluster_sizes) if n_clusters > 0 else np.nan,
            "median_cluster_size": np.median(cluster_sizes) if n_clusters > 0 else np.nan 
        }

    except Exception as e:
        if save_ap:
            raise RuntimeError(f"Final AP fit failed: {e}")
            
        print(e)        
        return {
            "preference": pref,
            "damping": damping,
            "seed": seed,
            "n_clusters": np.nan,
            "converged": False,
            "n_iter": np.nan,
            "n_singleton_clusters": np.nan,
            "fraction_singleton_clusters": np.nan,
            "fraction_singleton_events": np.nan,
            "min_cluster_size": np.nan,
            "max_cluster_size": np.nan,
            "mean_cluster_size": np.nan,
            "median_cluster_size": np.nan 
        }


########################################################################################################################################

def main():

    # Parse arguments
    args = _arg_config().parse_args()

    os.makedirs(args.save_directory, exist_ok=True)

    # Preprocess
    pulses = np.load(args.pulse_file)
    norm_pulses = preprocess(pulses["waveform"])  
    
    # Compute pairwise distance and store in S
    S = get_similarity_matrix(norm_pulses)

    # Load preference and damping values
    prefs = args.prefs
    dampings = args.dampings

    # Generate seed list
    max_seed = 1e6
    seeds = [np.random.randint(max_seed) for _ in range(args.num_seeds)]

    # Param grid of pref, damping, seed
    param_grid = list(
        ParameterGrid({"preference": prefs,
                       "damping": dampings,
                       "seed": seeds}
                     )
    )

    print(f'Number of tasks: {len(param_grid)}')

    # AP grid search
    results = Parallel(
        n_jobs=args.threads, 
        verbose=10
    )(
        delayed(run_ap)(
            params, 
            S,
            args
            ) for params in param_grid
    )

    # Save results from AP param search

    save_loc = make_save_name(args)
        
    results_df = pd.DataFrame(results)

    results_df.to_pickle(save_loc)

    print(f"AP grid search saved at {save_loc}")



########################################################################################################################################

if __name__ == '__main__':
    main()
