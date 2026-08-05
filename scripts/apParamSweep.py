###############################################
# Original author: Calvin Hinkle
# Last modified date: 8/4/2026 
# Description: 
# Load pulses from a npy array,
# preprocess the pulses using a DWT and min/max
# normalization, and run a parameter search for 
# AP clustering.
# Returns/saves: 
# parameter search grid as pandas df 
# the best AP model as a pkl

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
import joblib

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
        '--pref_range',
        nargs=2,
        metavar=('MIN', 'MAX'),
        type=float,
        required=False,
        help='Minimum and maximum preference values. '
             'Must provide exactly two values. '
             'Default: [-10, min(S)]'
    )  
    parser.add_argument(
        '--damp_range',
        nargs=2,
        metavar=('MIN', 'MAX'),
        type=float,
        default=[0.75, 0.99],
        required=False,
        help='Minimum and maximum damping values. '
             'Must provide exactly two values. '
             'Default: [0.75, 0.99]'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        required=False,
        help='Random seed'
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
        '--target_clusters',
        type=int,
        default=100,
        required=False,
        help='Target number of clusters for AP'
    )

    return parser

########################################################################################################################################

def run_ap(params, S, args, save_ap=False):
    pref = params["preference"]
    damping = params["damping"]

    try:
        ap = AffinityPropagation(
            preference=pref,
            damping=damping,
            affinity='precomputed', #pass S for precomputed, would need to change how S is computed for pref values
            #affinity='euclidean',
            random_state=args.seed,
            max_iter=args.max_iter,
            convergence_iter=args.converge_iter
        )

        labels = ap.fit_predict(S)

        converged = ap.n_iter_ < ap.max_iter

        if save_ap:
            return ap, labels

        if converged:
            n_clusters = len(np.unique(labels))
        else:
            n_clusters = np.nan
        return {
            "preference": pref,
            "damping": damping,
            "n_clusters": n_clusters,
            "converged": converged
        }

    except Exception as e:
        if save_ap:
            raise RuntimeError(f"Final AP fit failed: {e}")
            
        print(e)        
        return {
            "preference": pref,
            "damping": damping,
            "n_clusters": np.nan,
            "converged": False
        }

########################################################################################################################################

def get_best_point(results_df, target_clusters):
    valid_df = results_df.dropna(subset=["n_clusters"]).copy()

    if len(valid_df) == 0:
        raise RuntimeError("No AP runs converged.")

    best_idx = (
        (valid_df["n_clusters"] - target_clusters)
        .abs()
        .idxmin()
    )

    return valid_df.loc[best_idx]

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

    base_name = f"{today}_ap_grid_{pulse_file}"
    save_dir = args.save_directory.rstrip("/")

    save_loc = os.path.join(save_dir, f"{base_name}.pkl")
    save_loc_ap = os.path.join(
        save_dir,
        f"{base_name.replace('_ap_grid_', '_best_ap_')}.pkl"
    )

    # Avoid overwriting existing files
    count = 1
    while os.path.exists(save_loc) or os.path.exists(save_loc_ap):
        save_loc = os.path.join(
            save_dir, f"{base_name}_{count}.pkl"
        )
        save_loc_ap = os.path.join(
            save_dir,
            f"{base_name.replace('_ap_grid_', '_best_ap_')}_{count}.pkl"
        )
        count += 1

    return save_loc, save_loc_ap

########################################################################################################################################


    # NOTE: write out the logic here, then write functions for the logic. 
def main():
    # Parse arguments
    args = _arg_config().parse_args()

    os.makedirs(args.save_directory, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)

    # Preprocess
    pulses = np.load(args.pulse_file)
    norm_pulses = preprocess(pulses["waveform"])  
    
    # Compute pairwise distance and store in S
    S = get_similarity_matrix(norm_pulses)

    # Set preference and damping range
    if args.pref_range is None:
        #pref_min = -50
        S_no_diag = S[~np.eye(len(S), dtype=bool)]
        
        pref_min = np.percentile(S_no_diag, 5)
        pref_max = np.percentile(S_no_diag, 50)
    else:
        pref_min, pref_max = args.pref_range
    prefs = np.linspace(pref_min, pref_max, 12)
    
            #TO DO: change grid range to dynamic, not just 12x12
    damp_min, damp_max = args.damp_range
    dampings = np.linspace(damp_min, damp_max, 12)

    param_grid = list(
        ParameterGrid({"preference": prefs,
                       "damping": dampings}
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
    save_loc, save_loc_ap = make_save_name(args)
        
    results_df = pd.DataFrame(results)

    results_df.to_pickle(save_loc)

    print(f"AP grid search saved at {save_loc}")

    # Save best AP model
    
    best_point = get_best_point(
        results_df,
        args.target_clusters
    )
    
    best_params = {
        "preference": best_point["preference"],
        "damping": best_point["damping"]
    }
    
    ap_best, labels_best = run_ap(
        best_params,
        S,
        args,
        save_ap=True
    )
    
    joblib.dump(
        {
            "model": ap_best,
            "labels": labels_best,
            "best_params": best_params,
            "target_clusters": args.target_clusters,
            "wavelet": "haar",
            "dwt_level": 3,
            "similarity_metric": "l1",
            "normalization": "max_abs",
            "median_similarity": np.median(S[~np.eye(len(S), dtype=bool)])
        },
        save_loc_ap
    )
        
    print(f"Best AP model saved at {save_loc_ap}")
    

########################################################################################################################################

if __name__ == '__main__':
    main()

