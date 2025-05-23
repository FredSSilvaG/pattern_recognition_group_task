import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import param_utils
from utils import report_util

def load_gt(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, 
                       names= ['signature_id','label'])
    df['is_genuine'] = df['label'].str.lower() == 'genuine'
    return df

def dtw(s1, s2, distance_func=None):
    if distance_func is None:
        distance_func = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
    
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = distance_func(s1[i-1], s2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    
    return dtw_matrix[n, m]

def load_signature(file_path):
    return pd.read_csv(file_path, sep='\t', header=None, 
                     names=['t', 'x', 'y', 'pressure', 'penup', 'azimuth', 'inclination'])

def extract_features(signature_df):
    features = signature_df[['x', 'y', 'pressure']].values
    dx = np.gradient(signature_df['x'].values)
    dy = np.gradient(signature_df['y'].values)
    speed = np.sqrt(dx**2 + dy**2)
    direction = np.arctan2(dy, dx)
    all_features = np.column_stack([features, speed.reshape(-1, 1), direction.reshape(-1, 1)])
    normalized_features = all_features.copy()
    
    for i in range(all_features.shape[1]):
        min_val = np.min(all_features[:, i])
        max_val = np.max(all_features[:, i])
        if max_val > min_val:
            normalized_features[:, i] = (all_features[:, i] - min_val) / (max_val - min_val)
    
    return normalized_features

def signature_distance(sig1, sig2):
    features1 = extract_features(sig1)
    features2 = extract_features(sig2)
    return dtw(features1, features2)

def verify_signature(test_sig, reference_sigs, threshold=None):
    distances = [signature_distance(test_sig, ref_sig) for ref_sig in reference_sigs]
    avg_distance = np.mean(distances)
    
    if threshold is None:
        return avg_distance
    
    return avg_distance, avg_distance <= threshold

def generate_test_predictions(results_df):
    os.makedirs('results', exist_ok=True)
    output_lines = []
    
    for writer_id, group in results_df.groupby('writer_id'):
        group = group.drop_duplicates(subset=['signature_id'], keep='first')
        sorted_sigs = group.sort_values('distance')
        line = [f"{writer_id:03d}"]
        
        for _, row in sorted_sigs.iterrows():
            line.append(row['signature_id'])
            line.append(f"{row['distance']:.6f}")
        
        output_lines.append('\t'.join(line))
    
    with open('results/test.tsv', 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Predictions saved to results/test.tsv")
    return output_lines

def get_top_k_matches(results_df, k= 20):
    matches = []

    for writer_id, group in results_df.groupby('writer_id'):
        group = group.drop_duplicates(subset=['verification_signature_id'], keep='first')
        sorted_sigs = group.sort_values('distance').head(k)
        
        for _, row in sorted_sigs.iterrows():
            matches.append({
                'writer_id': f"{writer_id:03d}",
                'verification_signature_id': row['verification_signature_id'],
                'distance': row['distance']
            })

    return pd.DataFrame(matches) 


def prepare_matches_for_map(matches_df, gt_df) :
    matches_df = matches_df.merge(
    gt_df[['signature_id', 'is_genuine']],
    left_on='verification_signature_id',
    right_on='signature_id',
    how='left').drop(columns='signature_id')

    return matches_df

def average_precision(ranked_is_genuine):
    hits = 0
    sum_precisions = 0
    for i, relevant in enumerate(ranked_is_genuine, start=1):
        if relevant:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / hits if hits > 0 else 0.0


def compute_map(matches_df):
    writer_ap = {}
    ap_list = []

    for writer_id, group in matches_df.groupby('writer_id'):
        ranked_truth = group['is_genuine'].tolist()
        ap = average_precision(ranked_truth)
        ap_list.append(ap)
        writer_id = int(writer_id)
        writer_ap[writer_id] = ap

    map_k = np.mean(ap_list)
    return writer_ap, map_k


def main():

    args = param_utils.parse_args_SV()
    base_dir = args.base

    enrollment_dir = os.path.join(base_dir, 'enrollment')
    verification_dir = os.path.join(base_dir, 'verification')
    
    writers_df = pd.read_csv(os.path.join(base_dir, 'writers.tsv'), sep='\t', header=None, 
                           names=['writer_id'])
    
    print(f"Found {len(writers_df)} writers (starting from ID {min(writers_df['writer_id'])}).")
    
    results = []
    print("Processing verification signatures...")
    
    verification_files = os.listdir(verification_dir) if os.path.exists(verification_dir) else []
    
    # if not verification_files:
    #     try:
    #         test_df = pd.read_csv(os.path.join(base_dir, 'test.tsv'), sep='\t', header=None)
    #         verification_files = []
    #         for _, row in test_df.iterrows():
    #             writer_id = row[0]
    #             for i in range(1, len(row), 2):
    #                 if i < len(row):
    #                     verification_files.append(f"{row[i]}.tsv")
    #     except:
    #         print("Error reading test.tsv. No verification files will be processed.")

    for writer_id in tqdm(writers_df['writer_id']):
        writer_id = int(writer_id)
        
        reference_files = [f"{writer_id:03d}-g-{i:02d}.tsv" for i in range(1, 6)]
        reference_paths = [os.path.join(enrollment_dir, f) for f in reference_files]
        reference_sigs = [load_signature(path) for path in reference_paths if os.path.exists(path)]
        
        if not reference_sigs:
            print(f"Warning: No reference signatures found for writer {writer_id}.")
            continue
        
        writer_verification_files = [f for f in verification_files if f.startswith(f"{writer_id:03d}-")]
        
        for sig_file in writer_verification_files:
            sig_path = os.path.join(verification_dir, sig_file)
            if not os.path.exists(sig_path):
                continue
                
            verification_signature_id = sig_file.split('.')[0]
            test_sig = load_signature(sig_path)
            distance = verify_signature(test_sig, reference_sigs)
            
            results.append({
                'writer_id': writer_id,
                'verification_signature_id': verification_signature_id,
                'distance': distance,
                #'is_genuine': None,
                #'verified_as_genuine': False
            })

    results_df = pd.DataFrame(results)

    gt_df = load_gt(os.path.join(base_dir, 'gt.tsv'))

    top_k_df = get_top_k_matches(results_df, k= args.top_k)

    map_ready_df = prepare_matches_for_map(top_k_df, gt_df)

    writer_ap, map_k = compute_map(map_ready_df)

    print(f"MAP@20: {map_k:.4f}")

    report_util.save_report_as_markdown(writer_ap,map_k,args.out_path, args.top_k)

   # predictions = generate_test_predictions(results_df)
    
    print(f"\nProcessed {len(results)} signatures")
    
    return results_df

if __name__ == "__main__":
    main()