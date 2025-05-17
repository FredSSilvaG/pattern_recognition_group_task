import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from tqdm import tqdm

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

def main():
    # Update path to the test folder
    base_dir = './SignatureVerification-test'
    enrollment_dir = os.path.join(base_dir, 'enrollment')
    verification_dir = os.path.join(base_dir, 'verification')
    
    # Load writers from the test folder
    writers_df = pd.read_csv(os.path.join(base_dir, 'writers.tsv'), sep='\t', header=None, 
                           names=['writer_id'])
    
    print(f"Found {len(writers_df)} writers (starting from ID {min(writers_df['writer_id'])}).")
    
    # If there's no ground truth file in test, we'll just calculate distances without verification
    results = []
    print("Processing verification signatures...")
    
    # Get all verification files
    verification_files = os.listdir(verification_dir) if os.path.exists(verification_dir) else []
    
    if not verification_files:
        print("No verification files found. Will try to read test.tsv instead.")
        # If there are no verification files, try to read from test.tsv to get what we need to predict
        try:
            test_df = pd.read_csv(os.path.join(base_dir, 'test.tsv'), sep='\t', header=None)
            # Extract signature IDs that need verification from test.tsv
            verification_files = []
            for _, row in test_df.iterrows():
                writer_id = row[0]
                for i in range(1, len(row), 2):  # Skip distance columns
                    if i < len(row):
                        verification_files.append(f"{row[i]}.tsv")
        except:
            print("Error reading test.tsv. No verification files will be processed.")

    # Process each writer
    for writer_id in tqdm(range(31,34)):
        writer_id = int(writer_id)
        
        # Load reference signatures
        reference_files = [f"{writer_id:03d}-g-{i:02d}.tsv" for i in range(1, 6)]
        reference_paths = [os.path.join(enrollment_dir, f) for f in reference_files]
        reference_sigs = [load_signature(path) for path in reference_paths if os.path.exists(path)]
        
        if not reference_sigs:
            print(f"Warning: No reference signatures found for writer {writer_id}.")
            continue
        
        # Process verification signatures for this writer
        writer_verification_files = [f for f in verification_files if f.startswith(f"{writer_id:03d}-")]
        
        for sig_file in writer_verification_files:
            sig_path = os.path.join(verification_dir, sig_file)
            if not os.path.exists(sig_path):
                continue
                
            signature_id = sig_file.split('.')[0]  # Remove .tsv extension
            test_sig = load_signature(sig_path)
            distance = verify_signature(test_sig, reference_sigs)
            
            results.append({
                'writer_id': writer_id,
                'signature_id': signature_id,
                'distance': distance,
                # No ground truth available, just set to None/False
                'is_genuine': None,
                'verified_as_genuine': False
            })
    
    results_df = pd.DataFrame(results)
    
    # Generate predictions 
    predictions = generate_test_predictions(results_df)
    
    print(f"\nProcessed {len(results)} signatures")
    print("Results saved to results/test.tsv")
    
    return results_df

if __name__ == "__main__":
    main()