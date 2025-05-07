import numpy as np
import cv2
import os
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm

class KeywordSpotter:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.locations_dir = os.path.join(data_dir, 'locations')
        
        self.transcriptions = self._load_transcriptions()
        self.keywords = self._load_keywords()
        self.train_docs = self._load_doc_list('train.tsv')
        self.val_docs = self._load_doc_list('validation.tsv')

    def _load_transcriptions(self):
        df = pd.read_csv(os.path.join(self.data_dir, 'transcription.tsv'), 
                         sep='\t', header=None, names=['id', 'transcription'])
        return dict(zip(df['id'], df['transcription']))

    def _load_keywords(self):
        with open(os.path.join(self.data_dir, 'keywords.tsv'), 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _load_doc_list(self, filename):
        with open(os.path.join(self.data_dir, filename), 'r') as f:
            return [line.strip() for line in f.readlines()]

    def extract_word_image(self, word_id):
        doc_id, line_id, word_id_in_line = word_id.split('-')
        path_id = f"{line_id}-{word_id_in_line}"
        
        cutouts_png_dir = os.path.join(self.data_dir, "cutouts_png", doc_id)
        
        cutout_filename = f"{doc_id}-{doc_id}-{line_id}-{word_id_in_line}.png"
        cutout_path = os.path.join(cutouts_png_dir, cutout_filename)
        
        alt_path1 = os.path.join(cutouts_png_dir, f"{doc_id}-{line_id}-{word_id_in_line}.png")
        alt_path2 = os.path.join(cutouts_png_dir, f"{path_id}.png")
        
        if os.path.exists(cutout_path):
            img = cv2.imread(cutout_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        elif os.path.exists(alt_path1):
            img = cv2.imread(alt_path1, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        elif os.path.exists(alt_path2):
            img = cv2.imread(alt_path2, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
            
        print(f"No pre-cut image found for {word_id}, using fallback")
        return self._create_fallback_image()

    def _create_fallback_image(self):
        fallback = np.zeros((100, 100), dtype=np.uint8)
        
        cv2.line(fallback, (30, 30), (70, 70), 255, 2)
        cv2.line(fallback, (30, 70), (70, 30), 255, 2)
        
        return fallback
    
    def extract_features(self, image):
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        height, width = binary.shape
        feature_vectors = []
        
        for x in range(width):
            column = binary[:, x]
            
            foreground_pixels = np.where(column == 255)[0]
            
            if len(foreground_pixels) == 0:
                upper_contour = 0
                lower_contour = height - 1
                black_pixel_fraction = 0
                transitions = 0
                black_between = 0
            else:
                upper_contour = foreground_pixels[0]
                lower_contour = foreground_pixels[-1]
                black_pixel_fraction = len(foreground_pixels) / height
                
                # Count black/white transitions
                transitions = 0
                for i in range(1, height):
                    if column[i] != column[i-1]:
                        transitions += 1
                
                # Black pixels between contours
                between_contours = column[upper_contour:lower_contour+1]
                black_between = np.sum(between_contours == 255) / (lower_contour - upper_contour + 1)
            
            features = [
                upper_contour / height,  # Normalize
                lower_contour / height,  # Normalize
                transitions / height,    # Normalize
                black_pixel_fraction,    # Already normalized
                black_between            # Already normalized
            ]
            
            feature_vectors.append(features)
        
        feature_vectors = np.array(feature_vectors)
        for i in range(feature_vectors.shape[1]):
            std = np.std(feature_vectors[:, i])
            if std > 0:
                feature_vectors[:, i] = (feature_vectors[:, i] - np.mean(feature_vectors[:, i])) / std
        
        return feature_vectors

    def dtw_distance(self, s1, s2, window=15):
        n, m = len(s1), len(s2)
        cost = np.ones((n+1, m+1)) * np.inf
        cost[0, 0] = 0
        
        for i in range(1, n+1):
            start_j = max(1, i - window)
            end_j = min(m+1, i + window + 1)
            for j in range(start_j, end_j):
                dist = np.sqrt(np.sum((s1[i-1] - s2[j-1])**2))
                cost[i, j] = dist + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
        
        return cost[n, m]

    def find_matches(self, query_id, doc_ids, k=None):
        query_image = self.extract_word_image(query_id)
        query_features = self.extract_features(query_image)
        
        # Get candidate word IDs
        all_word_ids = []
        for doc_id in doc_ids:
            for word_id in self.transcriptions.keys():
                if word_id.startswith(doc_id):
                    all_word_ids.append(word_id)
        
        distances = []
        for word_id in tqdm(all_word_ids, desc=f"Matching {query_id}"):
            if word_id == query_id:
                continue
            word_image = self.extract_word_image(word_id)
            word_features = self.extract_features(word_image)
            distance = self.dtw_distance(query_features, word_features)
            distances.append((word_id, distance))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        return distances[:k] if k is not None else distances
    
    def evaluate(self, queries, doc_ids):
        all_precisions, all_recalls, all_aps = [], [], []
        
        for query_id in queries:
            query_word = self._get_clean_word(self.transcriptions[query_id])
            matches = self.find_matches(query_id, doc_ids)
            
            y_true = []
            for word_id, _ in matches:
                match_word = self._get_clean_word(self.transcriptions[word_id])
                y_true.append(1 if match_word == query_word else 0)
            
            if sum(y_true) > 0:
                y_scores = [-dist for _, dist in matches]
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                all_precisions.append(precision)
                all_recalls.append(recall)
                ap = average_precision_score(y_true, y_scores)
                all_aps.append(ap)
        
        mean_ap = np.mean(all_aps) if all_aps else 0

        
        return mean_ap
    
    def _get_clean_word(self, transcription):
        """Clean transcription text for comparison"""
        word = transcription.replace('s_mi', '-').replace('s_sq', ';').replace('s_qo', ':').replace('s_qt', "'")
        for i in range(10):
            word = word.replace(f's_{i}', str(i))
        word = word.replace('-', '')
        return word.lower()
    
    def create_test_output(self):
        val_docs = self._load_doc_list('validation.tsv')
        print(f"Creating test output for {len(self.keywords)} keywords across {len(val_docs)} documents...")
        
        # Process each keyword
        for keyword in tqdm(self.keywords, desc="Processing keywords"):
            query_id = None
            for word_id, transcription in self.transcriptions.items():
                if not any(word_id.startswith(doc_id) for doc_id in self.train_docs):
                    continue
                    
                if self._get_clean_word(transcription).lower() == keyword.lower():
                    query_id = word_id
                    break
            
            if not query_id:
                print(f"Warning: No training example found for keyword '{keyword}'. Skipping.")
                continue
                
            matches = self.find_matches(query_id, val_docs)
            
            output_line = keyword
            for word_id, distance in matches:
                output_line += f"\t{word_id}\t{distance:.6f}"
            
            print(output_line)
def main():
    kws = KeywordSpotter(data_dir='.')
    print(f"Loaded transcriptions for {len(kws.transcriptions)} words")
    print(f"Loaded {len(kws.keywords)} keywords")
    print(f"Train docs: {len(kws.train_docs)}")
    print(f"Validation docs: {len(kws.val_docs)}")

    print("\n=== Performance Evaluation ===")
    queries = []
    for doc_id in kws.train_docs[:2]:
        for word_id in kws.transcriptions:
            if word_id.startswith(doc_id):
                if kws._get_clean_word(kws.transcriptions[word_id]) in kws.keywords:
                    queries.append(word_id)
                    if len(queries) >= 5:
                        break
        if len(queries) >= 5:
            break

    if not queries:
        import random
        query_candidates = [wid for wid in kws.transcriptions.keys() if wid.startswith(kws.train_docs[0])]
        queries = random.sample(query_candidates, min(5, len(query_candidates)))

    print(f"Selected {len(queries)} queries for evaluation")
    print(f"Queries: {[f'{q} ({kws._get_clean_word(kws.transcriptions[q])})' for q in queries]}")
    print(f"Evaluating on {len(kws.val_docs[:2])} documents")

    mean_ap = kws.evaluate(queries, kws.val_docs[:2])
    print(f"\nFinal Mean Average Precision (mAP): {mean_ap:.4f}")

    print("\n=== Creating Test Output ===")
    kws.create_test_output()
    

if __name__ == "__main__":
    main()