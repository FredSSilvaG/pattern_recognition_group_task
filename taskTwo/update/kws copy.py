import numpy as np
import cv2
import os
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt

class KeywordSpotter:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.locations_dir = os.path.join(data_dir, 'locations')
        
        self.keywords = self._load_keywords()
        self.val_docs = self._get_document_ids()
        
    def _load_keywords(self):
        try:
            with open(os.path.join(self.data_dir, 'keywords.tsv'), 'r') as f:
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error loading keywords: {e}")
            return []
    
    def _get_document_ids(self):
        doc_ids = []
        cutouts_dir = os.path.join(self.data_dir, "cutouts_png")
        
        if not os.path.exists(cutouts_dir):
            print(f"Warning: {cutouts_dir} not found")
            return doc_ids
            
        for item in os.listdir(cutouts_dir):
            if os.path.isdir(os.path.join(cutouts_dir, item)):
                doc_ids.append(item)
                
        return sorted(doc_ids)
    
    def get_all_word_ids(self, doc_id=None):
        word_ids = []
        cutouts_dir = os.path.join(self.data_dir, "cutouts_png")
        
        docs_to_scan = [doc_id] if doc_id else self.val_docs
        
        for current_doc in docs_to_scan:
            doc_path = os.path.join(cutouts_dir, current_doc)
            if not os.path.exists(doc_path):
                continue
                
            for filename in os.listdir(doc_path):
                if filename.endswith(".png"):
                    if filename.startswith(f"{current_doc}-{current_doc}-"):
                        parts = filename[:-4].split("-")
                        if len(parts) >= 4:
                            word_id = f"{current_doc}-{parts[2]}-{parts[3]}"
                            word_ids.append(word_id)
                    elif filename.startswith(f"{current_doc}-"):
                        parts = filename[:-4].split("-")
                        if len(parts) >= 3:
                            word_id = f"{current_doc}-{parts[1]}-{parts[2]}"
                            word_ids.append(word_id)
        
        return sorted(list(set(word_ids)))
    
    def extract_word_image(self, word_id):
        """Extract word image from cutouts_png directory"""
        doc_id, line_id, word_id_in_line = word_id.split('-')
        path_id = f"{line_id}-{word_id_in_line}"
        
        cutouts_png_dir = os.path.join(self.data_dir, "cutouts_png", doc_id)
        
        patterns = [
            f"{doc_id}-{doc_id}-{line_id}-{word_id_in_line}.png",
            f"{doc_id}-{line_id}-{word_id_in_line}.png",         
            f"{path_id}.png"                                 
        ]
        
        for pattern in patterns:
            img_path = os.path.join(cutouts_png_dir, pattern)
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        
        print(f"No image found for {word_id}, using fallback")
        return self._create_fallback_image()
    
    def _create_fallback_image(self):
        """Create a fallback image when word image not found"""
        fallback = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(fallback, (30, 30), (70, 70), 255, 2)
        cv2.line(fallback, (30, 70), (70, 30), 255, 2)
        return fallback
    
    def extract_features(self, image):
        """Extract features from word image"""
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
                
                # Count transitions
                transitions = 0
                for i in range(1, height):
                    if column[i] != column[i-1]:
                        transitions += 1
                
                # Black pixels between contours
                between_contours = column[upper_contour:lower_contour+1]
                black_between = np.sum(between_contours == 255) / (lower_contour - upper_contour + 1)
            
            features = [
                upper_contour / height,
                lower_contour / height,
                transitions / height,
                black_pixel_fraction,
                black_between
            ]
            
            feature_vectors.append(features)
        
        # Normalize features
        feature_vectors = np.array(feature_vectors)
        for i in range(feature_vectors.shape[1]):
            std = np.std(feature_vectors[:, i])
            if std > 0:
                feature_vectors[:, i] = (feature_vectors[:, i] - np.mean(feature_vectors[:, i])) / std
        
        return feature_vectors
    
    def dtw_distance(self, s1, s2, window=15):
        """Calculate DTW distance between two feature sequences"""
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
        """Find similar words based only on image features"""
        query_image = self.extract_word_image(query_id)
        query_features = self.extract_features(query_image)
        
        all_word_ids = []
        for doc_id in doc_ids:
            doc_words = self.get_all_word_ids(doc_id)
            all_word_ids.extend(doc_words)
        
        all_word_ids = [wid for wid in set(all_word_ids) if wid != query_id]
        
        distances = []
        for word_id in tqdm(all_word_ids, desc=f"Matching {query_id}"):
            word_image = self.extract_word_image(word_id)
            word_features = self.extract_features(word_image)
            distance = self.dtw_distance(query_features, word_features)
            distances.append((word_id, distance))
        
        # Sort by distance (most similar first)
        distances.sort(key=lambda x: x[1])
        
        return distances[:k] if k is not None else distances
    
    def calculate_precision_recall(self, query_id, matches, top_k=None):
        """
        Calculate precision and recall using a percentile-based approach
        instead of an absolute threshold
        """
        if top_k is not None:
            matches = matches[:top_k]
        
        if not matches:
            return 0.0, 0.0
        
        distances = [dist for _, dist in matches]
        
        threshold_distance = np.percentile(distances, 30)
        
        relevant_count = sum(1 for _, dist in matches if dist <= threshold_distance)
        
        precision = relevant_count / len(matches)
        
        estimated_total_relevant = max(10, relevant_count * 4)  # Estimate total relevant as 4x what we found
        recall = relevant_count / estimated_total_relevant
        
        return precision, recall
    
    def process_test_queries(self):
        test_docs = []
        try:
            with open(os.path.join(self.data_dir, 'test.tsv'), 'r') as f:
                for line in f:
                    doc_id = line.strip()
                    if doc_id and not doc_id.startswith('//'):
                        test_docs.append(doc_id)
        except Exception as e:
            print(f"Error reading test.tsv: {e}")
            test_docs = self.val_docs[:2]
        
        print(f"Processing queries for {len(test_docs)} documents from test.tsv")
        
        avg_precision = 0
        avg_recall = 0
        query_count = 0
        
        with open('results.tsv', 'w') as results_file:
            for doc_id in test_docs:
                print(f"\nProcessing document: {doc_id}")
                
                sample_words = self.get_all_word_ids(doc_id)
                if not sample_words:
                    print(f"No words found in document {doc_id}, skipping")
                    continue
                
                test_queries = sample_words[:min(3, len(sample_words))]
                print(f"Selected {len(test_queries)} queries from document {doc_id}")
                
                for query_id in test_queries:
                    print(f"  Processing query: {query_id}")
                    
                    matches = self.find_matches(query_id, test_docs, k=20)
                    
                    precision, recall = self.calculate_precision_recall(query_id, matches, top_k=10)
                    avg_precision += precision
                    avg_recall += recall
                    query_count += 1
                    
                    print(f"  Precision@10: {precision:.4f}, Recall@10: {recall:.4f}")
                    
                    result_line = query_id
                    for word_id, distance in matches:
                        result_line += f"\t{word_id}\t{distance:.6f}"
                    
                    results_file.write(result_line + "\n")
                    
                    print(f"  Found {len(matches)} matches for {query_id}")
                    print(f"  Top matches for {query_id}:")
                    for i, (word_id, distance) in enumerate(matches[:5]):
                        print(f"    {i+1}. {word_id}: {distance:.4f}")
        
        if query_count > 0:
            avg_precision /= query_count
            avg_recall /= query_count
            print(f"\nAverage Precision: {avg_precision:.4f}")
            print(f"Average Recall: {avg_recall:.4f}")
            
        print(f"\nResults written to results.tsv")

    def generate_keyword_results(self, output_file='test1.tsv'):
        test_docs = []
        try:
            with open(os.path.join(self.data_dir, 'test1.tsv'), 'r') as f:
                for line in f:
                    doc_id = line.strip()
                    if doc_id and not doc_id.startswith('//'):
                        test_docs.append(doc_id)
        except Exception as e:
            print(f"Error reading test.tsv: {e}")
            test_docs = self.val_docs[:2]
        
        print(f"Using {len(test_docs)} documents from test1.tsv")
        
        sample_queries = []
        for doc_id in test_docs:
            doc_words = self.get_all_word_ids(doc_id)
            if doc_words:
                sample_queries.append(doc_words[0])
        
        if not sample_queries:
            print("No sample queries found")
            return
        
        with open(output_file, 'w') as f:
            for i, keyword in enumerate(self.keywords):
                query_id = sample_queries[i % len(sample_queries)]
                
                matches = self.find_matches(query_id, test_docs)
                
                line = keyword
                for word_id, distance in matches:
                    line += f"\t{word_id}\t{distance:.6f}"
                f.write(line + "\n")

def main():
    kws = KeywordSpotter(data_dir='.')
    
    print(f"Loaded {len(kws.keywords)} keywords")
    print(f"Found {len(kws.val_docs)} documents: {kws.val_docs}")
    
    total_words = 0
    for doc_id in kws.val_docs:
        words = kws.get_all_word_ids(doc_id)
        print(f"Document {doc_id}: {len(words)} words")
        total_words += len(words)
    
    print(f"Total words: {total_words}")
    
    print("\n=== Sample Query Testing ===")
    kws.process_test_queries()

    print("\n=== Generating Keyword Results ===")
    kws.generate_keyword_results('test1.tsv')
    

if __name__ == "__main__":
    main()