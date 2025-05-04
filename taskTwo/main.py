import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import defaultdict
from tqdm import tqdm
import cv2



from common import param_utils
import importlib
importlib.reload(param_utils)

from utils import decode_utils, image_utils,score_utils
import dtw


def read_transcriptions(path):
    transcriptions = {}
    with open(path, 'r') as f:
        for line in f:
            word_id, text = line.strip().split('\t')
            transcriptions[word_id] = decode_utils.decode_transcription(text)
    return transcriptions


def build_template_library(train_ids, images_dir, locations_dir, transcriptions, keywords):
    templates = defaultdict(list)
    for doc_id in tqdm(train_ids, desc="Evaluating Build Tempates"):
        image_path = os.path.join(images_dir, f"{doc_id}.jpg")
        svg_path = os.path.join(locations_dir, f"{doc_id}.svg")

        try:
            boxes = image_utils.parse_word_polygons(svg_path)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            for word_id, polygon in boxes.items():
                if word_id not in transcriptions:
                    continue
                text = transcriptions[word_id]
                if text in keywords:
                    word_img = image_utils.crop_word_image(image, polygon)
                    features = dtw.extract_hog_sequence(word_img)
                    if word_img is None or word_img.size == 0:
                        print(f"[WARN] Empty image: {word_id}")
                        continue
                    templates[text].append(features)
        except Exception as e:
            print(f"[ERROR] Failed to process {doc_id}: {e}")

    print(f"[INFO] Loaded {len(templates)} keyword images")        

    return templates

def run_dtw_matching(validation_ids, images_dir, locations_dir, transcriptions, keywords, templates, score_threshold):
    results = []
    for doc_id in tqdm(validation_ids, desc="Evaluating Validation DTW Matching"):
        image_path = os.path.join(images_dir, f"{doc_id}.jpg")
        svg_path = os.path.join(locations_dir, f"{doc_id}.svg")

        try:
            boxes = image_utils.parse_word_polygons(svg_path)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            for word_id, polygon in tqdm(boxes.items(), desc="Evaluating"):
                if word_id not in transcriptions:
                    continue
                true_text = transcriptions[word_id]
                # if true_text not in keywords:
                #     continue
                word_img = image_utils.crop_word_image(image, polygon)
                test_feat = dtw.extract_hog_sequence(word_img)

                best_keyword = None
                best_score = float("inf")

                for kw, template_list in templates.items():
                    for template in template_list:
                        score = dtw.dtw_distance(template, test_feat)
                        if score < best_score:
                            best_score = score
                            best_keyword = kw

                if score_threshold is None or best_score <= score_threshold:          
                    results.append((word_id, true_text, best_keyword, best_score))
        except Exception as e:
            print(f"[ERROR] Failed to process {doc_id}: {e}")

    return results


def main() -> None:

    args = param_utils.parse_args_KWS()

    # load data
    train_images = pd.read_csv(args.train, sep='\t', header=None)[0].tolist()
    validation_images = pd.read_csv(args.validation, sep='\t',header=None)[0].tolist()

    # load transcriptions
    transcriptions = read_transcriptions(args.trainscription)

    # load keywords
    keywords = pd.read_csv(args.keywords, sep='\t', header=None)[0].apply(decode_utils.decode_transcription).tolist()

    # get the templates of keywords from train data
    templates = build_template_library(train_images, args.images, args.locations,transcriptions, keywords)

    # validate the validation_images, run so slow   
    results = run_dtw_matching(validation_images, args.images, args.locations, transcriptions, keywords, templates, 140)
    # for word_id, true, pred, score in results:
    #     if (true == pred):
    #         print(f"True Positive: true={true}, pred={pred}, score={score}")
    #     else:
    #         print(f"False Positive: true={true}, pred={pred}, score={score}")    
        

    score_utils.precision_recall(results, keywords)
    pr_curves, ap_scores = score_utils.compute_pr_curve(results, keywords)
    score_utils.plot_pr_curves(pr_curves, ap_scores)




    # label_map = group_words(transcriptions_df)
    # pairs = generate_pairs(label_map, negatives_per_pos=1)
    # # print(pairs)

    # correct = 0
    # total = len(pairs)

    # for a, b, label in tqdm(pairs, desc="Evaluating"):
        
    #     path_a = os.path.join("word_images", f"{a}.jpg")
    #     path_b = os.path.join("word_images", f"{b}.jpg")
    #     try:
    #         seq1 = dtw.extract_hog_sequence(path_a)
    #         seq2 = dtw.extract_hog_sequence(path_b)
    #         distance_matrix = dtw.compute_distance_matrix(seq1, seq2)
    #         dist = dtw.dtw(distance_matrix)
    #         #dist = dtw.dtw_distance(seq1,seq2)
    #         print(dist)
    #         pred = 1 if dist < 50 else 0  # fixed threshold, tune this
    #         correct += (pred == label)
    #     except Exception as e:
    #         total -= 1 
    #         print(f"Error processing {a} and {b}: {e}")
    # print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")
    








if __name__ == "__main__":
    main()








