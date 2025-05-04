from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def precision_recall(results, keywords):
    per_label_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    
    for _, true, pred, _ in results:
        if pred in keywords:
            if pred == true:
                per_label_stats[pred]["TP"] += 1
            else:
                per_label_stats[pred]["FP"] += 1
        if true in keywords and pred != true:
            per_label_stats[true]["FN"] += 1

    print("\n[INFO] Precision / Recall per keyword:")
    total_TP = total_FP = total_FN = 0
    for kw in sorted(keywords):
        TP = per_label_stats[kw]["TP"]
        FP = per_label_stats[kw]["FP"]
        FN = per_label_stats[kw]["FN"]
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        print(f" - {kw:<15} Precision: {prec:.2f}, Recall: {rec:.2f}")
        total_TP += TP
        total_FP += FP
        total_FN += FN

    # Overall average (macro)
    macro_p = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    macro_r = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    print(f"\n[INFO] Overall Precision: {macro_p:.2f}, Recall: {macro_r:.2f}")


def compute_pr_curve(results, keywords):
    pr_curves = {}
    ap_scores = {}

    for kw in keywords:
        ranked = []
        for _, true, pred, score in results:
            if pred == kw:
                is_correct = (true == kw)
                ranked.append((is_correct, score))

        ranked.sort(key=lambda x: x[1])

        tp, fp = 0, 0
        precisions = []
        recalls = []
        total_positives = sum(1 for _, true, _, _ in results if true == kw)

        for correct, _ in ranked:
            if correct:
                tp += 1
            else:
                fp += 1
            precision = tp / (tp + fp)
            recall = tp / total_positives if total_positives > 0 else 0
            precisions.append(precision)
            recalls.append(recall)

        ap = 0.0
        for r in np.linspace(0, 1, 11):
            p = max([prec for prec, rec in zip(precisions, recalls) if rec >= r] + [0])
            ap += p / 11.0

        pr_curves[kw] = (recalls, precisions)
        ap_scores[kw] = ap

    return pr_curves, ap_scores


def plot_pr_curves(pr_curves, ap_scores):
    plt.figure(figsize=(8, 6))
    for kw, (recalls, precisions) in pr_curves.items():
        plt.plot(recalls, precisions, label=f"{kw} (AP={ap_scores[kw]:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (per keyword)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()