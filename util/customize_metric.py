import numpy as np

def calculate_iou(box1, box2):
    """ Calculate IoU between two bounding boxes [x_min, y_min, x_max, y_max]. """
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def summarize_ap(list_all_preds, list_all_gts, iou_threshold=0.5, score_threshold=0.5):
    all_ap = []
    for preds, gts in zip(list_all_preds, list_all_gts):
        sample_ap = calculate_ap(preds, gts, iou_threshold, score_threshold)
        all_ap.append(sample_ap)
    return sum(all_ap) / len(all_ap)

def calculate_ap(predictions, ground_truths, iou_threshold=0.5, score_threshold=0.5):
    """
    Calculate Average Precision (AP) for a set of predictions.
    
    :param predictions: List of predictions, each with bounding box coordinates and score.
    [[x_min, y_min, x_max, y_max, score, ...], ...]
    :param ground_truths: List of ground truth bounding boxes.
    [[x_min, y_min, x_max, y_max, ...]]
    :param iou_threshold: IoU threshold for positive sample classification.
    :param score_threshold: Score threshold for determining positive vs. negative prediction.
    :return: Average Precision (AP) score.
    """
    true_positives = []
    false_positives = []
    false_negatives = []
    true_negatives = []
    num_pos = len(ground_truths)

    for id_pred, prediction in enumerate(predictions):
        bbox, score = prediction[:4], prediction[4]
        
        if id_pred < num_pos:  # This prediction is on positive sample
            iou = calculate_iou(bbox, ground_truths[id_pred][:4])
            # check if this prediction is true
            if iou >= iou_threshold and score >= score_threshold:
                true_positives.append(1)
                false_positives.append(0)
                false_negatives.append(0)
                true_negatives.append(0)
            elif iou >= iou_threshold:
                true_positives.append(0)
                false_positives.append(1)
                false_negatives.append(0)
                true_negatives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(0)
                false_negatives.append(1)
                true_negatives.append(0)
        else:  # Negative sample (no ground truth)
            if score < score_threshold:
                true_positives.append(0)
                false_positives.append(0)
                false_negatives.append(0)
                true_negatives.append(1)
            else:
                true_positives.append(0)
                false_positives.append(1)
                false_negatives.append(0)
                true_negatives.append(0)

    # Cumulative sum to calculate precision and recall at each prediction
    # true_positives = np.cumsum(true_positives)
    # false_positives = np.cumsum(false_positives)
    TP = 1.0 * sum(true_positives)
    FP = 1.0 * sum(false_positives)
    precision = TP / (TP + FP + 1e-6)
    # recall = true_positives / num_pos
    # # Calculate Average Precision (AP) as the area under the Precision-Recall curve
    # ap = np.trapz(precision, recall)
    # if FP > 0:
    #     print(FP)
    return precision, FP

def calculate_curve_ap(predictions, ground_truths):
    pass

def calculate_mAP(predictions, ground_truths, iou_threshold=0.5, score_threshold=0.8):
    """
    Calculate Average Precision (AP) for a set of predictions.

    :param predictions: List of predictions, each with bounding box coordinates and score.
    [[x_min, y_min, x_max, y_max, score, ...], ...]
    :param ground_truths: List of ground truth bounding boxes.
    [[x_min, y_min, x_max, y_max, ...]]
    :param iou_threshold: IoU threshold for positive sample classification.
    :param score_threshold: Score threshold for determining positive vs. negative prediction.
    :return: Average Precision (AP) score.
    """
    # Sort the list by score in descending order, keeping track of original indices
    sorted_boxes = sorted(enumerate(predictions), key=lambda x: x[1][4], reverse=True)

    # Find the position of the element with the original index 0 (score1)
    order_of_score1 = next(i for i, (index, _) in enumerate(sorted_boxes) if index == 0)
    order_S = order_of_score1 + 1
    true_positives = []
    false_positives = []
    false_negatives = []
    true_negatives = []
    num_pos = len(ground_truths)

    for id_pred, prediction in enumerate(predictions):
        bbox, score = prediction[:4], prediction[4]

        if id_pred < num_pos:  # This prediction is on positive sample
            iou = calculate_iou(bbox, ground_truths[id_pred][:4])
            # check if this prediction is true
            if iou >= iou_threshold and score >= score_threshold:
                true_positives.append(1)
                false_positives.append(0)
                false_negatives.append(0)
                true_negatives.append(0)
            elif iou >= iou_threshold:
                true_positives.append(0)
                false_positives.append(1)
                false_negatives.append(0)
                true_negatives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(0)
                false_negatives.append(1)
                true_negatives.append(0)
        else:  # Negative sample (no ground truth)
            if score < score_threshold:
                true_positives.append(0)
                false_positives.append(0)
                false_negatives.append(0)
                true_negatives.append(1)
            else:
                true_positives.append(0)
                false_positives.append(1)
                false_negatives.append(0)
                true_negatives.append(0)

    # Cumulative sum to calculate precision and recall at each prediction
    # true_positives = np.cumsum(true_positives)
    # false_positives = np.cumsum(false_positives)
    TP = 1.0 * sum(true_positives)
    return TP/order_S

def calculate_recall(predictions, ground_truths, iou_threshold=0.5, score_threshold=0.8, K=1):
    # Get score1 (the score of the first element)
    score1 = predictions[0][4]

    # Extract all scores
    scores = [box[4] for box in predictions]

    # Sort scores in descending order
    sorted_scores = sorted(scores, reverse=True)

    # Check if score1 is in the top K scores
    is_score1_in_top_k = score1 >= sorted_scores[K-1] if K <= len(scores) else False
    
    # if not is_score1_in_top_k:
    #     import pdb; pdb.set_trace()
    # iou = calculate_iou(predictions[0][:4], ground_truths[0][:4])
    if is_score1_in_top_k:
        return 1.0
    else:
        return 0.0
# Example usage
# predictions = [
#     # Format: [x_min, y_min, x_max, y_max, score]
#     [50, 50, 100, 100, 0.9],  # Example positive prediction
#     [30, 30, 80, 80, 0.2],    # Example negative prediction
# ]
# ground_truths = [
#     # Format: [x_min, y_min, x_max, y_max]
#     [55, 55, 95, 95]           # Ground truth box
# ]

# ap = calculate_ap(predictions, ground_truths, iou_threshold=0.5, score_threshold=0.5)
# print(f"Average Precision (AP): {ap:.4f}")
import json
import ast
import os
def summarize_metric(file_path, score = [0.8]):#[0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]):
    data = []
    hack_ap = []
    # Open and read the file line by line
    with open(file_path, 'r') as fdata:
        for line in fdata:
            # Remove any leading/trailing whitespace (like newline characters)
            line = line.strip()
            if line:  # Ensure line is not empty
                # Parse each line as a dictionary and add it to the list
                data.append(ast.literal_eval(line))
    for score_threshold in score:
        all_AP = []
        all_mAP = []
        all_recall = []
        pred_ratio = []
        score_ratio = []
        for data_sample in data:
            given_box = data_sample['gt_box'][0]
            # highest_iou_box = max(data_sample['results_unsorted'], key=lambda box: calculate_iou(given_box, box))
            new_list = data_sample['results'] #[highest_iou_box] + [box for box in data_sample['results_unsorted'] if box != highest_iou_box]
            highest_iou_box = new_list[0]
            # import pdb; pdb.set_trace()
            my_ap = calculate_ap(new_list, [given_box], score_threshold=score_threshold)
            all_AP.append(my_ap[0])
            hack_ap.append(my_ap[1])
            all_mAP.append(calculate_mAP(new_list, [given_box], score_threshold=score_threshold))
            all_recall.append(calculate_recall(new_list, [given_box], score_threshold=score_threshold))
            if calculate_iou(given_box, highest_iou_box) >= 0.5:
                pred_ratio.append(1.0)
            else:
                pred_ratio.append(0.0)
            if highest_iou_box[4] > score_threshold:
                score_ratio.append(1.0)
            else:
                score_ratio.append(0.0)
        print("Summarize for score_threshold={}: AP={}, mAP={}, R@1={}".format(score_threshold, 
                                                                               (sum(all_AP) / len(all_AP)), 
                                                                               (sum(all_mAP) / len(all_mAP)), 
                                                                               (sum(all_recall) / len(all_recall))))
        print("Proportion of correct bbox pred: {}; Score: {}".format(sum(pred_ratio)/len(pred_ratio), 
                                                                                              sum(score_ratio)/len(score_ratio)))
        # print(hack_ap)
    all_K = [3, 5]
    all_recall_0 = [[] for _ in all_K]
    for data_sample in data:
        given_box = data_sample['gt_box'][0]
        new_list = data_sample['results'] #[highest_iou_box] + [box for box in data_sample['results_unsorted'] if box != highest_iou_box]
        highest_iou_box = new_list[0]
        for idx in range(len(all_K)):
            all_recall_0[idx].append(calculate_recall(new_list, [given_box], K=all_K[idx]))
    print("All recall: R@{}={}, R@{}={}".format(all_K[0], (sum(all_recall_0[0]) / len(all_recall_0[0])),
                                                all_K[1], (sum(all_recall_0[1]) / len(all_recall_0[1]))))
    return hack_ap
