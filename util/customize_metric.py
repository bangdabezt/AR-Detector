import numpy as np

def calculate_iou(box1, box2):
    """ Calculate IoU between two bounding boxes [x_min, y_min, x_max, y_max]. """
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def calculate_ap(predictions, ground_truths, iou_threshold=0.5, score_threshold=0.5):
    """
    Calculate Average Precision (AP) for a set of predictions.
    
    :param predictions: List of predictions, each with bounding box coordinates and score.
    [[x_min, y_min, x_max, y_max, score, ...], ...]
    :param ground_truths: List of ground truth bounding boxes.
    [[x_min, y_min, x_max, y_max, score, ...]]
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
    true_positives = np.cumsum(true_positives)
    false_positives = np.cumsum(false_positives)
    
    precision = true_positives / (true_positives + false_positives + 1e-6)

    # Calculate Average Precision (AP) as the area under the Precision-Recall curve
    # ap = np.trapz(precision, recall)
    
    return precision

def calculate_curve_ap(predictions, ground_truths):
    pass

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
