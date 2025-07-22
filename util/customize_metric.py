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
    # import pdb; pdb.set_trace()
    # if not is_score1_in_top_k: ## all_loss: 66 
    #     import pdb; pdb.set_trace()
    # iou = calculate_iou(predictions[0][:4], ground_truths[0][:4])
    if is_score1_in_top_k: # and score1 > 0.5:
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

def calculate_average_precision(predictions, ground_truths, iou_threshold=0.5, score_threshold=0.5):
    """
    Calculate Average Precision for a one-to-one correspondence between predictions and ground truths.
    Each ground_truths[i] corresponds to predictions[i].

    Parameters:
        predictions: List of predictions [x_min, y_min, x_max, y_max, score]
        ground_truths: List of ground truths [x_min, y_min, x_max, y_max]
        iou_threshold: IoU threshold to determine true positives
        score_threshold: Score threshold to filter predictions

    Returns:
        Average Precision (AP)
    """
    assert len(predictions) == len(ground_truths), "Predictions and ground truths must have the same length."

    # Filter predictions by score
    filtered_predictions = [pred for pred in predictions if pred[-1] >= score_threshold]
    filtered_ground_truths = [ground_truths[i] for i, pred in enumerate(predictions) if pred[-1] >= score_threshold]

    # Sort predictions by confidence score (descending)
    sorted_indices = sorted(range(len(filtered_predictions)), key=lambda i: filtered_predictions[i][-1], reverse=True)
    # sorted_indices = sorted_indices[11:]
    sorted_predictions = [filtered_predictions[i] for i in sorted_indices]
    sorted_ground_truths = [filtered_ground_truths[i] for i in sorted_indices]
    # import pdb; pdb.set_trace()
    # plot_prediction(filtered_predictions, filtered_ground_truths, sorted_indices[:24])
    new_gts = []
    for new_gt in sorted_ground_truths:
        if new_gt != [0.0, 0.0, 0.0, 0.0]:
            new_gts.append(new_gt)
    ##
    # last_idx = -1
    # for idx, grtruth in enumerate(sorted_ground_truths):
    #     if grtruth != [0.0, 0.0, 0.0, 0.0]: last_idx = idx
    # print('Last index:{}'.format(last_idx))
    # list_new_idx = []
    # for idx, grtruth in enumerate(sorted_ground_truths):
    #     if idx <= last_idx:
    #         if grtruth != [0.0, 0.0, 0.0, 0.0]: list_new_idx.append(idx)
    #     else:
    #         list_new_idx.append(idx)
    # sorted_predictions = [sorted_predictions[i] for i in list_new_idx]
    # sorted_ground_truths = [sorted_ground_truths[i] for i in list_new_idx]
    # Evaluate TPs and FPs
    tp = []
    fp = []
    for pred, gt in zip(sorted_predictions, sorted_ground_truths):
        iou = calculate_iou(pred[:4], gt)
        if iou >= iou_threshold:
            tp.append(1)  # True positive
            fp.append(0)  # Not a false positive
        else:
            tp.append(0)  # Not a true positive
            fp.append(1)  # False positive

    # Cumulative sums for precision and recall
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    total_gt = len(new_gts) #len(ground_truths)
    precision = tp / (tp + fp)
    recall = tp / total_gt
    # print(precision)
    # import pdb; pdb.set_trace()
    # Compute AP using the trapezoidal rule
    recall_diff = np.diff(np.concatenate(([0], recall, [1])))
    precision_interpolated = np.maximum.accumulate(np.concatenate(([0], precision, [0]))[::-1])[::-1]
    ap = np.sum(precision_interpolated[:-1] * recall_diff)

    return ap, precision, recall
  
def calculate_map(predictions, ground_truths, iou_threshold=0.5, score_thresholds=np.linspace(0.0, 1.0, 11), fig_name=None):
    """Compute mAP across multiple score thresholds."""
    aps = []
    for score_threshold in score_thresholds:
        ap, prec, recall = calculate_average_precision(predictions, ground_truths, iou_threshold, score_threshold)
        aps.append(ap)
        if fig_name is not None:
            plot_precision_recall_curve(prec, recall, fig_name + '.png')
        # if score_threshold == 0.6:
        #     print('AP{}:{}'.format(score_threshold, ap))
        # if score_threshold == 0.8:
        #     print('AP{}:{}'.format(score_threshold, ap))
    print(aps)
    return np.mean(aps)

def summarize_AP(file_path, num_pred = 100):
    all_preds = []
    all_gts = []
    # Open and read the file line by line
    with open(file_path, 'r') as fdata:
        for line in fdata:
            # Remove any leading/trailing whitespace (like newline characters)
            line = line.strip()
            if line:  # Ensure line is not empty
                # Parse each line as a dictionary and add it to the list
                data = ast.literal_eval(line)
                gt_box = data['gt_box'][0]
                unscaled_pred_score = []
                for idx, prediction in enumerate(data['results']):
                    if idx % 100 >= num_pred: continue
                    # if idx > 0: break
                    # if prediction[-1] > data['results'][0][-1]: continue
                    # all_preds.append(prediction)
                    unscaled_pred_score.append(prediction)
                    if idx < num_pred:
                        all_gts.append(gt_box)
                    else:
                        all_gts.append([0.0, 0.0, 0.0, 0.0])
                # scaled_pred_score = postprocess_score(unscaled_pred_score)
                all_preds = all_preds + unscaled_pred_score
    # all_map = []
    # for iou_thres in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    #     all_map.append(calculate_map(all_preds, all_gts, iou_threshold=iou_thres, score_thresholds=[0.0]))
    print('mAP: {}'.format(calculate_map(all_preds, all_gts, score_thresholds=[0.0], fig_name=file_path.split('/')[2])))
    # print(all_map)
    # print(sum(all_map) / 10)

def summarize_diff_AP(file_path):
    # all_preds = []
    # all_gts = []
    all_aps = []
    # Open and read the file line by line
    with open(file_path, 'r') as fdata:
        for line in fdata:
            # Remove any leading/trailing whitespace (like newline characters)
            line = line.strip()
            if line:  # Ensure line is not empty
                # Parse each line as a dictionary and add it to the list
                data = ast.literal_eval(line)
                # gt_box = data['gt_box'][0]
                len_gt = len(data['gt_box'])
                all_preds = []
                all_gts = []
                for idx, prediction in enumerate(data['results']):
                    # if idx % 100 >= num_pred: continue
                    # if idx > 0: break
                    # if prediction[-1] > data['results'][0][-1]: continue
                    all_preds.append(prediction)
                    # unscaled_pred_score.append(prediction)
                    if idx < len_gt:
                        all_gts.append(data['gt_box'][idx])
                    else:
                        all_gts.append([0.0, 0.0, 0.0, 0.0])
                # scaled_pred_score = postprocess_score(unscaled_pred_score)
                all_aps.append(calculate_ap_per_class(all_preds, all_gts))
    # all_map = []
    print(all_aps)
    # for iou_thres in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    #     all_map.append(calculate_map(all_preds, all_gts, iou_threshold=iou_thres, score_thresholds=[0.0]))
    print('mAP: {}'.format(sum(all_aps) / len(all_aps)))

def calculate_ap_per_class(predictions, ground_truths, iou_threshold=0.5, score_threshold=0.0):
    """
    Calculate Average Precision for a one-to-one correspondence between predictions and ground truths.
    Each ground_truths[i] corresponds to predictions[i].

    Parameters:
        predictions: List of predictions [x_min, y_min, x_max, y_max, score]
        ground_truths: List of ground truths [x_min, y_min, x_max, y_max]
        iou_threshold: IoU threshold to determine true positives
        score_threshold: Score threshold to filter predictions

    Returns:
        Average Precision (AP)
    """
    assert len(predictions) == len(ground_truths), "Predictions and ground truths must have the same length."

    # Filter predictions by score
    filtered_predictions = [pred for pred in predictions if pred[-1] >= score_threshold]
    filtered_ground_truths = [ground_truths[i] for i, pred in enumerate(predictions) if pred[-1] >= score_threshold]

    # Sort predictions by confidence score (descending)
    sorted_indices = sorted(range(len(filtered_predictions)), key=lambda i: filtered_predictions[i][-1], reverse=True)
    # sorted_indices = sorted_indices[11:]
    sorted_predictions = [filtered_predictions[i] for i in sorted_indices]
    sorted_ground_truths = [filtered_ground_truths[i] for i in sorted_indices]
    # import pdb; pdb.set_trace()
    # plot_prediction(filtered_predictions, filtered_ground_truths, sorted_indices[:24])
    new_gts = []
    for new_gt in sorted_ground_truths:
        if new_gt != [0.0, 0.0, 0.0, 0.0]:
            new_gts.append(new_gt)
    # Evaluate TPs and FPs
    tp = []
    fp = []
    for pred, gt in zip(sorted_predictions, sorted_ground_truths):
        iou = calculate_iou(pred[:4], gt)
        if iou >= iou_threshold:
            tp.append(1)  # True positive
            fp.append(0)  # Not a false positive
        else:
            tp.append(0)  # Not a true positive
            fp.append(1)  # False positive

    # Cumulative sums for precision and recall
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    total_gt = len(new_gts) #len(ground_truths)
    precision = tp / (tp + fp)
    recall = tp / total_gt
    # print(total_gt / len(sorted_ground_truths))
    # if (total_gt / len(sorted_ground_truths)) > 0.4:
    #     import pdb; pdb.set_trace()
    
    # Compute AP using the trapezoidal rule
    recall_diff = np.diff(np.concatenate(([0], recall, [1])))
    precision_interpolated = np.maximum.accumulate(np.concatenate(([0], precision, [0]))[::-1])[::-1]
    ap = np.sum(precision_interpolated[:-1] * recall_diff)

    return ap

# def summarize_APS(file_path):
#     all_preds = []
#     all_gts = []
#     # Open and read the file line by line
#     with open(file_path, 'r') as fdata:
#         for line in fdata:
#             # Remove any leading/trailing whitespace (like newline characters)
#             line = line.strip()
#             if line:  # Ensure line is not empty
#                 # Parse each line as a dictionary and add it to the list
#                 data = ast.literal_eval(line)
#                 gt_box = data['gt_box'][0]
#                 for idx, prediction in enumerate(data['results']):
#                     # if idx % 100 >= num_pred: continue
#                     if idx > 0: break
#                     all_preds.append(prediction)
#                     if idx < 1:
#                         all_gts.append(gt_box)
#                     else:
#                         all_gts.append([0.0, 0.0, 0.0, 0.0])
#     # all_map = []
#     # for iou_thres in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
#     #     all_map.append(calculate_map(all_preds, all_gts, iou_threshold=iou_thres))
#     print('mAP: {}'.format(calculate_map(all_preds, all_gts)))
#     # print(all_map)
#     # print(sum(all_map) / 10)

import matplotlib.pyplot as plt
import os
def plot_precision_recall_curve(precision, recall, save_path="precision_recall_curve.png"):
    """Plot and save the Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='o', label='PR Curve')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join('.', 'attribution_finetune', 'save_fig', save_path))  # Save the figure
    print(f"PR curve saved to {save_path}")
    plt.close()

import math
def postprocess_score(unscaled_pred_score):
    all_score = [pred[4] for pred in unscaled_pred_score]
    exp_values = [math.exp(v) for v in all_score]  # Exponentiate each value
    sum_exp = sum(exp_values)  # Sum of all exponentiated values
    return [[pred[0], pred[1], pred[2], pred[3] ,(v / sum_exp)] for pred, v in zip(unscaled_pred_score, exp_values)] 

import json
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
def plot_prediction(predictions, groundtruths, sample_idxs, pred_per_set = 1, data_anno='val_filename.jsonl', folder='./attribution_finetune/save_fig'):
    # AttributionGD/data/break-a-scene/coco2017/official_data/val_filename.jsonl
    query_root = './samples/break_a_scene/coco2017/query_set'
    target_root = './samples/break_a_scene/coco2017/target_set'
    anno_root = './data/break-a-scene/coco2017/official_data'
    save_folder = os.path.join(folder, 'predictions')
    # process the predictions and sample_idxs to get the indexes in data_anno
    neg_target_preds = []
    pos_target_preds = []
    ori_idxes = []
    neg_idxes = []
    for sample_idx in sample_idxs:
        ori_pos_idx = sample_idx // (91*pred_per_set)
        pos_idx = ori_pos_idx * (91*pred_per_set)
        ##
        neg_idxes.append((sample_idx - pos_idx)//pred_per_set)
        ori_idxes.append(ori_pos_idx)
        neg_target_preds.append([predictions[idx] for idx in range(sample_idx, sample_idx+pred_per_set)])
        pos_target_preds.append([predictions[idx] for idx in range(pos_idx, pos_idx+pred_per_set)])
    # plot image from data root
    unique_filenames = []
    data_annos = []
    with open(os.path.join(anno_root, data_anno), 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if data["filename"] not in unique_filenames:
                unique_filenames.append(data["filename"])
            data_annos.append(data)
    pos_samples = [data_annos[idx] for idx in ori_idxes]
    # target_samples = []
    for idx, sample in enumerate(pos_samples):
        temp_filenames = copy.copy(unique_filenames)
        temp_filenames.remove(sample['filename'])
        neg_filename = temp_filenames[neg_idxes[idx] - 1] if neg_idxes[idx] > 0 else sample['filename']
        pos_filename = sample['filename']
        if not os.path.exists(os.path.join(save_folder, str(sample_idxs[idx]))):
            os.makedirs(os.path.join(save_folder, str(sample_idxs[idx])), exist_ok=True)
        
        ## plot for neg_file ----------------------------------------------------------------------------------------
        image = Image.open(os.path.join(target_root, neg_filename))
        bounding_boxes = neg_target_preds[idx]
        # Plot the image
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Add the bounding boxes and scores
        for bbox in bounding_boxes:
            # Draw the bounding box
            x_min, y_min, width, height, score = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], bbox[4]
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add the score
            ax.text(x_min, y_min - 5, f"{score:.2f}", fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Remove axis
        plt.axis("off")

        # Save the figure
        output_path = os.path.join(save_folder, str(sample_idxs[idx]), "negative_{}.png".format(neg_idxes[idx]))
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Save with high quality
        plt.close()  # Close the figure to free memory

        ## plot for pos_file -----------------------------------------------------------------------------------------
        image = Image.open(os.path.join(target_root, pos_filename))
        bounding_boxes = pos_target_preds[idx]
        # Plot the image
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Add the bounding boxes and scores
        for bbox in bounding_boxes:
            # Draw the bounding box
            x_min, y_min, width, height, score = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], bbox[4]
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add the score
            ax.text(x_min, y_min - 5, f"{score:.2f}", fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Draw ground truth bounding boxes (in green)
        ground_truth_box = sample['detection']['instances'][0]['bbox']
        x_min, y_min, width, height = ground_truth_box[0], ground_truth_box[1], ground_truth_box[2]-ground_truth_box[0], ground_truth_box[3]-ground_truth_box[1]
        # for (x_min, y_min, width, height) in ground_truth_boxes:
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        # Remove axis
        plt.axis("off")

        # Save the figure
        output_path = os.path.join(save_folder, str(sample_idxs[idx]), "positive.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Save with high quality
        plt.close()  # Close the figure to free memory

        ## plot for query_file -----------------------------------------------------------------------------------------
        image = Image.open(os.path.join(query_root, sample['query_file']['filename']))
        bounding_boxes = sample['query_file']['exemplar']
        # Plot the image
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Add the bounding boxes and scores
        for bbox in bounding_boxes:
            # Draw the bounding box
            x_min, y_min, width, height = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add the score
            ax.text(x_min, y_min - 5, f"{sample['query_file']['category']}", fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Remove axis
        plt.axis("off")

        # Save the figure
        output_path = os.path.join(save_folder, str(sample_idxs[idx]), "query.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Save with high quality
        plt.close()  # Close the figure to free memory

## ranking metrics ---------------------------------------------------------------------------------------------------------
def calculate_NDCG_AP(predictions, ground_truths=None, K=5):
    # Get score1 (the score of the first element)
    score1 = predictions[0][4]

    # Extract all scores
    scores = [box[4] for box in predictions]

    # Sort scores in descending order
    sorted_scores = sorted(scores, reverse=True)

    # Check if score1 is in the top K scores
    top_K_scores = sorted_scores[:K]
    is_score1_in_top_k = score1 >= sorted_scores[K-1] if K <= len(scores) else False
    
    # if not is_score1_in_top_k:
    #     import pdb; pdb.set_trace()
    # iou = calculate_iou(predictions[0][:4], ground_truths[0][:4])
    if is_score1_in_top_k:
        rank_pred = top_K_scores.index(score1) + 1
        return 1.0 / rank_pred, 1.0 / math.log2(rank_pred + 1) # AP, DCG
    else:
        return 0.0, 0.0

def summarize_trainSet(file_path, mask_path=None):
    data = []
    mask_data = []
    # Open and read the file line by line
    with open(file_path, 'r') as fdata:
        for line_idx, line in enumerate(fdata):
            # Remove any leading/trailing whitespace (like newline characters)
            # if line_idx > 3335: break
            line = line.strip()
            if line:  # Ensure line is not empty
                # Parse each line as a dictionary and add it to the list
                data.append(ast.literal_eval(line))
    print('Total number of samples: {}'.format(line_idx))
    if mask_path != None:
        with open(mask_path, 'r') as mdata:
            for line_idx, line in enumerate(mdata):
                line = line.strip()
                if line:  # Ensure line is not empty
                    # Parse each line as a dictionary and add it to the list
                    mask_data.append(ast.literal_eval(line))
    # print('Total number of samples: {}'.format(line_idx))
    all_AP = []
    all_DCG = []
    all_recall_1 = []
    all_recall_3 = []
    all_recall_5 = []
    pred_ratio = []
    for dta_idx, data_sample in enumerate(data):
        # import pdb; pdb.set_trace()
        given_box = data_sample['gt_box'][0]
        # highest_iou_box = max(data_sample['results_unsorted'], key=lambda box: calculate_iou(given_box, box))
        new_list = data_sample['results'] #[highest_iou_box] + [box for box in data_sample['results_unsorted'] if box != highest_iou_box]
        ##
        if mask_data != []:
            new_mask = mask_data[dta_idx]['results'] 
            temp_list = []
            for bbox, mask in zip(new_list, new_mask):
                if mask == 1: temp_list.append(bbox)
            new_list = temp_list
        ##
        highest_iou_box = new_list[0]
        # import pdb; pdb.set_trace()
        ap, dcg = calculate_NDCG_AP(new_list, [given_box], K=5)
        all_AP.append(ap)
        all_DCG.append(dcg)
        # temp_rec = calculate_recall(new_list, [given_box], K=10)
        # if dta_idx == 20:
        #     import pdb; pdb.set_trace()
        all_recall_1.append(calculate_recall(new_list, [given_box], K=1))
        all_recall_3.append(calculate_recall(new_list, [given_box], K=5))
        all_recall_5.append(calculate_recall(new_list, [given_box], K=10))
        
        if calculate_iou(given_box, highest_iou_box) >= 0.5:
            pred_ratio.append(1.0)
        else:
            pred_ratio.append(0.0)
    print("Summarize: AP@5={}, DCG@5={}, R@1={}, R@3={}, R@5={}".format((sum(all_AP) / len(all_AP)), 
                                                                        (sum(all_DCG) / len(all_DCG)), 
                                                                        (sum(all_recall_1) / len(all_recall_1)),
                                                                        (sum(all_recall_3) / len(all_recall_3)),
                                                                        (sum(all_recall_5) / len(all_recall_5))))
    print("Proportion of correct bbox pred: {};".format(sum(pred_ratio)/len(pred_ratio)))

def figureOut_trainError(pretrain_path, finetune_path, ids_path, annotation_path, mask_path=None):
    query_root = './samples/break_a_scene/coco2017/query_set'
    target_root = './samples/break_a_scene/coco2017/target_set'
    fig_path = './attribution_finetune/save_fig'
    labels = ["teddy bear", "zebra", "horse", "cow", "elephant", "dog", "sheep", "cat", "bear", "bird"]
    annotations = {}
    mask_data = []
    # read annotations
    with open(annotation_path, 'r') as ann_file:
        data = json.load(ann_file)
    for ann_image in data['images']:
        annotations[ann_image['id']] = {'target_name': ann_image['file_name']}
    for ann_image in data['annotations']:
        ## target
        annotations[ann_image['image_id']]['target_box'] = ann_image['bbox']
        annotations[ann_image['image_id']]['label'] = labels[ann_image['category_id']]
        # query:
        annotations[ann_image['image_id']]['query_name'] = ann_image['query_file']['filename']
        annotations[ann_image['image_id']]['query_box'] = ann_image['query_file']['exemplar'][0]
    
    def posprocess_preds(pretrain_preds, finetune_preds, img_ids):
        # 1) Get the score of the first box
        pretrain_first_score = pretrain_preds[0][4] 
        finetune_first_score = finetune_preds[0][4] 

        # 2) Zip boxes and ids together, then sort by score in descending order
        pretrain_boxes_ids_sorted = sorted(zip(pretrain_preds, img_ids), key=lambda x: x[0][4], reverse=True)
        finetune_boxes_ids_sorted = sorted(zip(finetune_preds, img_ids), key=lambda x: x[0][4], reverse=True)

        # 3) Find all IDs where score > score_first_box
        pretrain_higher_score_ids = [box_id for (box, box_id) in pretrain_boxes_ids_sorted if box[4] > pretrain_first_score]
        finetune_higher_score_ids = [box_id for (box, box_id) in finetune_boxes_ids_sorted if box[4] > finetune_first_score]
        ## error ids:
        error_ids_list = []
        for id in finetune_higher_score_ids:
            if id not in pretrain_higher_score_ids:
                error_ids_list.append(id)
        return error_ids_list #[annotations[id] for id in error_ids_list]
    
    def load_data(path):
        temp_data = []
        # Open and read the file line by line
        with open(path, 'r') as fdata:
            for line_idx, line in enumerate(fdata):
                # Remove any leading/trailing whitespace (like newline characters)
                if line_idx > 3335: break
                line = line.strip()
                if line:  # Ensure line is not empty
                    # Parse each line as a dictionary and add it to the list
                    temp_data.append(ast.literal_eval(line))
        print('Total number of samples: {}'.format(line_idx))
        return temp_data

    def plot_box(boxes, texts, colors, src_path, des_path):
        ## -----------------------------------------------------------------------------------------
        image = Image.open(src_path)
        # Plot the image
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Add the bounding boxes and scores
        for idx, bbox in enumerate(boxes):
            # Draw the bounding box
            x_min, y_min, width, height, score = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=colors[idx], facecolor='none')
            ax.add_patch(rect)
            
            # Add the score
            if texts[idx] != '':
                ax.text(x_min, y_min - 5, texts[idx], fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        # Remove axis
        plt.axis("off")

        # Save the figure
        plt.savefig(des_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Save with high quality
        plt.close()  # Close the figure to free memory
    
    pretrain_data = load_data(pretrain_path)
    finetune_data = load_data(finetune_path)
    ids_data = load_data(ids_path)
    # read mask data
    if mask_path != None:
        with open(mask_path, 'r') as mdata:
            for line_idx, line in enumerate(mdata):
                line = line.strip()
                if line:  # Ensure line is not empty
                    # Parse each line as a dictionary and add it to the list
                    mask_data.append(ast.literal_eval(line))
    
    # all_recall = []
    for dta_idx, pretrain_sample in enumerate(pretrain_data):
        # import pdb; pdb.set_trace()
        # given_box = pretrain_sample['gt_box'][0]
        pretrain_list = pretrain_sample['results']
        finetune_list = finetune_data[dta_idx]['results']
        ids_list = ids_data[dta_idx]['results']
        ##
        if mask_data != []:
            new_mask = mask_data[dta_idx]['results'] 
            temp_pretrain_list = []
            temp_finetune_list = []
            temp_ids_list = []
            for pre_bbox, ft_bbox, id, mask in zip(pretrain_list, finetune_list, ids_list, new_mask):
                if mask == 1: 
                    temp_pretrain_list.append(pre_bbox)
                    temp_finetune_list.append(ft_bbox)
                    temp_ids_list.append(id)
            pretrain_list = temp_pretrain_list
            finetune_list = temp_finetune_list
            ids_list = temp_ids_list
        # get the mislabeled images by finetuned model
        # print('id: {}'.format(ids_list[0]))
        error_list = posprocess_preds(pretrain_list, finetune_list, ids_list)
        if len(error_list) > 0:
            sv_path = os.path.join(fig_path, f'id{ids_list[0]}')
            if not os.path.exists(sv_path):
                os.makedirs(sv_path, exist_ok=True)
            ## save groundtruth target along with gt box, predicted box + scores of pretrain and finetune
            pos_target_id = ids_list[0]
            pos_img = annotations[pos_target_id]['target_name']
            gt_box = annotations[pos_target_id]['target_box'] + [-1]
            pretrain_pred_box = [pretrain_list[0][0], pretrain_list[0][1], pretrain_list[0][2] - pretrain_list[0][0], pretrain_list[0][3] - pretrain_list[0][1], pretrain_list[0][-1]]
            pretrain_pos_score = pretrain_pred_box[-1]
            finetune_pos_score = finetune_list[0][-1]
            plot_box(boxes=[gt_box, pretrain_pred_box], texts=['', f'pt:{pretrain_pos_score:.2f}; ft:{finetune_pos_score:.2f}'], 
                     colors=['green', 'red'], src_path = os.path.join(target_root, pos_img), des_path = os.path.join(sv_path, 'positive.png'))
            ## save query image with corresponding query box and label
            qry_img = annotations[pos_target_id]['query_name']
            temp_box = annotations[pos_target_id]['query_box']
            qry_box = [temp_box[0], temp_box[1], temp_box[2] - temp_box[0], temp_box[3] - temp_box[1], -1]
            label = annotations[pos_target_id]['label']
            plot_box(boxes=[qry_box], texts=[label], colors=['green'], 
                     src_path = os.path.join(query_root, qry_img), des_path = os.path.join(sv_path, 'query.png'))
            ## save negative target with box + scores of pretrain and finetune
            for neg_target_id in error_list:
                neg_img = annotations[neg_target_id]['target_name']
                index_pred = ids_list.index(neg_target_id)
                neg_pred_box = [pretrain_list[index_pred][0], pretrain_list[index_pred][1], pretrain_list[index_pred][2] - pretrain_list[index_pred][0], pretrain_list[index_pred][3] - pretrain_list[index_pred][1], pretrain_list[index_pred][-1]]
                pretrain_neg_score = neg_pred_box[-1]
                finetune_neg_score = finetune_list[index_pred][-1]
                new_name = neg_img.split('.')[0]
                plot_box(boxes=[neg_pred_box], texts=[f'pt:{pretrain_neg_score:.2f}; ft:{finetune_neg_score:.2f}'], colors=['red'], 
                        src_path = os.path.join(target_root, neg_img), des_path = os.path.join(sv_path, f'{new_name}.png'))
            print('sample {}: {}'.format(dta_idx, error_list))
            print(dta_idx)
            if dta_idx > 30: break
        # temp_rec = calculate_recall(new_list, [given_box], K=10)
        # if dta_idx == 20:
        #     import pdb; pdb.set_trace()
    #     all_recall.append(calculate_recall(new_list, [given_box], K=10))
    # print("Summarize: R@10={}".format((sum(all_recall) / len(all_recall))))