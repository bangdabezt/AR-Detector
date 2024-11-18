# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import to_device
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator

from datasets.panoptic_eval import PanopticEvaluator
import json
import util.customize_metric as cmetric

def negative_train(model: torch.nn.Module, criterion: torch.nn.Module, dataset, target, args=None):
    device = torch.device(args.device)
    new_batch_size = 4
    # image id of positive sample
    img_id = target['image_id'].item()
    label = target['labels'].item()
    # get negative sample from dataset
    if args.eval_mode == 'all_negatives':
        neg_images, pos_target, pos_query = dataset.get_all_negatives(img_id, args.seed, new_batch_size-1)
    elif args.eval_mode == 'hard_negatives':
        neg_images, pos_target, pos_query = dataset.get_hard_negatives(img_id, target['cap_list'][label], args.seed, new_batch_size-1)
    else:
        raise ValueError("'eval_mode' hasn't supported value {}".format(args.eval_mode))
    assert len(neg_images) < new_batch_size
    # combine negative and positive samples to create a batch
    samples = [pos_target] + neg_images
    ng_bs = len(samples)
    queries = [pos_query for _ in range(ng_bs)]
    exemplars = [target["exemplars"].to(device) for _ in range(ng_bs)]
    labels_uncropped = [target["labels_uncropped"].to(device) for _ in range(ng_bs)]
    captions = [target["caption"] for _ in range(ng_bs)]
    # NestedTensor
    samples = utils.nested_tensor_from_tensor_list(samples).to(device)
    queries = utils.nested_tensor_from_tensor_list(queries).to(device)
    # train model with new batch
    with torch.cuda.amp.autocast(enabled=args.amp):
        outputs = model(samples, queries, exemplars, labels_uncropped, captions=captions)
    
    return criterion.finetune_loss(model, outputs)

def finetune_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    weight_dict = {
        'const': 1.0,
        'reg': 0.05
    }
    for samples, queries, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        # import pdb; pdb.set_trace()
        assert len(targets) == 1 # currently only learning with 1 positive sample
        # negative_train here
        loss_dict = negative_train(model, criterion, data_loader.dataset, targets[0], args)
        ## loss for backward
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    # if getattr(criterion, 'loss_weight_decay', False):
    #     resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0


    for samples, queries, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        samples = samples.to(device)
        queries = queries.to(device)
        # other info
        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        exemplars = [t["exemplars"].to(device) for t in targets]
        labels_uncropped = [t["labels_uncropped"].to(device) for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, queries, exemplars, labels_uncropped, captions=captions)
            loss_dict = criterion(outputs, targets, cap_list, captions)

            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat

@torch.no_grad()
def test_negatives(model, postprocessors, dataset, pos_results, targets, exemplars, captions, args=None):
    model.eval()
    device = torch.device(args.device)
    neg_bs = 4
    def postprocess_res(result, limit_num_pred = 1):
        """
        Input:
            - result: dict of torch.tensor {'scores', 'labels', 'boxes'}
            - limit_num_pred: number of prediction to retain in result
        Output: List of boxes combined with scores
        """
        all_boxes = result['boxes'][:limit_num_pred].tolist()
        all_scores = result['scores'][:limit_num_pred].tolist()
        combined_result = [(all_boxes[idx] + [all_scores[idx]]) for idx in range(limit_num_pred)]
        return combined_result

    all_results = []
    all_gt_boxes = []
    for pos_index, target in enumerate(targets):
        ## first of all, get the result of positive sample
        sample_result = []
        sample_result.extend(postprocess_res(pos_results[pos_index]))
        ## process for negative samples
        img_id = target['image_id'].item()
        if args.eval_mode == 'all_negatives':
            neg_images, neg_anns, pos_query, gt_bboxes = dataset.get_all_negatives(img_id, args.seed)
        elif args.eval_mode == 'hard_negatives':
            neg_images, neg_anns, pos_query, gt_bboxes = dataset.get_hard_negatives(img_id, target['labels'].item())
        else:
            raise ValueError("'eval_mode' hasn't supported value {}".format(args.eval_mode))

        # neg_queries = utils.nested_tensor_from_tensor_list([pos_query for _ in range(neg_bs)]).to(device)
        # neg_exemplars = [exemplars[pos_index] for _ in range(neg_bs)]
        for neg_index in range(0, len(neg_anns), neg_bs):
            print(neg_index)
            # if neg_index > 30: break 
            # import pdb; pdb.set_trace()
            # test with hard negatives only
            temp_neg_bs = len(neg_images[neg_index: neg_index + neg_bs])
            neg_queries = utils.nested_tensor_from_tensor_list([pos_query for _ in range(temp_neg_bs)]).to(device)
            neg_exemplars = [exemplars[pos_index] for _ in range(temp_neg_bs)]
            neg_samples = utils.nested_tensor_from_tensor_list(neg_images[neg_index: neg_index + neg_bs]).to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(neg_samples, neg_queries, neg_exemplars, [torch.tensor([0]).to(device) for _ in range(temp_neg_bs)], captions=[captions[pos_index] for _ in range(temp_neg_bs)])
        
            orig_target_sizes = torch.stack([t["orig_size"].to(device) for t in neg_anns[neg_index: neg_index + temp_neg_bs]], dim=0)

            results = postprocessors['bbox'](outputs, orig_target_sizes)
            for index in range(temp_neg_bs):
                sample_result.extend(postprocess_res(results[index]))
        
        ## Update all results
        all_results.append(sample_result)
        all_gt_boxes.append(gt_bboxes)
    return all_results, all_gt_boxes


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    # 
    coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)


    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only

    if args.use_coco_eval:
        from pycocotools.coco import COCO
        coco = COCO(args.coco_val_path)

        #
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item['name'] for item in category_dict]
    else:
        cat_list=args.label_list
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)
    customized_results = []
    all_gt_bboxes = []
    for samples, queries, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        queries = queries.to(device)
        
        exemplars = [t["exemplars"].to(device) for t in targets]
        # labels_uncropped = [t["labels_uncropped"].to(device) for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]

        bs = samples.tensors.shape[0]
        # input_captions = [caption] * bs ### this caption help to increase the AP and AR
        input_captions = [cat_list[target['labels'][0]] + " ." for target in targets]
        print("input_captions: " + str(input_captions))
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            ## add query samples
            outputs = model(samples, queries, exemplars, [torch.tensor([0]).to(device) for t in targets], captions=input_captions)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors['bbox'](outputs, orig_target_sizes)
        #### Process for negative samples -----------------------------------------------------------------------------------------------------------
        # import pdb; pdb.set_trace()
        all_results, all_gt_boxes = test_negatives(model, postprocessors, data_loader.dataset, results, 
                                                    targets, exemplars, input_captions, args)
        for sid in range(bs):
            save_res = {
                "results": all_results[sid],
                "gt_box": all_gt_boxes[sid]
            }
            with open(os.path.join(output_dir, "res_{}.txt".format(args.eval_mode)), "a") as f:  # Use open() directly
                f.write(json.dumps(save_res) + "\n")
        # customized_results.extend(all_results)
        # all_gt_bboxes.extend(all_gt_boxes)
        ## ----------------------------------------------------------------------------------------------------------------------------------------
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        ## test res
        # temp_results = []
        # for out in results:
        #     my_out = {key: value[:10] for key, value in out.items()}
        #     temp_results.append(my_out)
        # results = temp_results
        ##
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # import pdb; pdb.set_trace()
        ## test res
        for indx, key_id in enumerate(res.keys()):
            # res[key_id]['labels'][0] = targets[indx]['labels'][0].item()
            for lb_id in range(len(res[key_id]['labels'])):
                res[key_id]['labels'][lb_id] = targets[indx]['labels'][0].item()
        ## continue
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:



            for i, (tgt, res) in enumerate(zip(targets, results)):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = res['boxes']
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
       

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return stats, coco_evaluator