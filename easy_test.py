from util import customize_metric as csm 

# ./attribution_finetune/test_all/res_all_negatives.txt
# ./attribution_finetune/test_all_best/res_all_negatives.txt
# ./attribution_finetune/test_negative/res_hard_negatives.txt
# ./attribution_finetune/test_negative_best/res_hard_negatives.txt

# hack1 = csm.summarize_metric('./attribution_finetune/new_const_loss/new_test/res_all_negatives.txt')
# # csm.summarize_metric('./attribution_finetune/test_all_best/res_all_negatives.txt')
# hack2 = csm.summarize_metric('./attribution_finetune/new_const_loss/new_test/res_hard_negatives.txt')
# # hack2 = csm.summarize_metric('./attribution_finetune/new_const_loss/res_all_negatives.txt')
# # csm.summarize_metric('./attribution_finetune/test_negative_best/res_hard_negatives.txt')
# sum_it = []
# for neg_it, all_it in zip(hack2, hack1):
#     if all_it == 0:
#         continue
#     sum_it.append(neg_it/all_it)
# print(sum(sum_it)/len(sum_it))

# csm.summarize_metric('./attribution_finetune/new_all_loss/new_test/res_all_negatives.txt')
# csm.summarize_metric('./attribution_finetune/new_all_loss/best_test/res_all_negatives.txt')

# csm.summarize_AP('./attribution_finetune/pretrained/res_all_negatives.txt') abc
# csm.summarize_AP('./attribution_finetune/pretrained/res_all_negatives.txt', num_pred=10)

# csm.summarize_AP('./attribution_finetune/new_all_loss/test_100/res_all_negatives.txt')
# csm.summarize_AP('./attribution_finetune/new_all_loss/test_100/res_all_negatives.txt', num_pred=10)

# csm.summarize_AP('./attribution_finetune/new_const_loss/test_100/res_all_negatives.txt')
# csm.summarize_AP('./attribution_finetune/new_const_loss/test_100/res_all_negatives.txt', num_pred=10)

# csm.summarize_AP('./attribution_finetune/new_const_loss/test_100/res_hard_negatives.txt')
# csm.summarize_AP('./attribution_finetune/new_const_loss/test_100/res_hard_negatives.txt', num_pred=10)

# csm.summarize_AP('./attribution_finetune/new_all_loss/test_100/res_hard_negatives.txt')
# csm.summarize_AP('./attribution_finetune/new_all_loss/test_100/res_hard_negatives.txt', num_pred=10)
# csm.summarize_metric('./attribution_finetune/new_const_loss/res_all_negatives.txt')

# csm.summarize_AP('./attribution_finetune/pretrained/res_all_negatives.txt')
# csm.summarize_AP('./attribution_finetune/pretrained/res_all_negatives.txt', num_pred=1)
# csm.summarize_AP('./attribution_finetune/new_all_loss/test_100/res_all_negatives.txt')
# csm.summarize_AP('./attribution_finetune/new_all_loss/test_100/res_all_negatives.txt', num_pred=1)
# csm.summarize_AP('./attribution_finetune/new_const_loss/test_100/res_all_negatives.txt')
# csm.summarize_AP('./attribution_finetune/new_const_loss/test_100/res_all_negatives.txt', num_pred=1)


## test with old res_all_negatives.txt
# csm.summarize_APS('./attribution_finetune/new_all_loss/res_all_negatives.txt')
# csm.summarize_APS('./attribution_finetune/new_all_loss/new_test/res_all_negatives.txt')

# csm.summarize_APS('./attribution_finetune/new_const_loss/res_all_negatives.txt')
# csm.summarize_APS('./attribution_finetune/new_const_loss/new_test/res_all_negatives.txt')

## plot predictions on image? plot 2-10 boxes on each positive (negative) sample??
## remove some first FP prediction and observe the value of AP -> ok
## remove all FP with score > TP and observe the value of AP # 0.02275
## in each pos-sample test, remove all FP of neg-sample that have score > TP # 0.0133644

### scaled recall
# """
# -all_loss: 0.05583
# -all_loss remove first FPs: 0.05721
# -all_loss remove all FP with score > TP: 0.82318 # 58890 / 83902
# -all_loss remove all FP of neg-sample that have score > TP: 0.2289 # 76110
# """ => problem: score of FP is too high => small values are accumulated from early stage 
# => value of precision is low in the early period.
# each postive sample has its own threshold of detection score 
# (some positive samples are detected with very low score, 0.2-0.4 while 
# there are some other positive and negative samples of the sample tuple having very high score => these negative downgrade the overall AP)
# => scale this one with softmax to obtain attribution score and use this score instead?

## scale the score and test again
## with original softmax
# """
# -all_loss: 0.07479
# -all_loss remove first FPs: 
# -all_loss remove all FP with score > TP: 0.79916
# -all_loss remove all FP of neg-sample that have score > TP: 0.24223
# unfair normalization
# target image is duplicated over sets (many pos samples share top-1 negative sample)
# ==> biased by input text => biased by some negative samples
# ==> decrease weight of text when computing score

## plot predictions on image? ok

## ignore predictions of 'dog', test on other categories
## new metric that avoid negative samples of different tuple affect score of positive sample
## observe prediction of positive that stays in top 20, how high the score of negatives in these cases

## how are other detectors' predictions on negative samples evaluated? 
# Other detectors usually mark high score for TPs, 
# high score => always TP, and low score for both TPs and FPs
## detectron metrics for evaluation => detectron to re-label the query

## -----------------------------------------------------------------------------------------------------------------
### convert training jsonl file to COCO format for testing
### change the config
### check engine and customize_metric to make sure that it can test with training data
### test with training
### change the metric to original ranking metric
### find a ranking loss and an approach to modify the architecture for better learning to advance positive sample (more than negative samples)


# csm.summarize_diff_AP('./attribution_finetune/new_pretrained/newest_allCat/res_all_negatives.txt')
# csm.summarize_diff_AP('./attribution_finetune/new_all_loss/009_allCat/res_all_negatives.txt')
# csm.summarize_diff_AP('./attribution_finetune/new_const_loss/009_allCat/res_all_negatives.txt')

# csm.summarize_diff_AP('./attribution_finetune/new_const_loss/new_AP_diffCat/newest/res_all_negatives.txt')
# csm.summarize_diff_AP('./attribution_finetune/new_const_loss/new_AP_diffCat/best/res_all_negatives.txt')
# csm.summarize_diff_AP('./attribution_finetune/new_const_loss/new_AP_diffCat/test009/res_all_negatives.txt')

## test with train set # AttributionGD/attribution_finetune/summary_train
# AttributionGD/attribution_finetune/pretrained/best_trainSet/res_all_negatives.txt
# AttributionGD/attribution_finetune/new_pretrained/best_trainSet/res_all_negatives.txt # 3509
# csm.summarize_trainSet('./attribution_finetune/new_pretrained/best_trainSet/res_all_negatives.txt', './attribution_finetune/summary_train/mask_diffCat.txt')
# csm.figureOut_trainError('./attribution_finetune/summary_train/pretrain.txt', './attribution_finetune/summary_train/all_loss.txt',
#                          './attribution_finetune/summary_train/all_neg_ids.txt', './data/break-a-scene/coco2017/official_data/anno_train_filename.json', './attribution_finetune/summary_train/mask_diffCat.txt')
## continue running test on training set of pretrained model
## visualize the error with normal negatives (><hard negatives)

## visualize top-1 predictions ? -> why it is better than the others?
## implement DR Loss # AttributionGD/attribution_finetune/sum_test_ranking/best_test_0/ranking_finetune/res_all_negatives.txt
# csm.summarize_trainSet('./attribution_finetune/sum_test_ranking/adapter_rank/const_loss_0004/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_finetune/sum_test_ranking/adapter_rank/const_loss_0009/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_finetune/sum_test_ranking/adapter_rank/const_loss_0014/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_finetune/sum_test_ranking/adapter_rank/const_loss_0019/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_finetune/sum_test_ranking/adapter_rank/const_loss_best/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_finetune/sum_test_ranking/adapter_rank/const_loss_newest/res_all_negatives.txt')
# AttributionGD/attribution_finetune/summary_train/best_test/test_const_loss/res_all_negatives.txt
# csm.summarize_trainSet('./attribution_finetune/summary_train/best_test/test_const_loss/res_all_negatives.txt')
# AttributionGD/attribution_img_swinT/test_trained_model_pos/original_cf/best_test/res_all_negatives.txt
# csm.summarize_trainSet('./attribution_img_swinT/test_trained_model_pos/original_cf/newest_test/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_img_swinT/test_trained_model_pos/original_cf/best_test/res_all_negatives.txt')
# # csm.summarize_trainSet('./attribution_img_swinT/test_trained_model_pos/same_cf/newest_test/res_all_negatives.txt')
# # csm.summarize_trainSet('./attribution_img_swinT/test_trained_model_pos/same_cf/best_test/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_img_swinT/test_finetuned_model/all_loss/test_0029/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_img_swinT/test_finetuned_model/const_loss/test_0029/res_all_negatives.txt')
# csm.summarize_trainSet('./aattribution_img_BERTlarge/test_finetuned_all/best_20/res_all_negatives.txt')
# csm.summarize_trainSet('./aattribution_img_BERTlarge/test_finetuned_const/best_20/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_img_BERTlarge/test_trained_model_pos/best_test/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_img_BERTlarge/conf1/test_0023/res_all_negatives.txt')
# AttributionGD/attr_lvis_img/test_res/config1/best_test/res_all_negatives.txt
# csm.summarize_trainSet('./attribution_class/config1/best_test/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_class/config2/best_test/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_class/config11/best_test/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_class/config21/best_test/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_class/best_test/res_all_negatives.txt')

# AttributionGD/attribution_class/countgd_pretrain1/res_all_negatives.txt
# csm.summarize_trainSet('./attr_lvis_img/countgd_pretrain/res_all_negatives.txt')
# csm.summarize_trainSet('./attribution_class/countgd_pretrain1/res_all_negatives.txt')
# attribution_finetune/batch2/result
csm.summarize_trainSet('./attribution_finetune/batch6/result/res_all_negatives.txt')