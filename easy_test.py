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

csm.summarize_metric('./attribution_finetune/new_all_loss/new_test/res_all_negatives.txt')
csm.summarize_metric('./attribution_finetune/new_all_loss/best_test/res_all_negatives.txt')
# csm.summarize_metric('./attribution_finetune/new_const_loss/res_all_negatives.txt')