# General Concepts 
*metric* - how predictions are judged \
*loss* - quantity training is trying to minimize \
*optimizer* - rule that updates model’s weights using gradients (use AdamW for NN in ICAIC)\
*scheduler* - changes the optimizer's learning rate during training 

## metric 
choose the metric based off what the model counts as a mistake \
if task gives a metric, use it for comparing models and selecting checkpoints 

#### numerical
**MAE**: avg size of an error, in original units \
**RMSE**: gives large mistakes more weight, also in original units \
**MSE**: avg squared error \

Suppose three predictions miss by 0, 0, and 6. MAE = 2, avg absolute miss. MSE = 12 u^2, because you square the errors before avging. RMSE = √12 ≈ 3.46. Squaring makes the one large miss matter more.

#### classification 
**Accuracy**: fraction correct \
**Precision**: `TP / (TP + FP)`; minimize false positives (avoiding false alarms) \
**Recall**: `TP / (TP + FN)`; minimize false negatives (avoiding missed detections) \
**F1**: `2 × precision × recall / (precision + recall)`; combines precision and recall, drops when either is low. \
**Macro F1**: gives each class equal weight

precision and recall have different denominators\
precision, recall, F1: Say you flag 8 devices, 6 flags are correct, and 10 devices actually failed. \
Precision = 6/8 = 75% of your flags were right. \
Recall = 6/10 = 60% of the real failures were caught. \
F1 ≈ 67% combines those two rates. It favors models that do reasonably well at both.\
Macro F1: compute F1 separately for each class and avg them. If two classes have F1 scores of 90% and 50%, macro F1 is 70%, even if the first class is much more common.\

#### probabilities 
**Log loss**: penalizes confident wrong answers heavily \
**Calibration**: asks whether cases predicted at 80% actually happen about 80% of the time 

#### ranking results / find rare positives 
**REVIEW IN MORE DETAIL IF THERE'S TIME!!**
**Average precision**: summarizes precision and recall across thresholds \
**NDCG**: rewards putting relevant results near the top of a ranked list

#### predicting mask / predicting a count
**REVIEW IN MORE DETAIL IF THERE'S TIME!!**
**IoU/Dice**: measure mask overlap 
If your predicted mask has 8 pixels, the true mask has 10, and 6 overlap, IoU = 6/12 = 0.50 and Dice = 2×6/(8+10) ≈ 0.67. Both measure where you predicted correctly.\
Count MAE ignores location and averages the absolute difference between predicted and true counts. Predicting 8 birds when there are 10 gives a count error of 2.

## loss 
match loss to model ouput\

target | baseline loss (PyTorch)
| --- | --- |
One of several classes | CrossEntropyLoss with raw logits and integer class labels
Binary labels | BCEWithLogitsLoss with raw logits and float 0/1 targets
Number | L1Loss for absolute error. MSELoss to emphasize larger errors, or HuberLoss for a compromise

