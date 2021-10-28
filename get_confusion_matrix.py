import numpy as np
from pprint import pprint

tp = None
tn = None
fp = None
fn = None
history = np.load('model_history.npy', allow_pickle=True).item()

for key in history:
    if key == 'true_positives':
        tp = int(history[key][len(history[key])-1])
    if key == 'true_negatives':
        tn = int(history[key][len(history[key])-1])
    if key == 'false_positives':
        fp = int(history[key][len(history[key])-1])
    if key == 'false_negatives':
        fn = int(history[key][len(history[key])-1])

print('| ---- |  T  |  F  |\n|  P   |  {}|   {}|\n|  N   | {}|   {}|'.format(tp, fp, tn, fn))
#print('True Positives: {}, True Negatives: {}, False Positives: {}, False Negatives: {}'.format(tp, tn, fp, fn))
accuracy = (tp+tn) / (tp+tn+fp+fn)
misclass = (fp + fn) / (tp+tn+fp+fn)
precision = (tp) / (tp + fp)
recall = (tp) / (tp + fn)
f1 = (2*(precision*recall)) / (precision + recall)

print("Accuracy:          {}".format(round(accuracy, 2)))
# print("Misclassification: {}".format(round(misclass, 2)))
print("Precision:         {}".format(round(precision, 2)))
print("Recall:            {}".format(round(recall, 2)))
print("F1 Score:          {}".format(round(f1, 2)))