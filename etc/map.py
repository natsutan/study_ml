from sklearn.metrics import label_ranking_average_precision_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

y_true = [1, 1, 1, 0, 1, 1, 0, 0, 1, 0]
score = [1.0, 0.95, 0.90, 0.80, 0.70, 0.70, 0.65, 0.60, 0.60, 0.55]

precision, recall, threshold = precision_recall_curve(y_true, score)

ap = average_precision_score(y_true, score)
print("ap = %f" % ap)


plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(ap))

plt.show()
