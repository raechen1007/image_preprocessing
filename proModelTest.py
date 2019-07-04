# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:00:06 2019

@author: Administrator
"""

from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



model = load_model('./model_save/proModel.h5',compile=False)

img_width, img_height = 299, 299
batch_size = 50
n_classes = 4

validation_datagen = ImageDataGenerator(
        rescale=1./255,
        samplewise_std_normalization=True,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        fill_mode='constant', 
        cval=0)

validation_generator_1 = validation_datagen.flow_from_directory(
    directory='train_bak_pro',
    target_size=[img_width, img_height],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# 输出batch_size中的标签
test_sample_1, y_true_1 = next(validation_generator_1)
y_pred_1 = model.predict_generator(validation_generator_1, steps=1, max_queue_size=10,
                                   workers=1, use_multiprocessing=False, verbose=1)
print(y_pred_1)


validation_generator_2 = validation_datagen.flow_from_directory(
    directory='train_bak_pro',
    target_size=[img_width, img_height],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# 输出batch_size中的标签
test_sample_2, y_true_2 = next(validation_generator_2)
y_pred_2 = model.predict_generator(validation_generator_2, steps=1, max_queue_size=10,
                                   workers=1, use_multiprocessing=False, verbose=1)
# print(y_pred_2)
y_true = np.concatenate((y_true_1, y_true_2), axis=0)
y_pred = np.concatenate((y_pred_1, y_pred_2), axis=0)


# print(y_pred)


# roc
def plot_roc():
    #     test_sample, y_true = next(validation_generator)
    #     y_pred = model.predict_generator(validation_generator, steps=1, max_queue_size=10,
    #                                      workers=1, use_multiprocessing=False, verbose=1)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # print()
    #
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(n_classes):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    #
    # # Finally average it and compute AUC
    # mean_tpr /= n_classes
    #
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    lw = 2
    # plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)
    #
    # colors = cycle(['aqua', 'darkorange'])
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'coral'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('DR')
    plt.legend(loc="lower right")
    plt.show()


def matrix():
    #         test_sample, y_true = next(validation_generator)
    #         y_pred = model.predict_generator(validation_generator, steps=1, max_queue_size=10,
    #                                          workers=1, use_multiprocessing=False, verbose=1)

    y_true_label = np.argmax(y_true, axis=1)
    # print(y_true)
    y_pred_label = np.argmax(y_pred, axis=1)
    # print(y_pred)

    target_names = ['0', '1', '2', '3']
    # target_names = ['0', '1']
    print("***********classification_report*****************")
    print(classification_report(y_true_label, y_pred_label, target_names=target_names))

    # 四分类及二分类数据
    #
    cm = confusion_matrix(y_true_label, y_pred_label).T
    print("\n**********4class_confusion_matrix*************")
    print(cm)

    """三分类"""

    #         cm[0][1] = cm[0][1] + cm[0][2]
    #         cm[1][1] = cm[1][1] + cm[1][2]
    #         cm[2][1] = cm[2][1] + cm[2][2]
    #         cm[1][0] = cm[1][0] + cm[2][0]
    #         cm[1][1] = cm[1][1] + cm[2][1]

    """四分类"""

    a = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
    b = cm[0][2] + cm[0][3] + cm[1][2] + cm[1][3]
    c = cm[2][0] + cm[2][1] + cm[3][0] + cm[3][1]
    d = cm[2][2] + cm[2][3] + cm[3][2] + cm[3][3]

    """二分类"""

    # a = cm[0][0]
    # b = cm[0][1]
    # c = cm[1][0]
    # d = cm[1][1]

    print("\n**********2class_confusion_matrix*************")
    print(d, c)
    print(b, a)

    TP = d
    FP = c
    FN = b
    TN = a
    #
    precision = TP * 1.0 / (TP + FP)
    recall = TP * 1.0 / (TP + FN)
    specificity = TN * 1.0 / (TN + FP)
    f1_score = (2 * precision * recall) / (precision + recall)
    print("precision :", precision)
    print("Sensitivity/recall :", recall)
    print("specificity :", specificity)
    print("F1_score:", f1_score)


def test():
    score = model.evaluate_generator(validation_generator_1, steps=1, max_queue_size=10, workers=1,
                                     use_multiprocessing=False, verbose=1)  #亲测，steps=设置大小，不影响loss和accuary值
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


plot_roc()
matrix()
# test()