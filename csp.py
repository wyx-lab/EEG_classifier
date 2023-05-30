import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from mne.io import concatenate_raws
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score,cross_validate
from mne import Epochs, pick_types, events_from_annotations
from mne.decoding import CSP
import os
import mne
from sklearn import svm
from sklearn import tree
import xgboost as xgb

def judge(data):
    for da in data:
        if "E" in da:
            data.remove(da)

tmin, tmax = -1., 4.

root= "C:/Users/86181/Desktop/ugPRML_EEGProject/Data/data"
path = os.path.join(root, "A03T.gdf")
path2 = os.path.join(root, "A01T.gdf")
data = os.listdir(root)
judge(data)
raw = concatenate_raws([mne.io.read_raw_gdf(os.path.join(root,da)) for da in data])
#raw = mne.io.read_raw_gdf(path2)
# 获取想要读取的文件名称，这个应该是没有会默认下载的数据
# 将3个文件的数据进行拼接
raw.load_data()
# 去掉通道名称后面的（.），不知道为什么默认情况下raw.info['ch_names']中的通道名后面有的点
raw.rename_channels(lambda x: x.strip('.'))
raw.filter(4., 40., fir_design='firwin', skip_by_annotation='edge')  #
events, _ = events_from_annotations(raw)
raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
event_id = dict({'769': 7, '770': 8})
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
labels = epochs.events[:, -1] - 7
scores = {}
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
# 根据设计的交叉验证参数,分配相关的训练集和测试集数据
cv_split = cv.split(epochs_data_train)
lda = LinearDiscriminantAnalysis()
tree_model = tree.DecisionTreeClassifier(criterion='gini',
                                         max_depth=None,
                                         min_samples_leaf=1,
                                         ccp_alpha=0.0)
csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
# 创建机器学习的Pipeline,也就是分类模型，使用这种方式可以把特征提取和分类统一整合到了clf中
svm_model  = svm.SVC(kernel='linear', gamma=0.1, C=1.0)
clf = Pipeline([('CSP', csp), ('tree',svm_model)])
# 获取交叉验证模型的得分

scores= cross_validate(clf,epochs_data_train,labels,cv = cv,n_jobs=1,scoring=['f1','accuracy','precision','recall','roc_auc'])
#scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1,scoring=['precision','accuracy'])
# 输出结果，准确率和不同样本的占比
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores['test_accuracy']), class_balance))


print("Classification auc_score: %f" %(np.mean(scores['test_roc_auc'])))
print("Classification f1_score: %f" %(np.mean(scores['test_f1'])))
print("Classification Recall: %f,Classification precision: %f" %(np.mean(scores['test_recall']),np.mean(scores['test_precision'])))


scores = []
