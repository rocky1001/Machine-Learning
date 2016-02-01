# coding=utf-8
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import jieba

__author__ = 'rockychi1001@gmail.com'


def jieba_tokenizer(x): return jieba.cut(x)
# 加载训练文本（网易新闻抓取的分类数据），并切分为训练集
training_data = load_files(ur'D:\work_code\workspace\PythonML\crawler\netease', encoding='utf-8')
x_train, _, y_train, _ = train_test_split(training_data.data, training_data.target, test_size=0.00000001)
# 生成词语的Tfidf向量空间模型，注意，训练样本数据调用的是fit_transform接口
words_tfidf_vec = TfidfVectorizer(binary=False, tokenizer=jieba_tokenizer)
X_train = words_tfidf_vec.fit_transform(x_train)
# 训练分类器
clf = LinearSVC().fit(X_train, y_train)
# 加载待预测文本数据，并切分为测试集
testing_data = load_files(ur'D:\work_code\workspace\PythonML\text_classify\netease_test\predict_data', encoding='utf-8')
_, x_test, _, _ = train_test_split(testing_data.data, testing_data.target, test_size=0.99999999)
# 测试样本数据调用的是transform接口
X_test = words_tfidf_vec.transform(x_test)
# 进行预测
pred = clf.predict(X_test)

for label in pred:
    print u'predict label: %s ' % training_data.target_names[pred]
