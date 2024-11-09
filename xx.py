import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm

# 数据处理
# whole_data=pd.read_excel('/Users/ywo/Downloads/jqxx/ER_NewsInfo.xlsx')
# need_data = whole_data[(whole_data['Classify'] == '证券市场') & (whole_data['NewsSource'] == '证券日报')]
# columns_to_drop = ['NewsID', 'Symbol']
# need_data['DeclareDate'] = pd.to_datetime(need_data['DeclareDate'])
#
# # 提取日期部分并修改列格式
# need_data['DeclareDate'] = need_data['DeclareDate'].dt.strftime('%Y-%m-%d')
# need_data=need_data.drop(columns=columns_to_drop)
# need_data=need_data.reset_index(drop = True)
# print(need_data)
# need_data.to_excel('/Users/ywo/Downloads/jqxx/试用数据.xlsx', index=False)


use_data=pd.read_excel('/Users/ywo/Downloads/jqxx/试用数据.xlsx')
s=[]
m=[]
for i in use_data['DeclareDate']:   # 提取日期
    if i not in s:
        s.append(i)

# 每日新闻条数 柱状图
# for j in s:
#     target_date = use_data[use_data['DeclareDate'] == j]
#     num_target_date = len(target_date)
#     m.append(num_target_date)

# fig, ax = plt.subplots(figsize=(10, 7))
# ax.bar(s,m)
# plt.xlabel('Date')
# plt.ylabel('daily news')
# plt.xticks(rotation=45)
# plt.xticks([0, len(s) - 1], [s[0], s[-1]])
# plt.show()

X=[]
Y=[]
start_date='2023-06-01'
end_date='2023-11-24'  # end > '2023-06-06'
for i in use_data.loc[use_data['DeclareDate']<=end_date,'Title']:    # 总数据集
    X.append(i)
for j in use_data.loc[use_data['DeclareDate']<=end_date,'emotion']:  # 总数据集
    Y.append(j)
# 词云
# 将文本列表拼接成一个字符串
# text = ' '.join(X1)
#
# # 进行中文分词
# seg_list = jieba.cut(text)
# seg_text = ' '.join(seg_list)
#
# # 创建词云对象并设置参数
# wordcloud = WordCloud(font_path='/System/Library/Fonts/STHeiti Light.ttc', width=800, height=400, background_color='white')
#
# # 生成词云图像
# wordcloud.generate(seg_text)
#
# # 绘制词云
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

def chinese_tokenizer(text):      # 分词
    return jieba.lcut(text)

vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer)
X= vectorizer.fit_transform(X)
vocabulary = vectorizer.vocabulary_

X1=X[:129]
X2=X[129:]
Y=np.array(Y)
Y1=Y[:129]
Y2=Y[129:]

# 情感分类
def train_and_evaluate_model(X, Y):
    # 定义K-fold交叉验证的参数
    k = 5
    kf = KFold(n_splits=k, shuffle=True)

    # 初始化空列表，用于保存每次交叉验证的准确率
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    # 执行K-fold交叉验证
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        # 划分训练集和测试集
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # 初始化SVM随机梯度下降分类器
        classifier = SGDClassifier()

        # 训练模型
        classifier.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = classifier.predict(X_test)
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        # 将准确率添加到列表中
        accuracy_scores.append(accuracy)

        # 计算精确率、召回率和F1得分
        precision = precision_score(y_test, y_pred, average='weighted')
        precision_scores.append(precision)

        recall = recall_score(y_test, y_pred, average='weighted')
        recall_scores.append(recall)

        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
    # 输出每次交叉验证的准确率

        print("Fold {}: Accuracy={:.2f}%, Precision={:.2f}%, Recall={:.2f}%, F1 Score={:.2f}%".format(i+1, accuracy * 100, precision* 100, recall* 100, f1* 100))

    # 输出平均准确率
    print("平均准确率: {:.2f}%, 平均精确率: {:.2f}%, 平均召回率: {:.2f}%, 平均F1 Score: {:.2f}%".format(np.mean(accuracy_scores) * 100, np.mean(precision_scores)* 100, np.mean(recall_scores)* 100, np.mean(f1_scores)* 100))

    # print("平均准确率: {:.2f}%".format(np.mean(accuracy_scores) * 100))

    return classifier

model=train_and_evaluate_model(X1,Y1)

# 样本外预测
# y_pred_outside=model.predict(X[129:163])
# y_true=Y[129:163]
# print(y_pred_outside)
# print(y_true)
# accuracy2 = accuracy_score(y_true, y_pred_outside)
# precision2 =precision_score(y_true, y_pred_outside, average='weighted')
# Recall2 =recall_score(y_true, y_pred_outside, average='weighted')
# F1_Score2 =f1_score(y_true, y_pred_outside, average='weighted')
# print(accuracy2,precision2,Recall2,F1_Score2)

y_pred_outside=model.predict(X2)

# 实际与预测分类对比图
# date=use_data.head(129)
# date=date['DeclareDate']
# df = pd.DataFrame()
# 将date列添加到DataFrame
# df['Date'] = pd.to_datetime(date)
# # # 将Y1列添加到DataFrame
# df['True Values'] = Y1
# 将y_pred列添加到DataFrame
# y_pred=model.predict(X1)
# df['Predicted Values'] = y_pred
# df[['True Values','Predicted Values']].plot( figsize = (10,7))
# plt.title('True Values vs. Predicted Values')
# plt.ylabel('emotion')
# plt.gcf().autofmt_xdate()
# plt.legend()
# plt.show()

# 计算情绪指标
a=len(use_data[use_data['DeclareDate'] <= end_date])
for i in range(129,a):
    use_data.loc[i,['emotion']] =y_pred_outside[i-129]
pos = use_data[use_data['emotion'] == 1].groupby('DeclareDate')['emotion'].sum()
# print(pos)
neg = use_data[use_data['emotion'] == -1].groupby('DeclareDate')['emotion'].sum()
# print(neg)
score = use_data.groupby('DeclareDate').apply(lambda x: math.log((1 + x[x['emotion'] == 1]['emotion'].sum()) / (1 + abs(x[x['emotion'] == -1]['emotion'].sum()))))

# 情绪指标每日变化图
# plt.plot(score, marker='o')
# # 设置 X 轴标签和标题
# plt.xlabel('Date')
# plt.ylabel('Score')
# plt.title('Sentiment Score')
# # 自动调整日期标签格式
# plt.gcf().autofmt_xdate()
# plt.xticks([0, len(s) - 1], [s[0], s[-1]])
# # 显示图形
# plt.show()

# 预测股票
df2=pd.DataFrame(columns=['Dates','Score'])
date_range = pd.date_range(start=start_date, end='2023-11-24', freq='D')
df2['Dates']=pd.to_datetime(date_range)
score.index = pd.to_datetime(score.index)
for i in score.index:
    if i in df2['Dates'].values:
        df2.loc[df2['Dates'] == i, 'Score'] = score.loc[i]
df_filled = df2.fillna(method='ffill')
df2['Dates']=df2['Dates'].dt.strftime('%Y-%m-%d')

# df_filled.to_excel('/Users/ywo/Downloads/jqxx/综合日市场回报率文件/df_filled.xlsx', index=False)

# 预测市场回报率
stock=pd.read_excel('/Users/ywo/Downloads/jqxx/综合日市场回报率文件/TRD_Cndalym.xlsx')
stock=stock[stock['Markettype']==5]
stock=stock.reset_index(drop=True)
stock['Trddt']=pd.to_datetime(stock['Trddt'])
df3=pd.DataFrame(columns=['Dates','stock'])
date_range = pd.date_range(start='2023-06-01', end='2023-11-24', freq='D')
df3['Dates']=pd.to_datetime(date_range)
merged_df = pd.merge(df3, stock, left_on='Dates', right_on='Trddt', how='left')
k=merged_df['Cdretwdeq'].fillna(method='ffill')
merged_df['stock']=k.shift(-1)
stock_df=merged_df['stock'].fillna(method='ffill')
# # print(merged_df)
# print(df_filled['Score'])
# n=df_filled['Score'].head(30)
# stock_df=stock_df.head(30)

df_filled_array = np.array(df_filled['Score']).reshape(-1, 1)  # 转换为二维数组
#  df_filled_array = np.array(n).reshape(-1, 1)
stock_df_array = np.array(stock_df).reshape(-1, 1)  # 转换为二维数组
#
X = df_filled_array  # 自变量数据（假设为二维数组）
X = sm.add_constant(X)  # 添加常数列
Y=stock_df_array
model = sm.OLS(Y,X)
results = model.fit()
coefficients = results.params
t_values = results.tvalues
r_squared = results.rsquared
print("Coefficients:")
print(coefficients)
print("\nT-values:")
print(t_values)
print("\nR-squared:")
print(r_squared)
# 画图
# predicted_values = results.predict(X)
# time_axis =np.array(df3['Dates'])
# plt.plot(time_axis, Y, label='Actual Y')
# plt.plot(time_axis, predicted_values, label='Predicted Y')
# plt.xlabel('Time')
# plt.ylabel('Y')
# plt.title('Actual vs Predicted ')
# plt.legend()
# plt.show()


