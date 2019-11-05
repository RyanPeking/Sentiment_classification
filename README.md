# Sentiment_classification
## 本项目为以商家点评数据为基础的情感细粒度分类问题，对6大类，20小类进行打标，对于每个小类而言，都会有<正面情感, 中性情感, 负面情感, 情感倾向未提及 >  这4个类别。
## 访问网址http://39.100.3.165:1234/#/
## 处理方案
### 1.数据处理：
#### 数据清洗：去重、去空，繁体变简体，去除标点符号及停用词，但保留表情符号
#### 分词：分别以char和word级别分别测试
### 2.模型搭建(不同模型融合）：
#### TextCNN
#### biGRU、biLSTM
#### biGRU、biLSTM with Attention
#### RCNN
### 3.评价标准：
#### 样本存在严重的不平衡问题，负面情绪与中性情绪很少，如果以acc作为评价标准，虽然看起来准确度很高，但是却预测不准小样本数据，因此以f1为评价指标更为合理
#### 最终f1=0.6056,但并没跑完所有模型，RCNN整体稍好一些，由于时间原因，RCNN只跑了前几个类别
### 4.调参总结
#### 模型：TextCNN及RCNN总体表现更好一些
#### 分词：对于RCNN, char级别好于word级别，其他未测试
#### LSTM与GRU:LSTM略好于GRU
#### Hidden_size: 256>128
#### Class_weight: 设置过balanced方式，但发现效果不好，分析是因为某些样本数据过少，设置太大惩罚项之后会导致小样本的过拟合
#### 但对于TextCNN，Class_weight为0: 1, 1: 3, 2: 3, 3: 0.5更佳，但对于RCNN,不设置class_weight更佳
#### Batch_size:32>128,可能是由于小的batchsize随机性更强，带来的噪声有助于逃离sharp minimum，模型泛化能力更好。
#### Loss:虽然y为one_hot形式，但loss设置为binary_crossentropy,收敛更快，这是因为binary_crossentropy同时更新0类标签
### 5.效果展示
![image](https://github.com/RyanPeking/Sentiment_classification/blob/master/img/web.png)



