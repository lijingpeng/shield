# Shield

一个以文本类算法为基础、结合场景的风险防控系统。

## 简介  

风险控制系统有很多应用场景，比如反垃圾邮件、新闻风控、广告反作弊等等，本文旨在从文本角度入手，介绍一些风控系统的常用方法

## 文本预处理

### 将文本分割为段落
### 分词
[中文分词工具比较](https://ruby-china.org/topics/28000)
### 标准化
1. 去掉标点符号，可以使用正则表达式，全角转半角等  
2. 长度过小的词，比如单字  
3. 全部转换为小写字母

### 去掉停用词

### 简单的同义词替换

### 词性标注

分词的同时可以进行词性标注的工作，有些场景下可能只需要保留动词或者名词，形容词可能就没那么重要。

可以使用一些现有的同义词表进行替换，后续可以使用Word2vec来挖掘近义词。

## 1.文本匹配  


没错，这是最简单的方法了，我们需要通过词包管理系统管理违禁词，主要包括：政治人物词包、色情词包等。文本匹配就是直接对需要进行风控检测的文本进行词包匹配，如果命中词包的话说明这些文本就是有问题的，需要做相应的处理措施。这种方式的缺点很明显：  

1. 效率较低  

2. 无法及时应对文本的变异和变化  

提高文本匹配效率可以使用Trie的树结构，[这是一个代码示例](https://github.com/lijingpeng/python/blob/master/algo/Trie.py "这是一个代码示例")

## 2. 贝叶斯分类


关于贝叶斯算法的介绍请参见[这篇文章](http://blog.csdn.net/longxinchen_ml/article/details/50597149 "这篇文章")，这里我们利用NLTK自带的数据集来构建一个贝叶斯分类器:

载入NLTK包和movie_reviews数据，movie_reviews数据是一个情感分析的数据，每篇文章被标记为positive和negative。在风控领域，『积极的』和『消极的』可以替换为违规案例和正常案例。


```python
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

def word_exists(words):
    '''
    词袋模型，每一个字是特征名称带有一个True值
    '''
    return dict([(word, True) for word in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_exists(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_exists(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4

# 分割训练集和测试集
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

classifier = NaiveBayesClassifier.train(trainfeats)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()
```

    train on 1500 instances, test on 500 instances
    accuracy: 0.728
    Most Informative Features
                 magnificent = True              pos : neg    =     15.0 : 1.0
                 outstanding = True              pos : neg    =     13.6 : 1.0
                   insulting = True              neg : pos    =     13.0 : 1.0
                  vulnerable = True              pos : neg    =     12.3 : 1.0
                   ludicrous = True              neg : pos    =     11.8 : 1.0
                      avoids = True              pos : neg    =     11.7 : 1.0
                 uninvolving = True              neg : pos    =     11.7 : 1.0
                  astounding = True              pos : neg    =     10.3 : 1.0
                 fascination = True              pos : neg    =     10.3 : 1.0
                     idiotic = True              neg : pos    =      9.8 : 1.0


从结果中我们看到，准确率大概为72.8%，这个准确率不算高，当然贝叶斯算法也足够简单，为了提高准确率，一些其他的工作也需要做，比如去除停用词、改变训练集和测试集的比例并多次训练和测试。

NLTK的封装屏蔽了一些细节，下面简单介绍下思路：
我们的目标是求解在当前上下文环境下，一段文本属于『违规』文本的概率，即P(违规|词)，直接求解这个概率值是困难的。贝叶斯公式提供了计算此概率的一种方法：(朴素贝叶斯公式)
```
P(违规|词) = Pneg(违规) * P(词|违规)  
P(正常|词) = Ppos(正常) * P(词|正常)
```
对于预测样本中的每个词，分别计算P(词|违规)和P(词|正常)，即取出该词在正样本和负样本中出现的概率，与相应的先验概率相乘(为了防止乘以0出现，可以对概率取log，变为相加)。Pneg(违规)和Ppos(正常)是根据现有知识得到的先验概率，比如10篇邮件中有一封邮件是垃圾邮件，则Ppos(正常) = 1 - Pneg(违规) = 0.9, Pneg(违规) = 0.1。计算的得到P(违规|词)、P(正常|词)，看看更趋向于哪一类别

## 3. word2vec

Word2vec是一个基于神经网络的语言模型，不同于文本的TF-IDF编码和One-hot编码方式，Word2vec通过在语料库中训练，将词表示为向量的形式，然后通过计算向量之间的距离可以进行相似词的判断工作，或者文本的情感分类，使用举例请参考kaggle竞赛中的一段代码：[Bag_of_Words](https://github.com/lijingpeng/kaggle/tree/master/competitions/Bag_of_Words)

```python
model.most_similar("queen")
```




    [(u'princess', 0.6759523153305054),
     (u'bride', 0.6207793951034546),
     (u'belle', 0.6001157760620117),
     (u'shearer', 0.5995810031890869),
     (u'stepmother', 0.596365749835968),
     (u'victoria', 0.5917614698410034),
     (u'dame', 0.589063286781311),
     (u'latifah', 0.5790275931358337),
     (u'countess', 0.5776904821395874),
     (u'widow', 0.5727116465568542)]




```python
model.most_similar("awful")
```




    [(u'terrible', 0.7642339468002319),
     (u'atrocious', 0.7405279874801636),
     (u'horrible', 0.7376815676689148),
     (u'abysmal', 0.7010303139686584),
     (u'dreadful', 0.6942194104194641),
     (u'appalling', 0.6887971758842468),
     (u'lousy', 0.6646767854690552),
     (u'horrid', 0.6554058194160461),
     (u'horrendous', 0.6533403992652893),
     (u'amateurish', 0.6079087853431702)]

详细的代码可以参见上面的链接，在上面的示例中，获取与「awful」、「queen」最相近的词，结果符合直观的理解。在风险控制领域，变异词的挖掘是一个很重要的课题，风险词汇往往是千变万化的，为了规避检查，许多人会将管控词汇进行变异，比如将「古驰」变为「古奇」，一般的风控规则很难检测到这些变化，Word2vec从语义角度提供了检测这种变异词的一种手段。

## 4. Levenshtein距离
编辑距离，又称Levenshtein距离，[WIKI](https://zh.wikipedia.org/zh-cn/%E8%90%8A%E6%96%87%E6%96%AF%E5%9D%A6%E8%B7%9D%E9%9B%A2)，是指两个字串之间，由一个转成另一个所需的最少编辑操作次数。许可的编辑操作包括：将一个字符替换成另一个字符，插入一个字符，删除一个字符。俄罗斯科学家Vladimir Levenshtein在1965年提出这个概念。可以用来词相似度的比较。

```python
def normal_leven(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1

    matrix = [0 for n in range(len_str1 * len_str2)]

    for i in range(len_str1):
        matrix[i] = i
    for j in range(0, len(matrix), len_str1):
        if j % len_str1 == 0:
            matrix[j] = j // len_str1

    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                           matrix[j * len_str1 + (i - 1)] + 1,
                                           matrix[(j - 1) * len_str1 + (i - 1)] + cost)

    return matrix[-1]

if __name__ == '__main__':
    s1 = 'abcde'
    s2 = 'adcdef'
    print normal_leven(s1, s2)
```

输出：2，即abcde与adcdef的编辑距离是2

## 文章链接

- [支付风控场景分析](http://blog.lixf.cn/essay/2016/12/08/risk-1-scenarios/?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io)
- [支付风控数据仓库建设](http://blog.lixf.cn/essay/2016/12/18/risk-2-database/)
- [更多支付相关文档来自于这里](http://blog.lixf.cn/)
