# Shield

一个以文本类算法为基础、结合场景的风险防控系统

简介  

----

风险控制系统有很多应用场景，比如反垃圾邮件、新闻风控、广告反作弊等等，本文旨在从文本角度入手，介绍一些风控系统的常用方法

#### 1.文本匹配  

----

没错，这是最简单的方法了，我们需要通过词包管理系统管理违禁词，主要包括：政治人物词包、色情词包等。文本匹配就是直接对需要进行风控检测的文本进行词包匹配，如果命中词包的话说明这些文本就是有问题的，需要做相应的处理措施。这种方式的缺点很明显：  

1. 效率较低  

2. 无法及时应对文本的变异和变化  

提高文本匹配效率可以使用Trie的树结构，[这是一个代码示例](https://github.com/lijingpeng/python/blob/master/algo/Trie.py "这是一个代码示例")

#### 2. 贝叶斯分类

----

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
