# idea: try to learn a model that understands whether a word is a keyword
# then outputs keywords in some random order as the summary
# implementation may be terrible, I'm not a great software engineer
# feel free to optimize stuff and fix errors
import collections
import numpy as np

def keywordFeatureExtractor(words):
    # assumes input words is a list of words
    # returns output featureVec which is a dictionary of dictionaries
    # keys of featureVec are words in the 
    # for each word, we generate 2 features + 1 bias unit
    # 3 features are all indicator features
    # whether the word appears more than 1/5/15 times

    countDict = collections.defaultdict(int)
    threshs = [1, 5, 15]
    featureVec = {}
    for word in words:
        countDict[word] += 1

    def featureExtractor(word, countDict, threshs):
        featureVec = {}
        for thresh in threshs:
            if countDict[word] >= thresh:
                featureVec[thresh] = 1
        # for thresh in threshs:
            # if countDict[word] >= thresh:
                # featureVec[(word,thresh)] = 1
        return featureVec

    for word in words:
        if word not in featureVec:
            featureVec[word] = featureExtractor(word, countDict, threshs)        
   
    return featureVec

def trainClassifier(dataset, keywordFeatureExtractor):

    # dataset is a list of tuples (story, highlight)
    # I assumed both story and highlight are strings
    # I assumed highlight string doesn't contain @highlight
    # the input to this should only be the training set
    # don't forget to split the data in train/dev/test before

    # some helper functions:
    def sigmoid(z): 
        return 1 / (1 + np.exp(-z))

    def generatePrediction(feature, w):
        z = 0
        for key,value in feature.items():
            z += value * w[key]
        return sigmoid(z)
    def computeLoss(prediction, label):
        return prediction - label

    def updateWeights(predLoss, w, feature, learnRate, regParam):
        for key,value in feature.items():
            w[key] -= learnRate * (predLoss * value + regParam * w[key])
        return w

    w = collections.defaultdict(float) # initialize weights to 0
    T = 50 # num epochs
    learnRate = 0.01 # self explanatory
    regParam = 0.01 * 2 # self explanatory, times 2 to avoid doing that in update
    for t in range(T):
        for (story, highlight) in dataset:
            words = story.split()
            keywords = highlight.split()
            labels = collections.defaultdict(int) 
            features = keywordFeatureExtractor(words) # generate features
            for keyword in keywords: # read ground truth labels, 0 means not a keyword
                labels[keyword] = 1

            for word in words: 
                prediction = generatePrediction(features[word], w) # generate prediction for each word
                predLoss = computeLoss(prediction, labels[word]) # compute loss
                w = updateWeights(predLoss, w, features[word], learnRate, regParam) # update weights
        print(t) # I was on Python 3
    return w

def generateSummary(story, w, keywordFeatureExtractor):
    words = story.split()
    summary = set()
    features = keywordFeatureExtractor(words)

    def sigmoid(z): 
        return 1 / (1 + np.exp(-z))
    def generatePrediction(feature, w):
        z = 0
        for key,value in feature.items():
            z += value * w[key]
        return sigmoid(z)
    for word in words:
        pred = generatePrediction(features[word], w)
        print(word,pred)
        if pred > 0.5:
            summary.add(word)
    return ' '.join(list(summary))

# try with fake data
dataset = []
dataset.append(('this is just a dummy dummy dummy dummy dummy story', 'dummy'))
dataset.append(('this is another another another another another dummy story', 'another dummy'))
dataset.append(('this is not not not not not a real story', 'not real'))

w = trainClassifier(dataset, keywordFeatureExtractor)
story = 'this is a very very very very very real real real real real story'
highlightStory = 'real story'

summary = generateSummary(story, w, keywordFeatureExtractor)
print(summary)
print(highlightStory)
print(w)

