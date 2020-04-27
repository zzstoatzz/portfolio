import pandas as pd
import numpy as np
import copy 

def add(q, d, v=1):
    if q in d:
        d[q] += v
    else:
        d[q] = 0
        d[q] += v
    return d

def ll(obj, newPost, label):
    num = len(obj.posts)
    bank, labels, pairs = obj.words, obj.labels, obj.pairs
    words = newPost.words
    sum = 0
    for word in words:
        pair = (label, word)
        if word not in bank:
            sum += np.log(1/num)
        elif pair not in pairs:
            sum += np.log(bank[word]/num)
        elif pair in pairs:
            sum += np.log(pairs[pair]/labels[label])
        else:
            print("yikes")
            assert(False)
    return sum

def train(obj, file):
    df = pd.read_csv(file).dropna()
    obj.posts = [Post(i, j) for i,j in df.iterrows()]
    for post in obj.posts:
        label = post.label
        obj.labels = add(label, obj.labels)
        for word in post.words:
            obj.words = add(word, obj.words)
            obj.pairs = add((label, word), obj.pairs)

class Post():
    def __init__(self, i, data):
        string = data["content"]
        self.words = set(string.split())
        self.content= string
        self.label = data["tag"]
    def summarize(self):
        print("words:", *self.words, "labels:", self.label)
    def has(self, word):
        return word in self.words

class Classifier():
    def __init__(self, data):
        self.posts = []
        self.labels = {}
        self.words = {}
        self.pairs = {}
        self.prediction = []
        train(self, data)

    def predict(self, file):
        df = pd.read_csv(file).dropna()
        newPosts = [Post(i, j) for i,j in df.iterrows()]
        prediction = []
        for newPost in newPosts:
            l = {}
            for label in self.labels:
                prior = np.log(self.labels[label]/len(self.posts))
                likelihood = ll(self, newPost, label)
                p = prior + likelihood
                add(label, l, v=p)
            key = max(l, key=l.get)
            pred = (key, newPost.label, key == newPost.label, round(l[key], 1))
            prediction.append(pred)
        self.summarize(prediction)
    
    def summarize(self, pred):
        x = 0
        for p in pred:
            if p[2] : x += 1
            #print('(Predicted, Actual, Correct, LP-score) =', p, '\n')
        print('Correct / Total = ', x, '/', len(pred), '=',  round(x/len(pred), 2))
    #def update(self)
