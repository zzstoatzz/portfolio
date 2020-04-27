# Nathan Nowack: Projects
### Bag of Words Classifier (based on EECS 280 project)
#### Background
For this project, we were tasked with creating a forum post classifier with C++ by using a Bag of Words model. I have recreated the functionality of this program in Python for practice (and to avoid violating the Honor Code by posting my actual assignment).

Some csv files with Piazza posts (student-instructor Q&A forum) were provided, containing post labels and content:

| tag | content |
|:---:|:---:|
| exam | in w15 problem 4d when we create base baseptr der does this make an instance of base with value 5|
| image | if youve got your resize program working feel free to share the results with some images of your own|
| recursion | i may have missed this but would we expect to have negative numbers in our liststrees |
|...|...|

During this project we were learning about Binary Search Trees, recursion and maps, so we were required to implement our own search tree and map classes before creating the post classifier. These implementations were pretty slow, so when it came time to write the high-level classifier we used the more efficient standard C++ map/BST structures. In my recreation, I used Python's analogous `dict` class. 

To train our classifier to recognize posts, we used a simplified form of Bayes' conditional probability theory:    <img src="https://render.githubusercontent.com/render/math?math=P(C|X) = \frac{P(C) * P(X|C)}{P(X)}">

... which is to say the probability a post X has label C is a function of the number of posts that have occurred with label C times some factor representing the likelihood that X has C given the history of associations between the words in the post and post labels.

#### Solution
I created a class for the high-level classifier `Classifier()` with members to store a history of posts, labels, unique words, unique word-label pairs and with member functions to read in and record instances of words with labels in the training data, predict the labels of new posts, and to summarize the results. I used Python's `set()` to avoid adding duplicate items.

I created a class `Post()` representing a single forum post with members for the unique constituent words and the label along with a `Post.has(word)` member function returning a bool if a post has a word. 

Next, I created a function to calculate the log-probability score of all labels for a given based on the classifier map of word-label instances, using Bayes' theorem in the following manner:

log-probability score = <img src="https://render.githubusercontent.com/render/math?math=ln P(C)"> + <img src="https://render.githubusercontent.com/render/math?math=\sum_i^N ln P (w_i | C )"> 

The probability score is the prior value from the number of posts with the label you're calculating plus the contribution of each word in the post. To avoid null values in situations where a word has never been seen before or seen before with a specific label, I conditionally calculate the contribution of each word as shown below:

```
def log_score(obj, newPost, label):
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
            print("should never happen")
            assert(False)
    return sum
```
For a single new post unknown to the classifier, the `log_score()` routine runs for each label known to the classifier and select the label giving the largest score, using the classifier predict member function as shown below:
```
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
```
This process is repeated throughout the whole csv of new posts and each time I check to see if the classifier predicted correctly (since the post content was actually labeled). 

I remembering being excited about this because even for a relatively simple Bag of Words model, it correctly matched 2563 of 2959 new posts based on 11,283 training posts, 87% accuracy. This project's source is linked [here](https://github.com/zzstoatzz/portfolio/blob/master/classify.py). 

A future exploit of mine may be to alter the structure of this project to attack a similar problem (sentiment analysis using Tweepy) while using more of a neural network / stochastic gradient descent correction method to predict more abstract ideas than a post label. 
