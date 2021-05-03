# Nathan Nowack
**email** : nnowack@umich.edu 

**LinkedIn** : [Profile Page](https://www.linkedin.com/in/nathan-nowack-a6b59b143/)

---

portfolio directory: 

- [Professional and Academic Background)](#Professional-and-Academic-Background)

- [Project Summaries](#Project-Summaries)

- [other whimsical projects](#Somewhat-whimsical-projects)


---

## Introduction


<img align="right" height="24" src="pictures/umichCOE.png"/>

#### University of Michigan - Ann Arbor :  *College of Engineering*


-  August 2020 alumnus, cum laude 

- **Major:** Chemical Engineering
<img align="right" style="margin-right: 20px" height="200" src="pictures/me.png"/>

- **Minor**: Physics

Currently, I am an entry-level engineer seeking roles similar to:
- Process Engineer
- Data Scientist / Python Developer
- Market Forecasting Analyst
- Machine Learning Intern
- Quality Engineer

---

##  Professional and Academic Background

As a chemical engineering (ChE) undergrad, I've learned to use formalized math, physics, and chemistry frameworks to understand and process matter. To develop analytic models of phenomena *(e.g. systems of ODEs representing CSTR reactor evolution in Reactor Design class)*, ChE students often applied time-series regression to do things like specify coefficients *(e.g. N-order reaction rate laws from concetration over time)*. We were directed to use *PolyMath* (a literal ODE-system solver) or *Python* to accomplish this. Although intended as a means of informing process-engineering-oriented design considerations, I found the process of creating the solver in Python really exciting. Having this one small project and limited programming knowledge, I decided to focus my attention on skills in the union of engineering and data science. Consequently, since sophomore year, I’ve been practicing telling stories about datasets, mostly with Python’s statistical, databasing, visualization modules (*numpy*, *pandas*, *matplotlib*, *keras*, etc). Since then, I've taken courses in data structures, computational physics, and discrete math to build a conceptual background. I've learned to apply OOP concepts and software good practices by implementing ODE solvers, financial sensitivity models, custom python modules to generate visuals for homework, and  [personal projects](##-Personal-Projects).

During the summer of 2018, my role at Air Products (AP) as a Quality Engineer was to create predictive equipment health models. Using an understanding of the thermodynamics at play in system sub-domains and SQL queries to gather model input from AP manufacturing plants' cloud databases, we developed time-series regression models to associate and predict plant data.  These models effectively predicted equipment behaviour and retroactively isolated root causes of equipment failures. Throughout the role, I created and monitored equipment models for several dozen Air Products’ HYCO plants, proactively recommending maintenance to plant managers with clear, concise, and portable dashboards. In exploring the proprietary data processing framework we were using, I became familiar with SQL-oriented cloud-data pipelines common to the development of regression and classification models in general (e.g. BigQuery, Apache, and running on AWS).

While I didn’t remain in the industrial gas industry, I continue to find intrigue in the insightful storytelling of structured and unstructured data. My recent role in the scale-up of a polystyrene recycling plant, my senior design project, was to **model the likelihood of project capital outcomes**. I used Python to collect and analyze market data on relevant process variables’ capital costs, so I could determine NPV sensitivities and generate [visuals](###-Polystyrene-recycling-scale-up). I presented my findings to my team and administrative chair, **recommending a specific project capital allocation to ensure the highest probability of maximum profitability**. 

Recently, I’ve utilized COVID-mandated free-time by using Tensorflow's *Keras* API as a tool to quickly perform linear and DNN classification / regression. As documented [below](###-Using-Tensorflow's-Keras-to-predict-song-popularity), I believe that the analysis used in the song popularity model with multiple-factor regression could be applied to, for instance, a N-factor regression model predicting when consumers are most likely to interact with company advertisements or classification model identifying social media users receptive to targeted advertisements.


---

## Project Summaries

Here I've summarized some relevant projects that functioned as learning experiences. I've tried to select material that demonstrates a template of skills in data manipulation and visualization that could apply to a range of responsibilites in industry-specific analytics.

--- 
### Using Tensorflow's ***Keras*** to predict song popularity
<p style="margin-top: -15px; font-size: 10px"> October 2020 </p>

#### Background
I recently audited IBM / Coursera's [Introduction to Deep Learning & Neural Networks with Keras](https://www.coursera.org/learn/introduction-to-deep-learning-with-keras?action=enroll&authMode=signup) course, which made me want learn more about modeling with *Keras*. 

Using a structured [Spotify song dataset](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks) I found on Kaggle, I created a neural-network regression model to predict song **popularity** given features like the Spotify-generated metrics (e.g. *energy*, *danceability*, *tempo*) and song features like the musical *key* of the song. 

Below is the structure of the dataset as I found it. It includes 169,909 songs where each has both *numerical* and *categorical* factors, such as *energy* and *key* respectively. Here, I truncated the dataset to include 1000 songs (since I'm on my laptop's GPU):

<div align="center"> <img style="height:400; border: 1px" src="pictures/spotify_db.png"> </img> </div>

#### Cleaning and Preprocessing the dataset
I removed the following features from consideration:  *artists*, *id*, *name*, *year*, and *release date*. This is because they were either irrelevant to popularity (e.g. *id*) or introduced a bias that doesn't (in my opinion) have anything to do with the nature of the songs themselves (e.g. artist, year). 

Here, the *key* feature is categorical (0 corresponding to the key of C, 1 to D flat, and so on...). Therefore, in order to regress my model, I needed to *one-hot encode* the *key* feature, turning the single categorical column into 12 binary columns (one for each key).

```python
# one-hot encode the categorical 'key' feature
keys = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
key_map = {}

for i in range(0, len(keys)): key_map[i] = keys[i]
    
df['key'] = df['key'].map(key_map)
df = pd.get_dummies(df, prefix='', prefix_sep='')
```

Next, I used a `sklearn` method called `train_test_split` to get training and testing sets from my `pandas` dataset:

```python
# split df into training and testing sets
train, test = train_test_split(df, test_size=0.2)
```

Since features like *loudness* and *duration_ms* had significantly different value ranges than the Spotify-generated song metrics (which are in [0,1]), I decided to normalize these value ranges. First, I used `pandas.DataFrame.describe()` to generate a statistical summary of my training set, called `train_stats`. The first 5 features in `train_stats` are shown below:

<div align="center"> <img style="height:400; border: 1px" src="pictures/train_stats.png"> </img> </div>

After using `df.pop('popularity')` to split off training and testing labels into their own lists, I normalized my training and testing sets using a method that leverages `train_stats`:

```python
# define z-score normalization method using train_stats
def normalize(x):
    return (x - train_stats['mean']) / train_stats['std']

# normalize training and testing sets
normalized_train = normalize(train)
normalized_test = normalize(test)
```
#### Model Training and Evaluation
After preprocessing, I was ready to build the *Keras* model:

```python
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train.keys())]),
        keras.layers.Dense(1)
    ])
    
    model.compile(
        loss='mae', 
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=['mae', 'mse'])
   
    return model

model = build_model()
```
I chose a `Sequential` model with 2 defined layers, the *Rectified Linear Unit* (relu) activation function and the `Adam` optimizer.

With the model created, I trained the model for 100 epochs:

```python
# specify number of epochs and train model with validation
EPOCHS = 100

history = model.fit(
    normalized_train, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()]
)
```
**NOTE**: the callback function `PrintDot()` is an indicator of training progress.

... and plotted a *"learning curve"* to show the evolution of *mean absolute error* over the course of the training epochs. Looking at the learning curve below, my validation error is above my training error and trending up slightly. This suggests that I have overfit my model, which could be a result of:
- too many training epochs
- improper use of the `Adam` optimizer (which I don't fully understand yet)

<div align="center"> <img style="height:400; border: 1px" src="pictures/spotModelOver.png"> </img> </div>

Lastly, I evaluated the model on my testing dataset as follows:

```python
results = model.evaluate(normalized_test, test_labels, batch_size=32)
```

... which gave my model results for this training session:

<center>

| **loss** | **mean absolute error** |
| :---: | :--: |
| 0.3561 | 1.1009 |

</center>

#### Conclusion

Since Spotify's popularity metric is a number from 0-100, a mean absolute error of ~1 is not bad for my (relatively crude) model. If I wanted to improve this model's accuracy / robustness, some design modifications I could explore are:
- the topology of the neural network (either more layers or changing layer width)
- including more than 1000 song examples in the model
- a different optimizer 
- conducting an F-score analysis on my features and choosing the top 10

---

### Bag of Words Classifier (based on EECS 280 project)
<p style="margin-top: -15px; font-size: 10px"> April 2020 </p>

#### Background
For this project, we were tasked with creating a forum post classifier with C++  using a Bag of Words model. I have recreated the functionality of this program in Python for practice (and to avoid violating the Honor Code by posting my actual assignment).

Some csv files with Piazza posts (student-instructor Q&A forum) were provided, containing post labels and content:

| tag | content |
|:---:|:---:|
| exam | in w15 problem 4d when we create base baseptr der does this make an instance of base with value 5|
| image | if youve got your resize program working feel free to share the results with some images of your own|
| recursion | i may have missed this but would we expect to have negative numbers in our liststrees |
|...|...|

During this project we were learning about Binary Search Trees, recursion and maps, so we were required to implement our own search tree and map classes before creating the post classifier. These implementations were pretty slow, so when it came time to write the high-level classifier we used the more efficient standard C++ map/BST structures. In my recreation, I use Python's analogous `dict` class. 

To train our classifier to recognize posts, we used a simplified form of Bayes' conditional probability theory:    <img src="https://render.githubusercontent.com/render/math?math=P(C|X) = \frac{P(C) * P(X|C)}{P(X)}">

... the interpretation in this case being: the probability a given post *X* has label *C* is a function of: the number of posts with label *C* and some factor representing the likelihood that *X* has *C* given the history of associations between the words (& word pairs) in a given post and the post labels.

#### Solution
I created a class `Classifier()` for the high-level classifier with members to store a history of posts, labels, unique words, unique word-label pairs and with member functions to read in and record instances of words with labels in the training data, predict the labels of new posts, and to summarize the results. I used Python's `set()` for the word bank to avoid adding duplicate items.

I created a class `Post()` representing a single forum post with members for the unique constituent words and the label along with a `Post.has(word)` member function returning `True` when a post has `word`. 

Next, I created a function to calculate the *log-probability score* of all labels for a given based on the classifier map of word-label instances, using Bayes' theorem in the following manner:

**log-probability score** =  <img src="https://render.githubusercontent.com/render/math?math=ln P(C)"> + <img src="https://render.githubusercontent.com/render/math?math=\sum_i^N ln P (w_i | C )"> 

The probability score is the prior value from the number of posts with the label it's calculating plus the contribution of each word in the post. To avoid null values in situations where a word has never been seen before or seen before with a specific label, I conditionally calculate the contribution of each word as shown below:

```python
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
For a single new post unknown to the classifier, the `log_score()` routine runs for each label known to the classifier and selects the label giving the largest score, using the `Classifier.predict()` member function defined as shown below:
```python
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

I remembering being excited about this because even for a relatively simple Bag of Words model, it correctly matched 2563 of 2959 new posts based on 11,283 training posts, i.e. 87% accuracy. This project's source is linked [here](https://github.com/zzstoatzz/portfolio/blob/master/classify.py). 

In the future I want to alter the structure of this project to attack a similar problem (sentiment analysis on Twitter and other platforms using Tweepy / relevant API) while using more of a neural network / stochastic gradient descent correction method to predict more abstract ideas than just post labels. 


---

### Polystyrene recycling scale-up
<p style="margin-top: -15px; font-size: 10px"> January-March 2020 </p>


---

### Physics Presentation: Percolation
<p style="margin-top: -15px; font-size: 10px"> April 2020 </p>

#### Background
For the physics lab that accompanied Physics 390: Modern Physics (that I took for to complete my physics minor), we had to give a presentation on some active area of particle physics. Since I spent summer 2019 researching percolating networks with my fluids professor, I figured I would give a summary of my work as well as overview of the topic in general. Here is a snapshot of what I learned and created throughout my research and the creation of the presentation.

My professor has spent a great deal of time developing the theory of percolation, or long-range connectivity in networks. This is applied to wide range of studies and research applications, like: [modeling adsorbing and reacting surfaces](https://en.wikipedia.org/wiki/Random_sequential_adsorption), [modeling epidemics](https://en.wikipedia.org/wiki/Epidemic_models_on_lattices), the susceptibility of server networks to localized attacks, and is even used in the development of [more robust natural language processing tehcnologies](https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f). The characteristic value of a network in percolation theory is the critical threshold probability <img src="https://render.githubusercontent.com/render/math?math=p_c">. Which, depending on the regime (site/bond/site-bond), represents the probability of occupying a site or bond that leads to unchecked cluster growth. To put this in (unfortunate) context, what would be the critical percolation threshold of network representing the spread of Coronavirus between people on Earth? Making some assumptions, something like the likelihood of adjacent individuals spreading the virus that corresponds to the greatest instantaneous growth of the sick collective.

While I spent a lot of time reading about these different sub-disciplines, I focused most of my time working on networks that are of a transitional dimension (i.e. a really large 2-d network with a very small height).  Intuitively, if a randomly evolving network has more dimensions to grow in, long-range connectivity will emerge more quickly for the same 'infection rate' (which is analogous to saying that a virus will spread more quickly if more people are around). We were interested in describing the relationship between percolation threshold values and marginal dimensionality, so I studied <img src="https://render.githubusercontent.com/render/math?math=p_c"> values in increasingly thick network 'slabs'. Here I will show the strict 2-d case for purposes of illustration.

#### Solution
To setup the confines of our network (in the bond percolation regime that I am showing), one can create a "lattice" or system of nodes (also called sites) of a specific size. Programmatically, this can be modeled by a simple 1-d array equal in length to the total number of nodes, as shown by the typical array below:

<p align="center"><img height="70" src="percolation/array.png"  /></p>

One could imagine "chopping" this array into *L* rows, chopping every *L* nodes to create an abstract lattice from a simple array. Below is the chopped array, resulting in an abstracted 4x4x1 square lattice (where *L* = 4):

<p align="center"><img width="300" src="percolation/start.png"  /></p>

Next we have to model the ways in which nodes can interact with each other. In most 2-d models, nodes are limited to interacting with their N, S, E, W "nearest neighbors" and so we can define a list of all the possible bonds that can exist in terms of the vertices these bonds would connect at (each bond you add has to connect at two vertices, so I've represented the list of bonds as a list of vertices necessitated by possible bonds). The following code allows for 3 dimensional systems by including bonds between nodes directly above/below each other in adjacent 2-d layers, but here I'm just talking about a 2-d lattice so the `for z in range(0, WIDTH)` loops won't add any extra bonds to our list of possible vertices in this case.

```python
# init vertex lists, return iterator (3/2*N) as 'index'
def init_lists(v1, v2):
    index = 0
    for x in range(0, WIDTH):
        for y in range(0, WIDTH):
            for z in range(0, HEIGHT):
                s1 = x + WIDTH*y + WIDTH*WIDTH*z
                s2 = ((x+1)&W) + WIDTH*y + WIDTH*WIDTH*z
                index = connect(v1, v2, s1, s2, index)
                s2 = x + WIDTH*((y+1)&W)+ WIDTH*WIDTH*z
                index = connect(v1, v2, s1, s2, index)
                if (z < HEIGHT-1):
                    s2 = x + WIDTH*y + WIDTH*WIDTH*((z+1)&W)
                    index = connect(v1, v2, s1, s2, index)
                else:
                    s2 = x + WIDTH*y + WIDTH*WIDTH*(0)
                    index = connect(v1, v2, s1, s2, index)
    return index
```
I am using the `WIDTH` of the system as a pre-defined 'jump' within the array, where +/-`WIDTH*y` corresponds to a jump of *y* rows N or S, while +/- 1 is a movement left or right. In this way, we define all the possible vertices of bonds, connect them:

```python 
def connect(v1, v2, A, B, i):
    v1[i] = A
    v2[i] = B        
    return i + 1
```
and store them in lists that I've called `v1` and `v2`. Now that we have defined all the possible ways in which local connections can appear, it's time to start growing the network!

The evolution of these networks involves a couple interesting pieces of code. First is the `findroot()` routine, which is analogous to a contact tracing algorithm that finds the root node of a cluster, given any node:

```python
# recursive path compression 'find' algorithm
def findroot(i, ptr):
    if ptr[i] < 0: 
        return i
    ptr[i] = findroot(ptr[i], ptr)
    return ptr[i]
```
This is situation is sort of like employees (nodes) acting as agents of companies (clusters) in a quest to merge with the largest company, but only the CEO of each company knows how many employees they have. So if two employees of different companies randomly meet, they would ask their respective superiors how many people are in the company and if that superior doesn't know, the superior "points" up the chain of command until they get to the CEO. This is the portion of the program referred to as 'path-compression' or 'contact-tracing' in epidemiology or 'find' in 'union-find' algorithms. 

Once you finally get to the CEO (root node), the company (cluster) with a smaller workforce (# nodes) would decide to merge with the larger company. This is essentially the 'union' in 'union-find' algorithms, and my implementation (based on [the work of my professor Robert Ziff and Mark Newman](https://arxiv.org/abs/cond-mat/0101295)) is shown below:
```python
# union-find routine: bonds connect sites
def cluster(index, lists, big, M2):
    v1, v2, ptr, smax, M2tot, M2minus = lists
    for i in range(0, index):
        r1, r2 = findroot(v1[i], ptr), findroot(v2[i], ptr)
        if (r2 != r1):
            M2 += ptr[r1]*2.0*ptr[r2]
            if ptr[r1] > ptr[r2]:
                ptr[r2] += ptr[r1]
                ptr[r1] = r2
                r1 = r2
            else:
                ptr[r1] += ptr[r2]
                ptr[r2] = r1
            if (-ptr[r1]>big):
                big = -ptr[r1]
        smax[i] += big
        M2tot[i] += M2
        M2minus[i] += (M2 - big*1.0*big)
    return ptr
```
Using `igraph` (which I believe sits on top of `matplotlib.pyplot`) and `ffmpeg` I created and stitched together images of the clusters after each new added bond to show the evolution of the network for widths of 4, 16, 32, and 64 nodes as shown:

<p align="center"><img src="percolation/1.gif" /></p>

<p align="center"><img src="percolation/2.gif" /></p>

<p align="center"><img src="percolation/3.gif" /></p>

<p align="center"><img src="percolation/4.gif" /></p>

To obtain <img src="https://render.githubusercontent.com/render/math?math=p_c"> from what we've done already, all we need to do is find the inflection point of maximum cluster size against evolution time. However, since this is a stochastic process, we want to find the average inflection point over a large number of network evolutions to determine more precisely where this threshold is. Running the simuation 100 times for `L = 128` and averaging our values, we obtain the logistic curve below (similar to something you would see for number of virus hosts in an extraordinarily reckless society where no one dies or recovers):

<p align="center"><img src="percolation/smax.png" /></p>

Using a differential calculus Python package `findiff` one can find the inflection point to appear consistently at <img src="https://render.githubusercontent.com/render/math?math=p_c = 0.497">, very much in agreement with the accepted (analytically determined) [threshold for bond percolation on 2-d sqaure lattices](https://en.wikipedia.org/wiki/Percolation_threshold#Percolation_on_2D_lattices) <img src="https://render.githubusercontent.com/render/math?math=p_c = 1/2"> A common theme in percolation theory is to investigate infinite networks and consequently the behaviour of the system near the emergance of an infinite cluster (at the critical probability threshold). You could imagine that the logistic curve above gets squished in toward the middle as the system gets very large and eventually becomes a step function with a step at the exact point the largest cluster goes from finite to infinite. This is the point of interest, long-range connectivity, the point at which seemingly disparate areas of a network become members of the same cluster. Empirically describing an infinite network, as well as one with fractional dimension, helps us find universalities among networks. Finding common features allows for characterization of growth and interaction in networks of more general size and dimensionality (for say, an *M* member epidemic with *N* factors determining transmission probability during member interactions).

Computationally, it is impossible to create an infinite array of memory space, but the properties of infinite networks can be reasonably estimated using extrapolating methods developed over years of identifying universal properties of lattices. For example, if one needed to know critical probability threshold of a infinite 2-d lattice, one could find <img src="https://render.githubusercontent.com/render/math?math=p_c"> for a series of values of L and use the universal constant for 2-d systems <img src="https://render.githubusercontent.com/render/math?math=\nu = 4/3 "> to extrapolate and estimate <img src="https://render.githubusercontent.com/render/math?math=p_c"> of an infinite 2-d lattice as shown below:

<p align="center"><img src="percolation/extrapolation.png" /></p>

Finding the intercept (corresponding to the limit of <img src="https://render.githubusercontent.com/render/math?math=L^{-1/ \nu}"> as *L* goes to infinity) is therefore akin to finding the percolation threshold of an infinite square 2-d lattice!

My value in this case, only running the simulation 100 times, was <img src="https://render.githubusercontent.com/render/math?math=p_c = 0.5007"> . This is a small but significant improvement over critical threshold of the single `L = 128` lattice, an improvement that becomes more significant in larger and more complex networks.

---
## Somewhat whimsical projects

### Mood Board 
<p style="margin-top: -15px; font-size: 10px"> June 2020 - present </p>

I love music, it is one of the central pillars of my life among my family, friends, and academia/work. As a means of learning UI/UX, to make animations, and real-time sound analysis... I started a personal project that will incorporate all of these things into an "online mood ring". That is, some song will represent a user's 'color' state (frequency state) at any given time, to be regressively derived their song history (via Spotify API).
#


#### Basic Fourier Analysis
##### June 1, 2020

To begin, I wanted to review the classic Fourier transform before I started thinking about how to try and apply some of those methods here. Below is a visual of a [psuedo-Fourier transform animation I created](board/media) (based on [this video](https://www.youtube.com/watch?v=spUNpyF58BY)) to get started:

<p align="center"><img src="board/media/dash.gif" /></p>

The sinusoid at the top is some superimposed signal (hypothetically unknown, here it's a combined signal of the first two C octaves) that contains some number of distinct frequencies. The radial figure at the bottom left is just the original signal wrapped around a circle at a variable angular frequency (f_wrap, indicated by the green slider). You'll notice that when the green marker reaches the same value as a blue marker (signal frequency), two things happen: the radial signal "lines up" in a lopsided way and we get a peak in the bottom right figure. This corresponds to a "match" between f_wrap and some frequency that is present in the signal. In fact, every peak on the bottom right represents a frequency present in the original signal.

My example might seem sort of redundant.. you might say  "*greeeaaat*, you got peaks at 5 and 10 Hz and so you can say that two signals with frequencies 5 and 10 Hz are in the signal, but you already knew that in the first place!". You'd be right, but cool thing is that I would be able to find peaks even if I didn't know anything about the signal. As long as I varied the green slider (f_wrap) enough, when I 'matched' a signal frequency (even if invisible on the blue slider), I would get a peak at that frequency. I used this example as a proof of concept. So, like finding the exact notes in an unknown chord, I can find all frequencies present in an arbitrary singal using the setup below: 

```python
##### Frequency identification
from IPython.display import HTML
from FREQhelpers import createDash
import sys, time

# animation / dynamics inputs
resolution = .15
duration, T = 30, 2 # seconds, seconds/cycle
freqs, signal = [5, 10], False # Hz, varySignalFreq? else varyWrapFreq
t0 = time.time()

# display animation via HTML embedded mp4
args = [duration, resolution, T, freqs, signal]
anim = createDash(args)
HTML(anim.to_html5_video())

```
... where I used a custom method `createDash` to setup figures, set data, and return the `matplotlib` animation object, shown below:

```python
def createDash(args, plts='all'):
    # derive anim characteristics from inputs
    duration, resolution, T, freqs, signal = args
    steps = int(maxSteps(SizeLim)*resolution)
    t_est = t_required(steps)
    buff = np.std(freqs)/2
    f_0, f_f = freqs[0]-buff, freqs[-1]+buff  # Hz, set for plot scaling
    t = np.linspace(0, T, steps)
    f = np.linspace(f_0, f_f, steps)
    interval = duration*1e3/steps

    # setup mpl fig
    if plts=='all':
        objs = setupS(f_0, f_f, T, signal, freqs)
        fig, ax = objs[0], objs[1:]
        args = (t, f, ax, signal, freqs, t_est)

    elif plts=='wrap':
        objs = setupW(f_0, f_f, T, freqs)
        fig, ax = objs[0], objs[1]
        c = plt.cm.jet(np.linspace(0, T, steps))
        args = (t, f, ax, freqs, t_est, c)
        t_est = t_est/5

    # pack args and call animator
    anim = animation.FuncAnimation( fig, 
                                    spectrum, 
                                    frames=steps, 
                                    fargs=args, 
                                    interval=interval, 
                                    blit=True)
    return anim
```
... where I used custom methods `t_required`, `maxSteps`, `setupS`, and `setupW` to create a progress bar and set up the figures.

You might notice that I could have made a functional frequency-analyzer really quickly if I had just used `PyAudio` with some Discrete Fourier Transform routine, and in my application, that's what I will do. I just wanted to understand better where the idea came from, and get a little animation experience in the process.

# 
#### A visual reference for playing guitar along with *the mood*
##### June 9, 2020

I have taken a rather non-linear path on this project, partly because of the sporradic nature of my interests but also because I had this idea and wanted to get it into code. Instead of getting right into the Spotify API and figuring out how to extract user info or working on the predictive 'color' model, I created an interface to be used after a song is predicted, so that one can play guitar along with their 'mood' (key). Better displayed on my [GitHub page](https://zzstoatzz.github.io/), here is what I'm talking about:

<p align="center"><img src="board/media/demo.gif" /></p>

... where instead of having the option to select the key, eventually the diagram will show the notes in the key of the predicted song. Another idea is to display only the pentatonic notes of the key as a framework and to give an option for users to add specific intervals to the diagram to get an idea of what 'color' that interval adds to the mix.

As for how these diagrams were created, I again used `matplotlib` libraries and defined Python classes as summarized below: 

<center>

| Class | Purpose | Members | Methods |
|:---:|:---:| :---: | :---:|
| `Board` | Loads note, tuning, and instrument data for renderings of the fretboard| `len` &nbsp; `keys` &nbsp; `baseNotes` &nbsp; `frets` &nbsp; `noteBank` &nbsp; `notes` &nbsp; `fig`|`getBN()` &nbsp; `show()` &nbsp;`save()`|
| `Fret` | Loads appropriate notes for each string from an instance of `Board`|`id` &nbsp;`notes` |`__str__()`|
| `Key` | Loads appropriate notes given the name of a key| `name` &nbsp; `notes` &nbsp; `intervals` &nbsp; `root`| --|
| `Note` | Houses note-identifiers within the context of a `Board` instance| `stringName` &nbsp; `FREQ` &nbsp; `name` &nbsp; `genName`| `__str__()`|
| `Tuning` | Loads appropriate "[open notes](https://nationalguitaracademy.com/guitar-string-notes/)" given the name of a tuning, loads stored tunings | `freqs` &nbsp; `tunings` &nbsp; `name`|`getNotes()` |

</center>

Defined in this way, it is straight-forward to render PNGs corresponding to each `Key` using an instance of `Board`:

```python
from objects import Board, Key
B = Board()

for key in B.keys[:12]:
    majorK, minorK = Key(key+' major'), Key(key+' minor')
    ext = '.png'
    B.show(majorK)
    B.save(str(majorK)+ext)
    
    B.show(minorK)
    B.save(str(minorK)+ext)
```

... giving diagrams for all 12 major and 12 minor keys. One might say that creating diagrams for a major key AND it's [relative minor](https://www.musictheoryacademy.com/understanding-music/relative-major-and-relative-minor-scales/) key is redundant, but I believe it is still useful to consider the distinct tonal center (i.e. root) of a key even if the notes are the same (as in C major and A minor). As I work out the frequency analysis side of things, I will make the distinction between such keys, since the tonal center is the reference point that make such diagrams useful in the first place.

At some point down the road when I have a functioning script that transcribes a given song into a series of sub-sections of a key (i.e. chords), I would like to implement a transform animation so users can follow along cleanly with the progression of a song.
