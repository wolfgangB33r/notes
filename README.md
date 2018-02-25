# Notes

## Basics working with numpy

### Assign N columns 

```python
two_columns = text_f[:,[0,1]]
```

### Calculate the sum of one row
```python
sum = np.sum(axis=1)
```
## Basics working with pandas

### Drop rows with missing values

```python
df['price'].dropna()
```

### Drop rows where value does not meet criteria

```python
df = df[df['price'] > 0]
```

### Replace missing values with a default value

```python
df = df.fillna(value=0)
```
```python
df = df.replace([np.inf, -np.inf], np.nan)
```
### Calculate the correlation with a specific column
```python
df.corr()['price'])
```

### Write first N rows to a CSV file
```python
df.head(n=100).to_csv('test.csv', encoding='utf-8', index=False)
```

## Text analysis and machine learning

Classical bad-of-word approches such as count vectors and Tfidf vectors are outperformed by sentiment analysis such as Word2vec or Glove.

### Vectorize a pandas text column by using Tfidf vectors
```python
tv = TfidfVectorizer(max_features=NR_MAX_TEXT_FEATURES, stop_words='english')
tfidf = tv.fit_transform(data['item_description'])
print(tfidf.shape)
```

### Convert a Tf-idf sparse matrix to a pandas dataframe

The The TF-IDF vectoriser results in a sparse output as a scipy CSR matrix. This CSR matrix cannot be directly added to a pandas dataframe object. 

Solutions are:
```python
df['vector'] = list(x)
```
Stores the data in a column named 'vector'. The newly added column is in sparse format. 

If you transform the sparse vector to a an array you can store a dense representation as follows: 
```python
df['vector'] = list(x.toarray())
```
Or to directly create a pandas Dataframe use:
```python
df = pd.DataFrame(list(text_features.toarray()))
```

### Sentiment analysis

- Google's Word2Vec
- Glove 
- 

https://www.kaggle.com/c/word2vec-nlp-tutorial

- Load a pretrained Google Word2Vec model:
```python
import gensim
# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
```

### Clean textual input

```python
def text_to_words(raw_text, remove_stopwords=False):
    # 1. Remove non-letters, but including numbers
    letters_only = re.sub("[^0-9a-zA-Z]", " ", raw_text)
	  # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english")) # In Python, searching a set is much faster than searching
        meaningful_words = [w for w in words if not w in stops] # Remove stop words
        words = meaningful_words
    return words 
```

### Train word2vec model

```python
from gensim.models import word2vec
# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                   sentences_train = [['hello', 'this', 'is', 'a', 'sentence'], ['Second', 'one'],['third', 'sentence']] 

downsampling = 1e-3   # Downsample setting for frequent words
# Initialize and train the model (this will take some time)
model = word2vec.Word2Vec(sentences_train, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)
model.init_sims(replace=True) # end training to speed up use of model
# print most similar words
print(model.most_similar('woman'))
```

### Load a GloVe model
In case you have a pretrained GloVe word file available for your textual corpus, you can use this little script to load the words along their vector representation.

```python
def loadGloveModel(gloveFile):
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done. %d words loaded!" % len(model))
    return model
 ```

## Normalizing

### Normalize an array between min and max with sklearn

```python
from sklearn import preprocessing as pre
min_max_scaler = pre.MinMaxScaler()
normalized = min_max_scaler.fit_transform(matrix)
```

### Normalize a column of a pandas dataframe between 0 and 1

```python
df['col1'] = (df['col1']-df['col1'].min())/(df['col1'].max()-df['col1'].min())
```

## Feature selection

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])
# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)
# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()
```

## Feature selection on train and test data

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
# Perform feature selection
selector = SelectKBest(f_classif, k=5) # select 5 best correlating features
selector.fit(train_x, y)
train_selected_x = selector.transform(train_x)
test_selected_x = selector.transform(test_x)
# and go on training the model
``` 

## Some useful snippets 

### Hash a string into a normalized float between 0 and 1

It may seem strange to use random function below to get a normalized hash number between the boundaries of 0 and 1 but it works like a charm. As the seed is set through the standard has function you always get the corresponding mapped hash number between 0 and 1. 

```python
import random
random.seed(hash(your_string))
random.random()
```

## Load Glove model from file

```python

def makeFeatureVec(words, model, num_features):
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in model: 
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
    # Divide the result by the number of words to get the average
    if nwords == 0:
        nwords = 1
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # Initialize a counter
    counter = 0
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       #if counter%1000 == 0:
        #   print("Review %d of %d" % (counter, len(reviews)))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs


def loadGloveModel(gloveFile):
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done. %d words loaded!" % len(model))
    return model
    
    
    
# load glove model
wm = loadGloveModel('../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt')
num_features = 50
f_matrix_train_word = getAvgFeatureVecs(sentences_train, wm, num_features)
f_matrix_test_word = getAvgFeatureVecs(sentences_test, wm, num_features)
    
# load glove model from twitter
wm_tw = loadGloveModel('../input/glove-twitter/glove.twitter.27B.25d.txt')
num_features = 25
f_matrix_train_word_twitter = getAvgFeatureVecs(sentences_train, wm_tw, num_features)
f_matrix_test_word_twitter = getAvgFeatureVecs(sentences_test, wm_tw, num_features)
    
```


## Glossary

- LSTM: Long short-term memory units are a building block for layers of a recurrent neural network (RNN). An RNN composed of LSTM units is often called an LSTM network

## References

- [Scikit](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) Basic Python machine learning libraries
- [Keras](https://keras.io/) High level Python Deep Learning library
- [Word2vec](https://code.google.com/archive/p/word2vec/) Google word vector implementation
- [GloVe](https://nlp.stanford.edu/projects/glove/) Standford Global Vectors for Word Representation

