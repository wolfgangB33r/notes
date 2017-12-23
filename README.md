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

## Normalizing

### Normalize an array between min and max
```python
from sklearn import preprocessing as pre
min_max_scaler = pre.MinMaxScaler()
normalized = min_max_scaler.fit_transform(matrix)
```
## References

http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

