# Notes






## Convert a Tf-idf sparse matrix to a pandas dataframe

The The TF-IDF vectoriser results in a sparse output as a scipy CSR matrix. This CSR matrix cannot be directly added to a pandas dataframe object. 

Solutions are:

df['vector']=list(x)

Stores the data in a column named 'vector'. The newly added column is in sparse format. 

If you transform the sparse vector to a an array you can store a dense representation as follows: 

df['vector'] = list(x.toarray())


