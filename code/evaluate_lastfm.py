
# coding: utf-8

# In this module I wrap the ALS estimation and scoring with the scikit-learn base estimator to make it compatible with other scikit-learn modules such as cross-validation and grid-search. Have a look at [jobch's excellent blog post and implementation](https://towardsdatascience.com/recommending-github-repositories-with-google-bigquery-and-the-implicit-library-e6cce666c77), that's where I stole most of the code :).

# In[ ]:


import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from implicit.utils import nonzeros
import os
# disable internal multithreading
get_ipython().run_line_magic('env', 'MKL_NUM_THREADS=1')


# In[ ]:


os.chdir("C:\\Users\\simon.weiss\\Documents\\Freaky-Friday\\recommender\\recommender-lastfm\\code")


# In[ ]:


data = pd.read_table("../data/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv", 
                         usecols=[0, 2, 3], 
                         names=["user", "artist", "plays"],
                         na_filter = False,
                         encoding = "utf-8")


# In[ ]:


data['user'] = data['user'].astype("category")
data['artist'] = data['artist'].astype("category")

artist_user_mat = coo_matrix((data['plays'].astype(float), 
                   (data['artist'].cat.codes, 
                    data['user'].cat.codes)))


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# In[ ]:


class UtilityMatrixTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, confidence=40):
        self.confidence = confidence
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # TODO: abstract column names
        return coo_matrix((data['plays'].astype(float),
                           X["artist"].cat.codes.copy(),
                           X["user"].cat.codes.copy())) * self.confidence


# In[ ]:


class ALSEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, factors=50,
                       regularization=0.01,
                       iterations=10,
                       filter_seen=True):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.filter_seen = filter_seen
        
    def fit(self, X, y=None):
        self.model = AlternatingLeastSquares(factors=self.factors,
                                             regularization=self.regularization,
                                             iterations=self.iterations,
                                             dtype=np.float64,
                                             use_native=True,
                                             use_cg=True)
        self.model.fit(X)
        if self.fiter_seen:
            self.fit_X = X
        return self
        
    def predict(self, X, y=None):
        predictions = np.dot(self.model.item_factors, self.model.user_factors.T)
        if self.filter_seen:
            predictions[self.fit_x.nonzero()] = -99
        return predictions

