#v2 sklearn transform 

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

class HadamardKReduder(TransformerMixin, BaseEstimator):

  def __init__(self, n_components = 2, normalize = 'default'):
    self.normalize = normalize
    self.n_components = n_components

  def fit(self, X, y=None):
    return self

  def transform(self, X):
      self._m, self._n = X.shape
      if self.normalize == 'normalize':
        X = self._pre_process(X) 
      hdim = int(2**(round(np.log2(self._n)+1)))
      padd = csr_matrix((self._m, hdim-self._n), dtype='float64') 
      #if scipy.sparse.issparse(X) else np.zeros((self._m, hdim-self._n), dtype='float64') 
      Xh = hstack([X, padd]) if hdim > self._n else X 
      yp = Xh.dot(hadamard(hdim))
      emb = yp[:, 2:self.n_components + 2]
      return emb


# hk = HadamardKReduder(n_components=3)
# hk.fit(x)
# hk.transform(x)