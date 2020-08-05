from scipy.sparse import csr_matrix, vstack, hstack
from scipy.linalg import hadamard

class HadamardReduder():
  def __init__(self, n_components = 2, normalize = 'default'):
    self.normalize = normalize
    self.n_components = n_components
  
  def _pre_process(self, X):
    X = X.toarray() if scipy.sparse.issparse(X) else X
    return (X-np.mean(X, axis = 0))/ np.mean(X, axis = 0)

  def transform(self, X):
    self._m, self._n = X.shape
    if self.normalize == 'normalize':
      X = self._pre_process(X) 
    hdim = int(2**(round(np.log2(self._n)+1)))
    padd = csr_matrix((self._m, hdim-self._n), dtype='float64') 
    #csr_matrix((self._m, hdim-self._n), dtype='float64') if scipy.sparse.issparse(X) 
      #else np.zeros((self._m, hdim-self._n), dtype='float64') 
    Xh = hstack([X, padd]) if hdim > self._n else X 
    yp = Xh.dot(hadamard(hdim))
    return yp[:, 2:self.n_components + 2]
    # X = self._pre_process(X)

##had = HadamardReduder(n_components = 3)
##had.transform(x)

     