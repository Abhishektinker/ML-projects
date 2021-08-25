class MyLinearRegression:
    def __init__(self):
        self._coef = None
        self._intercept = None
    
    def fit(self, feature, target):
        """This method will make our model learn and find the best values 
        of coeffiecient and intercept"""
        feature = np.array(feature)
        target = np.array(target)
        X = np.c_[np.ones(len(feature)), feature]
        self.theta = (np.linalg.pinv(X.T.dot(X))).dot(X.T).dot(target)
        self.coef_ = self.theta[1:]
        self.intercept_ = self.theta[0]
        
    def predict(self, X_test):
        try:
            len(X_test)
        except:
            raise ValueError("Testing Data Should Be Iterable Object") 
        else:
            X = np.array(X_test)
            X = np.c_[np.ones(len(X_test)), X]
            y_test = self.theta.dot(X.T)
            return y_test
        
    def mean_absolute_error(self, y, y_hat):
        return (abs((y - y_hat)).sum())/len(y)
    
    def mean_square_error(self, y, y_hat):
        return (((y - y_hat)**2).sum())/len(y)
    
    def rmse(self, y, y_hat):
        return np.sqrt((((y - y_hat)**2).sum())/len(y))

    def r2_score(self, y, y_hat):
        r2score = 1 - (sum((y - y_hat)**2) / sum((y - y.mean())**2))
        return r2score
