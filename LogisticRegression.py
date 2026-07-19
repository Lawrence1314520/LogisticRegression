
import numpy as np

class LogisticRegression:
    def __init__(self,learning_rate=0.01,epochs=1000):
        self.lr=learning_rate
        self.epochs=epochs
        self.w=None
        self.b=None
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def fit(self,X,y):

        n_samples,n_features=X.shape
        self.w=np.zeros(n_features)
        self.b=0.0

        for epoch in range(self.epochs):
            
            linear=np.dot(X,self.w)+self.b
            y_pred=self.sigmoid(linear)

            # cross entropy
            epsilon=1e-15
            loss=-np.mean(y*np.log(y_pred+epsilon)+(1-y)*np.log(1-y_pred+epsilon))

            # calculate the partial derivative of w and b in loss(w,b)
            # using Chain Rule with the dependencies: L->p->z->w,b
            dw=(1/n_samples)*np.dot(X.T,(y_pred-y))
            db=(1/n_samples)*np.sum(y_pred-y)

            self.w-=self.lr*dw
            self.b-=self.lr*db

            if epoch%100==0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    def predict_probability(self,X):
        
        linear=np.dot(X,self.w)+self.b
        return self.sigmoid(linear)
    
    def predict(self,X):

        probabilities=self.predict_probability(X)
        return (probabilities>=0.5).astype(int)
