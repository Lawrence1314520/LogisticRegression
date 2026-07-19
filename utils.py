
import pandas as pd

def load_data(file_name):

    # data is a Pandas DataFrame with shape(592,32)
    # dtypes: id->int64, diagnosis->string, all others->float64
    data=pd.read_csv(file_name)

    # this is a simple value replacement not hashing(after operation the column is numeric)
    data["diagnosis"]=data["diagnosis"].map({"M":1,"B":0})

    # .to_numpy() extracts the column as a NumPy array
    # the shape of ids is (596,), which is 1D array (treat it as a vector), not (596,1)
    # the dtype is still int64
    ids=data["id"].to_numpy()

    # X: DataFrame (569,30), dtype is float64
    X=data.drop(columns=["id","diagnosis"])
    y=data["diagnosis"]
    
    # after .to_numpy(), X: NumPy array of shape (569,30), dtype is float64
    X=X.to_numpy()
    # after .to_numpy(), y: NumPy array of shape (569,), int64 (0/1)
    y=y.to_numpy()
    
    '''
    print(data.shape)                    # (569, 32)
    print(data.dtypes)                   # check types
    print(X.shape, y.shape, ids.shape)   # (569, 30) (569,) (569,)
    print(y[:5])                         # should show 1s and 0s
    '''
    
    return X,y,ids

# feature-wise standardization 特征级标准化(z-scoure normalization 归一化)
# std: standard deviation 标准差
# all features's number will keep with similar magnitude
# avoid feature like area_mean(~100s-1000s) dominating features like smoothness_mean(~0.1)
# critical for gradient descent
def normalize(X):
   # return a 1D array of shape (30,)-each element is the mean of a feature
   mean=X.mean(axis=0) # shape(30,)
   # the method to calculate std is same with mean
   std=X.std(axis=0) # shape(30,)
   # scale it to unit variance  单位方差(most values fall roughly between -3 and 3)
   return  (X-mean)/std









