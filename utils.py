
import pandas as pd

def load_data(file_name):

    data=pd.read_csv(file_name)

    data["diagnosis"]=data["diagnosis"].map({"M":1,"B":0})

    ids=data["id"].to_numpy()
    X=data.drop(columns=["id","diagnosis"])
    y=data["diagnosis"]

    X=X.to_numpy()
    y=y.to_numpy()

    return X,y,ids

def normalize(X):
   return  (X-X.mean(axis=0))/X.std(axis=0)







