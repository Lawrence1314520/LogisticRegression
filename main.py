
import numpy as np
from LogisticRegression import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import load_data,normalize

# load data from the file
# matrix, 1D vector, 1D vector
# X.shape=(569,30), y.shape=(569,), ids.shape(569,)
X,y,ids=load_data("breast-cancer.csv")

# after normalizing,  all features's number will keep with similar magnitude
X=normalize(X)

# spilting the dataset into two parts of train and test by invoking the train_test_split
# random_state makes the split reproducible
# without it, it'll output a different random split every time when i run the script
X_train, X_test, y_train, y_test,id_train,id_test=train_test_split(
    X, 
    y,
    ids,
    test_size=0.2,
    random_state=100
    )

model=LogisticRegression()
model.fit(X_train,y_train)

probabilities=model.predict_probability(X_test)
prediction=(probabilities>=0.5).astype(int)

for i in range(len(prediction)):
    if  (prediction[i]!=y_test[i])or(probabilities[i]>0.4 and probabilities[i]<0.6):
        print(f"Patient ID: {id_test[i]}")
        print(f"True label: {y_test[i]}")
        print(f"Prediction: {prediction[i]}")
        print(f"Probabilities: {probabilities[i]}")
        print("-"*40)

accuracy=(prediction==y_test).mean()
print(accuracy)


