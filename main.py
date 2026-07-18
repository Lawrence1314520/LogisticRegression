
import numpy as np
from LogisticRegression import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import load_data,normalize

X,y,ids=load_data("breast-cancer.csv")

X = normalize(X)

X_train, X_test, y_train, y_test,id_train,id_test=train_test_split(
    X, 
    y,
    ids,
    test_size=0.2,
    random_state=42
    )

model=LogisticRegression()

model.fit(X_train,y_train)

probabilities=model.predict_probability(X_test)
prediction=(probabilities>=0.5).astype(int)

accuracy=(prediction==y_test).mean()
np.set_printoptions(suppress=True, precision=4)

for i in range(len(prediction)):
    if prediction[i] != y_test[i]:
        print(f"Patient ID:      {id_test[i]}")
        print(f"Sample:          {i}")
        print(f"True label:      {y_test[i]}")
        print(f"Probability:     {probabilities[i] * 100:.2f}%")
        print(f"Prediction:      {prediction[i]}")
        print("-" * 40)


