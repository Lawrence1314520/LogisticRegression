
from LogisticRegression import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import load_data,normalize

X,y=load_data("breast-cancer.csv")

X = normalize(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size=0.2,
    random_state=42
    )

model=LogisticRegression()

model.fit(X_train,y_train)

probabilities=model.predict_probability(X_test)
prediction=(probabilities>=0.5).astype(int)

accuracy=(prediction==y_test).mean()

print(f"Accuracy: {accuracy:.4f}")
print("Prediction: ",prediction)
print("Probability: ",probabilities)
