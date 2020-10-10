# %%
# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# %%
# Import our input dataset
diabetesDF = pd.read_csv('./resources/diabetes.csv')
diabetesDF.head()

# %%
# Logistic Regression does not need any data preparation. However,
# Neural Networks requires standardized data.

# Remove diabetes outcome target from features data
y = diabetesDF.Outcome
X = diabetesDF.drop(columns="Outcome")

# Split training/test datasets
XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=42, stratify=y)

# %%
# Preprocess numerical data for neural network

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
XScaler = scaler.fit(XTrain)

# Scale the data
XTrainScaled = XScaler.transform(XTrain)
XTestScaled = XScaler.transform(XTest)

# %%
# LOGISTIC REGRESSION:

# Define the logistic regression model
logClassifier = LogisticRegression(solver="lbfgs",max_iter=200)

# Train the model
logClassifier.fit(XTrain,yTrain)

# Evaluate the model
yPred = logClassifier.predict(XTest)
print(f" Logistic regression model accuracy: {accuracy_score(yTest,yPred):.3f}")

# %%
# NEURAL NETWORK
# Define the basic neural network model
nnmodel = Sequential()
nnmodel.add(Dense(units=16, activation="relu", input_dim=8))
nnmodel.add(Dense(units=1, activation="sigmoid"))

# Compile the Sequential model together and customize metrics
nnmodel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
fit_model = nnmodel.fit(XTrainScaled, yTrain, epochs=100)

# Evaluate the model using the test data
modelLoss, modelAccuracy = nnmodel.evaluate(XTestScaled,yTest,verbose=2)
print(f"Loss: {modelLoss}, Accuracy: {modelAccuracy}")

# %%
