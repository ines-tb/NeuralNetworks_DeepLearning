# %%
# Import our dependencies
import pandas as pd
import matplotlib as plt
from sklearn.datasets import make_blobs
import sklearn as skl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# %%
# Generate dummy dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=78)

# Creating a DataFrame with the dummy data
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Target"] = y

# Plotting the dummy data
df.plot.scatter(x="Feature 1", y="Feature 2", c="Target", colormap="winter")

# %%
# Use sklearn to split dataset
XTrain, XTest, yTrain, yTest = train_test_split(X,y,random_state=78)

# %%
# Now that we have the training data let's build the neural network
# Also, we normalize or standardize to avoid focusing on outliers.
# Create scaler instance
XScaler = skl.preprocessing.StandardScaler()

# Fit the scaler
XScaler.fit(XTrain)

# Scale the data
XTrainScaled = XScaler.transform(XTrain)
XTestScaled = XScaler.transform(XTest)

# %%
# Create the Keras Sequential model
nnModel:Sequential = Sequential()

# %%
# Add our first Dense layer, including the input layer
nnModel.add(Dense(units=1, activation="relu", input_dim=2))

# %%
# Add the output layer that uses a probability activation function
nnModel.add(Dense(units=1, activation="sigmoid"))

# %%
# Check the structure of the Sequential model
nnModel.summary()

# %%
# Compile the Sequential model together and customize metrics
nnModel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# %%
# *******************************************
# TRAIN AND TEST A BASIC NEURAL NETWORK
# *******************************************

# Fit the model to the training data
fitModel = nnModel.fit(XTrainScaled, yTrain, epochs=100)

# %%
# Create a DataFrame containing training history
historyDF = pd.DataFrame(fitModel.history, index=range(1,len(fitModel.history["loss"])+1))

# Plot the loss
historyDF.plot(y="loss")

# %%
# Plot the accuracy
historyDF.plot(y="acc")

# %%
# Evaluate the model using the test data
modelLoss, modelAccuracy = nnModel.evaluate(XTestScaled,yTest,verbose=2)
print(f"Loss: {modelLoss}, Accuracy: {modelAccuracy}")

# %%
# **************************************
# PREDICT NEW DATA
# **************************************

# Predict the classification of a new set of blob data
newX, newY = make_blobs(n_samples=10, centers=2, n_features=2, random_state=78)
newXScaled = XScaler.transform(newX)
nnModel.predict_classes(newXScaled)
# %%
