# %%
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

#%%
# **************************************
# NON-LINEAR DATA
# **************************************
# Creating dummy nonlinear data
XMoons, yMoons = make_moons(n_samples=1000, noise=0.08, random_state=78)
# Transforming y_moons to a vertical vector
yMoons = yMoons.reshape(-1, 1)
# Creating a DataFrame to plot the nonlinear dummy data
moonsDF = pd.DataFrame(XMoons, columns=["Feature 1", "Feature 2"])
moonsDF["Target"] = yMoons
# Plot the nonlinear dummy data
moonsDF.plot.scatter(x="Feature 1",y="Feature 2", c="Target",colormap="winter")

#%%
# We can use a different model as we will use 
#   the rule of thumb: hidden neurons = two to three times the number of input reurons
#  ------------------------------------
# Generate our new Sequential model
newModel = Sequential()

# Add the input and hidden layer
numberInputs = 2
numberHiddenNodes = 6 # 3 times the number of Inputs

newModel.add(Dense(units=numberHiddenNodes, activation="relu", input_dim=numberInputs))

# Add the output layer that uses a probability activation function
newModel.add(Dense(units=1, activation="sigmoid"))
# ------------------------------------

# %%
# We can use the same process to compile and train our model as we did in the non-linear model for one neuron
# ---------------------------------------------- 
# Create training and testing sets
XMoonTrain, XMoonTest, yMoonTrain, yMoonTest = train_test_split(
    XMoons, yMoons, random_state=78)
# Create the scaler instance
XMoonScaler = StandardScaler()
# Fit the scaler
XMoonScaler.fit(XMoonTrain)
# Scale the data
XMoonTrainScaled = XMoonScaler.transform(XMoonTrain)
XMoonTestScaled = XMoonScaler.transform(XMoonTest)
# -------------------------------------
# Compile the Sequential model together and customize metrics
newModel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit the model to the training data
newFitModel = newModel.fit(XMoonTrainScaled, yMoonTrain, epochs=100, shuffle=True)

#%%
# Create a DataFrame containing training history
historyDF= pd.DataFrame(newFitModel.history, index=range(1,len(newFitModel.history["loss"])+1))
# Plot the loss
historyDF.plot(y="loss")
# Plot the loss
historyDF.plot(y="acc")

# %%
