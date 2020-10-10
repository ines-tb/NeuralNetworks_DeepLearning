# %%
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

# %%
# We can use the same Sequential model object as of basicNeuralNetwork
# ------------------------------------
# Create the Keras Sequential model
nnModel: Sequential = Sequential()
# Add our first Dense layer, including the input layer (input_dim is the number of input neurons)
nnModel.add(Dense(units=1, activation="relu", input_dim=2))
# Add the output layer that uses a probability activation function
nnModel.add(Dense(units=1, activation="sigmoid"))
# Check the structure of the Sequential model
nnModel.summary()
# Compile the Sequential model together and customize metrics
nnModel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# ------------------------------------

# %%
# Create training and testing sets
XMoonTrain, XMoonTest, yMoonTrain, yMoonTest = train_test_split(
    XMoons, yMoons, random_state=78
)

# Create the scaler instance
XMoonScaler = StandardScaler()

# Fit the scaler
XMoonScaler.fit(XMoonTrain)

# Scale the data
XMoonTrainScaled = XMoonScaler.transform(XMoonTrain)
XMoonTestScaled = XMoonScaler.transform(XMoonTest)

# %%
# Training the model with the nonlinear data
modelMoon = nnModel.fit(XMoonTrainScaled, yMoonTrain, epochs=100, shuffle=True)

# %%
# Create a DataFrame containing training history
historyDF= pd.DataFrame(modelMoon.history, index=range(1,len(modelMoon.history["loss"])+1))

# Plot the loss
historyDF.plot(y="loss")
# %%
# Plot the loss
historyDF.plot(y="acc")
# %%
