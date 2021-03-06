# %%
# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Import checkpoint dependencies
import os
from tensorflow.keras.callbacks import ModelCheckpoint


# %%
# SETUP:
# *******************************

# Import our input dataset
attritionDF = pd.read_csv("./resources/HR-Employee-Attrition.csv")
attritionDF.head()

# %%
# Generate our categorical variable list
attritionCat = attritionDF.dtypes[attritionDF.dtypes == "object"].index.tolist()

# %%
# PREPROCESSING:
# *******************************

# Check the number of unique values in each column (No more than 10 per variable)
attritionDF[attritionCat].nunique()
# => none of the categorical variables have more than 10 unique values, 
#   which means we’re ready to encode

# %%
# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encodeDF = pd.DataFrame(enc.fit_transform(attritionDF[attritionCat]))

# Add the encoded variable names to the DataFrame
encodeDF.columns = enc.get_feature_names(attritionCat)
encodeDF.head()

# %%
# Merge one-hot encoded features and drop the originals
attritionDF = attritionDF.merge(encodeDF,left_index=True, right_index=True)
attritionDF = attritionDF.drop(attritionCat,1)
attritionDF.head()

# %%
# Split our preprocessed data into our features and target arrays
y = attritionDF["Attrition_Yes"].values
X = attritionDF.drop(["Attrition_Yes","Attrition_No"],1).values

# Split the preprocessed data into a training and testing dataset
XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=78)

# %%
# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
XScaler = scaler.fit(XTrain)

# Scale the data
XTrainScaled = XScaler.transform(XTrain)
XTestScaled = XScaler.transform(XTest)

# %%
# CREATE DEEP LEARNING MODEL
# *******************************

# Define the model - deep neural net
numberInputFeatures = len(XTrain[0])
hiddenNodesLayer1 =  8
hiddenNodesLayer2 = 5

nn = Sequential()

# First hidden layer
nn.add(
    Dense(units=hiddenNodesLayer1, input_dim=numberInputFeatures, activation="relu")
)

# Second hidden layer
nn.add(Dense(units=hiddenNodesLayer2, activation="relu"))

# Output layer
nn.add(Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()

# %%
# CHECKPOINTS
# *******************************
# Define the checkpoint path and filenames
os.makedirs("checkpoints/",exist_ok=True)
checkpointPath = "checkpoints/weights.{epoch:02d}.hdf5"

# %%
# Compile the model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Create a callback that saves the model's weights every 5 epochs
cpCallback = ModelCheckpoint(
    filepath=checkpointPath,
    verbose=1,
    save_weights_only=True,
    save_freq=1000)

# %%
# TRAIN AND EVALUATE THE MODEL
# *******************************

# Train the model adding the callback Checkpoint
fitmodel = nn.fit(XTrain, yTrain, epochs=100,callbacks=[cpCallback])

# %%
# Evaluate the model using the test data
modelLoss, modelAccuracy = nn.evaluate(XTest, yTest, verbose=2)
print(f"Loss: {modelLoss}, Accuracy: {modelAccuracy}")


# %%
# RESTORE WEIGHTS STORED WITH CHECKPOINTS
# ******************************************

# Define the model - deep neural net
numberInputFeatures = len(XTrain[0])
hiddenNodesLayer1 =  8
hiddenNodesLayer2 = 5

nnNew = Sequential()

# First hidden layer
nnNew.add(
    Dense(units=hiddenNodesLayer1, input_dim=numberInputFeatures, activation="relu")
)

# Second hidden layer
nnNew.add(Dense(units=hiddenNodesLayer2, activation="relu"))

# Output layer
nnNew.add(Dense(units=1, activation="sigmoid"))

# Compile the model
nnNew.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Restore the model weights
nnNew.load_weights("checkpoints/weights.100.hdf5")

# Evaluate the model using the test data
modelLoss, modelAccuracy = nnNew.evaluate(XTestScaled,yTest,verbose=2)
print(f"Loss: {modelLoss}, Accuracy: {modelAccuracy}")
# %%
