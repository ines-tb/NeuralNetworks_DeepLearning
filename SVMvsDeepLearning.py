# %%
# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# %%
# Import our input dataset
teleDF = pd.read_csv('./resources/bank_telemarketing.csv')
teleDF.head()

# %%
# CHECK BUCKETING
# *******************************
# Check if categorical data require bucketing

# Generate our categorical variable list
teleCat = teleDF.dtypes[teleDF.dtypes == "object"].index.tolist()

# Check the number of unique values in each column
teleDF[teleCat].nunique()
# => no categories require bucketing prior to encoding

# %%
# ENCODE DATA
# *******************************

# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encodeDF = pd.DataFrame(enc.fit_transform(teleDF[teleCat]))

# Add the encoded variable names to the dataframe
encodeDF.columns = enc.get_feature_names(teleCat)
encodeDF.head()

# %%
# Merge one-hot encoded features and drop the originals
teleDF = teleDF.merge(encodeDF,left_index=True, right_index=True)
teleDF = teleDF.drop(teleCat,1)
teleDF.head()

# %%
# SPLIT DATA INTO TRAIN AND TEST
# *******************************

# Remove loan status target from features data
y = teleDF.Subscribed_yes.values
X = teleDF.drop(columns=["Subscribed_no","Subscribed_yes"]).values

# Split training/test datasets
XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=42, stratify=y)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
XScaler = scaler.fit(XTrain)

# Scale the data
XTrainScaled = XScaler.transform(XTrain)
XTestScaled = XScaler.transform(XTest)

# %%
# SVM MODEL
# *********************************

# Create the SVM model
svm = SVC(kernel='linear')

# Train the model
svm.fit(XTrain, yTrain)

# Evaluate the model
yPred = svm.predict(XTestScaled)
print(f" SVM model accuracy: {accuracy_score(yTest,yPred):.3f}")


# %%
# DEEP LEARNING MODEL
# *********************************

# - The first hidden layer will have an input_dim equal to the 
#       length of the scaled feature data X , 10 neuron units, and will 
#       use the relu activation function.
# - Our second hidden layer will have 5 neuron units and also will use 
#       the relu activation function.
# - The loss function should be binary_crossentropy, using the adam optimizer

# Define the model - deep neural net
numberInputFeatures = len(XTrainScaled[0])
hiddenNodesLayer1 =  10 # => we do not use 2 or 3 times input neurons to avoid overfitting
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

# Compile the Sequential model together and customize metrics
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# %%
# Train the model 
fitModel = nn.fit(XTrainScaled, yTrain, epochs=50) # Few features so we need less epochs
# Evaluate the model using the test data 
modelLoss, modelAccuracy = nn.evaluate(XTestScaled,yTest,verbose=2)
print(f"Loss: {modelLoss}, Accuracy: {modelAccuracy}")

# %%
# CONCLUSION
# **************************
# The SVM and deep learning models both achieved a predictive accuracy around 87%. 
# Additionally, both models take similar amounts of time to train on the input data.
# The only noticeable difference between the two models is implementation—the amount 
#   of code required to build and train the SVM is notably less than the comparable 
#   deep learning model.