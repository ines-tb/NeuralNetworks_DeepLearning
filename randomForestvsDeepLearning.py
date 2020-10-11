# %%
# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# %%
# Import our input dataset
loansDF = pd.read_csv('./resources/loan_status.csv')
loansDF.head()

# %%
# Both RandomForestClassifier and TensorFlow’s Sequential class require preprocessing.

# BUCKETING NEED CHECK
# ********************************

# Generate our categorical variable list
loansCat = loansDF.dtypes[loansDF.dtypes == "object"].index.tolist()

# Check the number of unique values in each column
loansDF[loansCat].nunique()

# %%
# Check the unique value counts to see if binning is required
loansDF.Years_in_current_job.value_counts()
# => all of the categorical values have a substantial number of data points

# %%
# ENCODING
# ********************************

# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encodeDF = pd.DataFrame(enc.fit_transform(loansDF[loansCat]))

# Add the encoded variable names to the DataFrame
encodeDF.columns = enc.get_feature_names(loansCat)
encodeDF.head()

# %%
# Merge one-hot encoded features and drop the originals
loansDF = loansDF.merge(encodeDF,left_index=True, right_index=True)
loansDF = loansDF.drop(loansCat,1)
loansDF.head()

# %%
# STANDARDIZATION
# ********************************

# Remove loan status target from features data
y = loansDF.Loan_Status_Fully_Paid
X = loansDF.drop(columns=["Loan_Status_Fully_Paid","Loan_Status_Not_Paid"])

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
# RANDOM FOREST
# ********************************
# Create a random forest classifier.
rfModel = RandomForestClassifier(n_estimators=128, random_state=78)

# Fitting the model
rfModel = rfModel.fit(XTrainScaled, yTrain)

# Evaluate the model
yPred = rfModel.predict(XTestScaled)
print(f" Random forest predictive accuracy: {accuracy_score(yTest,yPred):.3f}")


# %%
# DEEP NEURAL NETWORK
# *******************************
# Define the model - deep neural net
numberInputFeatures = len(XTrainScaled[0])
hiddenNodesLayer1 = 24
hiddenNodesLayer2 = 12

nn = Sequential()

# First hidden Layer
nn.add(
    Dense(units=hiddenNodesLayer1, input_dim=numberInputFeatures, activation="relu")
)

# Second hidden layer
nn.add(
    Dense(units=hiddenNodesLayer2, activation="relu")
)

# Output layer
nn.add(Dense(units=1,activation="sigmoid"))

# Compile the Sequential model and customize metrics
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
fitModel = nn.fit(XTrainScaled, yTrain, epochs=50)

# Evaluate the model using the test data
modelLoss, modelAccuracy = nn.evaluate(XTestScaled, yTest, verbose=2)
print(f"Loss: {modelLoss}, Accuracy: {modelAccuracy}")

# %%
# CONCLUSION
# *********************************
# Both the random forest and deep learning models were able to predict correctly 
#   whether or not a loan will be repaid over 80% of the time. Although their 
#   predictive performance was comparable, their implementation and training times 
#   were not—the random forest classifier was able to train on the large dataset 
#   and predict values in seconds, while the deep learning model required a couple 
#   minutes to train on the tens of thousands of data points. 
# In other words, the random forest model is able to achieve comparable predictive 
#   accuracy on large tabular data with less code and faster performance. 
#
# In general, if your dataset is tabular, random forest is a great place to start.