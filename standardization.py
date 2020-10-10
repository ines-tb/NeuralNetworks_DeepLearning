#%%
# Import our dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler

# %%
# Read in our dataset
hrDF = pd.read_csv("./resources/hr_dataset.csv")
hrDF.head()

# If this dataset contained categorical data, we would need to Slice out 
#   the categorical data prior to scaling.

# %%
# Create the StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
scaler.fit(hrDF)

# Scale the data
scaledData = scaler.transform(hrDF)

# %%
# Create a DataFrame with the scaled data
transformedScaledData = pd.DataFrame(scaledData, columns=hrDF.columns)
transformedScaledData.head()
# => all of the variables have now been standardized, with a mean value of 0 
#   and a standard deviation of 1. Now the data is ready to be passed along to our 
#   neural network model.

# %%
