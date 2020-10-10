#%%
# Import our dependencies
import pandas as pd
import sklearn as skl
from sklearn.preprocessing import OneHotEncoder


#%%
# Read in our ramen data
ramenDF = pd.read_csv("./resources/ramen-ratings.csv")

# Print out the Country value counts
countryCounts = ramenDF.Country.value_counts()
countryCounts

# %%
# Visualize the value counts
countryCounts.plot.density()

# %%
# We can bucket any country that appears fewer than 
#   100 times in the dataset as “other”.
# Determine which values to replace
replaceCountries = list(countryCounts[countryCounts < 100].index)

# Replace in DataFrame
for country in replaceCountries:
    ramenDF.Country = ramenDF.Country.replace(country,"Other")

# Check to make sure binning was successful
ramenDF.Country.value_counts()

# %%
# Create the OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit the encoder and produce encoded DataFrame
encodeDF = pd.DataFrame(enc.fit_transform(ramenDF.Country.values.reshape(-1,1)))

# Rename encoded columns
encodeDF.columns = enc.get_feature_names(['Country'])
encodeDF.head()

# %%
# Merge the two DataFrames together and drop the Country column
ramenDF.merge(encodeDF,left_index=True,right_index=True).drop("Country",1)

# %%
