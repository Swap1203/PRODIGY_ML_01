import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv('American_Housing_Data_20231209.csv')

# Step 2: Prepare the data
#Here the columns we require to predict the price are named as Living Space(Square Footage), Beds, and Baths.
A = data[['Living Space', 'Beds', 'Baths']]
# Our target variable or variable to predict is named Price
b = data['Price']

# Spliting the data into training and testing sets
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)

# Step 3: Building and training the model
model = LinearRegression()
model.fit(A_train, b_train)

# Step 4: Asking the user for input to make predictions
print("Enter details for a new house to predict its price:")

living_space = float(input("Square Footage: "))
beds = int(input("Number of Bedrooms: "))
baths = int(input("Number of Bathrooms: "))

# Creating a DataFrame for the new input
new_data = pd.DataFrame({
    'Living Space': [living_space],
    'Beds': [beds],
    'Baths': [baths]
})

# Making predictions
predictions = model.predict(new_data)
print(f'Predicted Price: {predictions[0]}')
