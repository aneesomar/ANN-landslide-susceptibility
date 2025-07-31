import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the datasets
landslides = pd.read_csv("landslides.csv")
nonLandslides = pd.read_csv("non_landslides.csv")

scaler = MinMaxScaler()
columns_to_scale = ['aspect', 'elv', 'flowAcc', 'planCurv', 'profCurv', 'riverProx', 'roadProx', 'slope', 'SPI', 'TPI', 'TRI', 'TWI']  # add all your continuous variables
landslides[columns_to_scale] = scaler.fit_transform(landslides[columns_to_scale])
nonLandslides[columns_to_scale] = scaler.transform(nonLandslides[columns_to_scale])

landslides = pd.get_dummies(landslides, columns=['lithology', 'soil'])
nonLandslides = pd.get_dummies(nonLandslides, columns=['lithology', 'soil'])

landslides.to_csv("output_landslides.csv", index=False)
nonLandslides.to_csv("output_non_landslides.csv", index=False)

