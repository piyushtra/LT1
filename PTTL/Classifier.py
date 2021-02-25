import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("PTTL/drug200.csv")
print(data)

data["label"] = LabelEncoder().fit_transform(data[["Drug"]])
data["Sex_Encoded"] = LabelEncoder().fit_transform(data[["Sex"]])
data["BP_Encoded"] = LabelEncoder().fit_transform(data[["BP"]])
data["Cholesterol_Encoded"] = LabelEncoder().fit_transform(data[["Cholesterol"]])

print(data)
