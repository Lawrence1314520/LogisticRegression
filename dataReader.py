
import  pandas as pd

selected_ids=[8611161,8810158,886452,883270,859983]

# Optimized: Only loads the specified columns into memory
data = pd.read_csv("breast-cancer.csv")
columns_to_drop=['fractal_dimension_mean','fractal_dimension_se','fractal_dimension_worst']

data=data.drop(columns=columns_to_drop)
filter_data=data[data['id'].isin(selected_ids)]

# To view the output
print(filter_data)
