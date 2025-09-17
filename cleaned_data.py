import pandas as pd

data = pd.read_csv('data/Crime_Data_from_2020_to_Present.csv')

# print(", ".join(data.columns.tolist()))

data = data[['Crm Cd', 'Crm Cd Desc', 'Vict Age', 'Vict Sex']]

# Creates a new column and sets the value to 1 at all locations with M, and 0 at all else.
data['Vict Sex Num'] = (data['Vict Sex'] == 'M').astype(int)

print(data.dtypes)

data.to_csv('./data/cleaned_crime_data.csv')