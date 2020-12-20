from datetime import date
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
url_to_covid = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'

initial_df = pd.read_csv(url_to_covid)

test_df = initial_df[initial_df.location == 'India']
"""
plt.scatter(sorted(test_df.date), test_df.new_cases)
plt.show()
"""
_ = plt.figure(figsize=(30, 15))
sns.scatterplot(sorted(test_df.date), test_df.new_cases)
#plt.show()
df = initial_df.copy()

percent_missing = df.isnull().sum() * 100 / len(df)
#print(percent_missing)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', inplace=True, ascending=False)

#print(missing_value_df)

cols_too_many_missing = missing_value_df[missing_value_df.percent_missing > 50].index.tolist()

df_reduced = df.drop(columns=cols_too_many_missing)

df = df_reduced

missing_iso_code = df[df.iso_code.isna()]
df = df.drop(index=missing_iso_code.index)

missing_continent = df[df.continent.isna()]
df = df.drop(index=missing_continent.index)
"""
for col in df.columns:
    print("%s\t%d"%(col,df[col].isna().sum()))
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

nominal = df.select_dtypes(include=['object']).copy()
nominal_cols = nominal.columns.tolist()
# print(nominal_cols)

encoder = LabelEncoder()
for col in nominal_cols:
    if df[col].isna().sum()>0:
        df[col].fillna('MISSING', inplace=True)
    df[col] = encoder.fit_transform(df[col])
"""

for col in nominal_cols:
    print(df[col].unique())
"""

numerical = df.select_dtypes(include=['float64']).copy()

for col in numerical:
    df[col].fillna((df[col].mean()), inplace=True)

X = df.drop(columns=['new_cases'])
y = df.new_cases

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score 
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump, load

rf = RandomForestRegressor(
    n_estimators = 100, # 400 
    random_state = 0, 
    max_depth=30)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f'{r2_score(y_test, y_pred):.2%}')

random_grid = {'n_estimators': np.arange(200,600,100), 'max_depth': np.arange(10,40,10)}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,  n_iter = 100, cv = 3, verbose=2, random_state=0)

rf_random.fit(X_train, y_train)

rf_random.best_params_

rf = RandomForestRegressor(**rf_random.best_params_, random_state = 1)
y_pred = rf.predict(X_test)
print(f'{r2_score(y_test, y_pred):.2%}')

#dump(rf, 'rf_model.joblib') 
dump(rf, 'rf_model.joblib',compress=3)
#dump(rf, 'rf_model.pkl.z')
