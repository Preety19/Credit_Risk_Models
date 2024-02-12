import pandas as pd
import numpy as np
import sys
from scipy.stats import norm
import statsmodels.api as sm
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from random import uniform
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')
import os
import json

def dataload(pool_id, segment_id,path):
    pool_dict = {
        7: {1: [2019, 2020, 2021, 2022], 2: [2019, 2020, 2021, 2022], 3: [2019, 2020, 2021, 2022],4: [2019, 2020, 2021, 2022],5: [2019, 2020, 2021, 2022]},
        8: {1: [2019, 2020, 2021, 2022], 2: [2019, 2020, 2021, 2022], 3: [2019, 2020, 2021, 2022],4: [2019, 2020, 2021, 2022],5: [2019, 2020, 2021, 2022]},
        9: {1: [2019, 2020, 2021, 2022], 2: [2019, 2020, 2021, 2022], 3: [2019, 2020, 2021, 2022],4: [2019, 2020, 2021, 2022],5: [2019, 2020, 2021, 2022]},
        11: {1: [2019, 2020, 2021, 2022], 2: [2019, 2020, 2021, 2022], 3: [2019, 2020, 2021, 2022],4: [2019, 2020, 2021, 2022],5: [2019, 2020, 2021, 2022]},
        12: {1: [2019, 2020, 2021, 2022], 2: [2019, 2020, 2021, 2022], 3: [2019, 2020, 2021, 2022],4: [2019, 2020, 2021, 2022],5: [2019, 2020, 2021, 2022]}
    }
    default_rates = pd.read_json(os.path.join(path, 'data1.json'))
    #print(default_rates)
#     default_rates = pd.read_json("default_rate.json")

    macro_eco_fact = pd.read_json(os.path.join(path, 'data2.json'))
    #print(macro_eco_fact)
#     macro_eco_fact = pd.read_json("macro_factors.json")
    
    default_rates['SEGMENT_DESC'] =default_rates['SEGMENT_DESC'].replace({'0-0 DAYS': 1,'1-30 DAYS':2,'31-60 DAYS':3,'61-90 DAYS':4,'ABOVE 91 DAYS':5,'0-0 DAYS ':1,'0-0 Days':1})
    default_rates['ASSESSMENT_DATE'] = pd.to_datetime(default_rates['ASSESSMENT_DATE'])
    default_rates['ASSESSMENT_DATE'] = default_rates['ASSESSMENT_DATE'].dt.year
    macro_eco_fact = macro_eco_fact[['ECF Year', 'Economic Factor Param', 'Parameter Value']]
    macro_eco_fact.rename(
        columns={'ECF Year': 'ASSESSMENT_DATE', 'Economic Factor Param': 'Macro_economic_factor'}, inplace=True
    )
    GDP = macro_eco_fact[macro_eco_fact['Macro_economic_factor'] == 'GDP Growth']
    Inflation = macro_eco_fact[macro_eco_fact['Macro_economic_factor'] == 'Inflation (YoY) (CCPI)']
    GDP = GDP.drop_duplicates(subset=['ASSESSMENT_DATE'])
    Inflation = Inflation.drop_duplicates(subset=['ASSESSMENT_DATE'])
    GDP = GDP[GDP['ASSESSMENT_DATE'].isin(pool_dict[pool_id][segment_id])]
    Inflation = Inflation[Inflation['ASSESSMENT_DATE'].isin(pool_dict[pool_id][segment_id])]
    yearly_macro_factors = GDP.merge(
        Inflation[['ASSESSMENT_DATE', 'Parameter Value', 'Macro_economic_factor']],
        left_on='ASSESSMENT_DATE', right_on='ASSESSMENT_DATE', suffixes=('_GDP', '_Inflation')
    )
    default_rates = default_rates[
        (default_rates['COLLECTIVE_POOL_ID'] == int(pool_id)) &
        (default_rates['SEGMENT_DESC'] == int(segment_id))
    ]
    result_df = default_rates.merge(yearly_macro_factors, on='ASSESSMENT_DATE')
    result_df['PIT_score'] = pd.Series(dtype='float')
    print(result_df)
    print(result_df.columns)
    for i in range(len(result_df)):
        result_df['PIT_score'][i] = norm.ppf(result_df['DEFAULT_RATE'][i])
    mean = result_df['PIT_score'].mean()
    sd = result_df['PIT_score'].std()
    rho = ((sd ** 2) / (1 + sd ** 2))
    k = mean * np.sqrt(1 - rho)
    result_df['Z'] = (k - (result_df['PIT_score'] * np.sqrt(1 - rho)) / np.sqrt(rho))
    reg = result_df[['Z', 'Parameter Value_GDP', 'Parameter Value_Inflation']]
    X = reg[['Parameter Value_GDP', 'Parameter Value_Inflation']]
    Y = reg[['Z']]
    x = sm.add_constant(X, has_constant='add')
    model = sm.OLS(Y, x)
    results = model.fit()  # Fit the model and obtain the results
    parameter_output = results.params
    parameter = pd.DataFrame(parameter_output)
    intercept = results.params[0]
    beta_1 = results.params[1]
    beta_2 = results.params[2]
    future = pd.DataFrame()
    years=[]
    for i in range(6):
        val = max(result_df['ASSESSMENT_DATE']) + i
        years.append(val)
    future['Years'] = years
    future['Future_GDP'] = [round(uniform(0.4, 0.7), 1) for _ in range(len(future))]
    future['Future_Inflation'] = [round(uniform(0.5, 0.9), 1) for _ in range(len(future))]
    future['Future_GDP_Lag'] = pd.Series(dtype='float')
    future['Future_Inflation_Lag'] = pd.Series(dtype='float')
    future['Future_GDP_Lag'][0] = result_df['Parameter Value_GDP'].tail(1).values[0]
    future['Future_Inflation_Lag'][0] = result_df['Parameter Value_Inflation'].tail(1).values[0]
    for i in range(1, len(future)):
        future['Future_GDP_Lag'][i] = future['Future_GDP'][i - 1]
        future['Future_Inflation_Lag'][i] = future['Future_Inflation'][i - 1]
    future['future_Z'] = intercept + (beta_1 * future['Future_GDP_Lag']) + (beta_2 * future['Future_Inflation_Lag'])
    future['PIT_score'] = (k - np.sqrt(rho * future['future_Z']) / np.sqrt(1 - rho))
    future['PIT_PD'] = pd.Series(dtype='float')
    for i in range(len(future)):
        future['PIT_PD'][i] = (scipy.stats.norm.sf(abs(future['PIT_score']))[i])
        future['survival_probability'] = 1 - ((future['PIT_PD']).cumsum())
    future['unconditional_PD'] = pd.Series(dtype='float')
    for i in range(1, len(future)):
        future['unconditional_PD'][0] = future['PIT_PD'][0]
        future['unconditional_PD'][i] = future['PIT_PD'][i] * future['survival_probability'][i - 1]
    future['PIT_PD'] = future['PIT_PD'].apply(lambda x: "{:.5%}".format(x))
    future['PIT_PD'] = future['PIT_PD'].str.replace('%', '')
    future['PIT_PD'] = future['PIT_PD'].astype('float')
    future['Future_PIT_PD'] = future['unconditional_PD']
    future = future[['Years', 'Future_PIT_PD']]
    #print(future)
    future_1 = future.to_json(orient='records')
    return future

# def predict_default_rate(path):
# #     data = pd.read_csv("vasicek_pool_data.csv")
#     data = pd.read_json(os.path.join(path, 'data3.json'))
#     data['ASSESSMENT_DATE'] = pd.to_datetime(data['ASSESSMENT_DATE'])
#     data['year'] = data['ASSESSMENT_DATE'].dt.year
#     data = data.drop(['ASSESSMENT_DATE'], axis=1)
#     df_encoded = pd.get_dummies(data, columns=['COLLECTIVE_POOL_ID', 'SEGMENT_DESC'])
#     train_data = df_encoded[df_encoded['year'].between(2019, 2021)]
#     test_data = df_encoded[df_encoded['year'] == 2022]
#     train_X = train_data.drop('DEFAULT_RATE', axis=1)
#     train_y = train_data['DEFAULT_RATE']
#     test_X = test_data.drop('DEFAULT_RATE', axis=1)
#     test_y = test_data['DEFAULT_RATE']
#     model = LinearRegression()
#     model.fit(train_X, train_y)
#     test_y_pred = model.predict(test_X)
#     plt.figure(figsize=(10, 6))
#     plt.plot(test_y, label='Actual')
#     plt.plot(test_y_pred, label='Predicted')
#     # Set plot title and labels
#     plt.title('Actual vs Predicted')
#     plt.xlabel('Data Points')
#     plt.ylabel('Values')
#     plt.legend()
# #     plt.show()
#     plt.savefig('actual_vs_pred.png')
#     mse = mean_squared_error(test_y, test_y_pred)
#     # Calculate root mean squared error (RMSE)
#     rmse = np.sqrt(mse)
#     # Calculate R-squared (coefficient of determination)
#     r2 = r2_score(test_y, test_y_pred)
#     print("R-squared (Coefficient of Determination):", r2)
#     print("root_mean_square_error :",rmse)
#     scores = cross_val_score(model, train_X, train_y, cv=5, scoring='r2')
#     print("Cross-Validation R-squared scores:", scores)
#     return r2,scores

# def predict_default_rate_1(path):
#     data = pd.read_json(os.path.join(path, 'data3.json'))
#     data['ASSESSMENT_DATE'] = pd.to_datetime(data['ASSESSMENT_DATE'])
#     data['year'] = data['ASSESSMENT_DATE'].dt.year
#     data = data.drop(['ASSESSMENT_DATE'], axis=1)
#     df_encoded = pd.get_dummies(data, columns=['COLLECTIVE_POOL_ID', 'SEGMENT_DESC'])
#     train_data = df_encoded[df_encoded['year'].between(2019, 2021)]
#     test_data = df_encoded[df_encoded['year'] == 2022]
#     train_X = train_data.drop('DEFAULT_RATE', axis=1)
#     train_y = train_data['DEFAULT_RATE']
#     test_X = test_data.drop('DEFAULT_RATE', axis=1)
#     test_y = test_data['DEFAULT_RATE']   
#     # Define the time series cross-validation strategy
#     tscv = TimeSeriesSplit(n_splits=5)   
#     # Create an empty list to store the R-squared scores for each fold
#     r2_scores = []    
#     # Perform backtesting
#     for train_index, test_index in tscv.split(train_X):
#         X_train, X_test = train_X.iloc[train_index], train_X.iloc[test_index]
#         y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]     
#         model = LinearRegression()
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)   
#         r2 = r2_score(y_test, y_pred)
#         r2_scores.append(r2)      
#     # Calculate the average R-squared score across all folds
#     avg_r2 = np.mean(r2_scores)
#     print("Average R-squared (Backtesting):", avg_r2)  
#     # Fit the model using all training data
#     model.fit(train_X, train_y)  
#     # Predict on the test set
#     test_y_pred = model.predict(test_X)   
#     plt.figure(figsize=(10, 6))
#     plt.plot(test_y, label='Actual')
#     plt.plot(test_y_pred, label='Predicted')
#     plt.title('Actual vs Predicted')
#     plt.xlabel('Data Points')
#     plt.ylabel('Values')
#     plt.legend()
#     plt.savefig('actual_vs_pred.png') 
#     mse = mean_squared_error(test_y, test_y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(test_y, test_y_pred)
#     print("R-squared (Coefficient of Determination):", r2)
#     print("Root Mean Square Error (RMSE):", rmse)
#     return r2

if __name__ == '__main__':
    arguments = sys.argv[0]
    path = os.getcwd()
    parent = os.path.dirname(path)
    
    Pool_ids = [7, 8, 9, 11, 12]
    Segment_ids = [1,2,3,4,5]
    all_output = pd.DataFrame()
    for pool_id in Pool_ids:
        for segment_id in Segment_ids:
            output = dataload(pool_id, segment_id,path)
            output['Pool_ID'] = pool_id
            output['Segment_ID'] = segment_id
            all_output = pd.concat([all_output, output], ignore_index=True)
            #all_output = all_output.append(output)
    #pdr = predict_default_rate_1(path)
    all_output.reset_index(drop=True, inplace=True)
    json_output = all_output.to_json(orient='records')
    #model_output = {"modelOutput": json_output}
    # Convert the dictionary to a JSON string
    #modelOutput = json.dumps(model_output, indent=4)    
    #print("modelOutput : ",json_output)
    print("modelOutput: ",json_output)
    #print("modelOutput",all_output.to_json(orient='records'))
#     all_output.to_csv("Vasicek_PD_output.csv", index=False)