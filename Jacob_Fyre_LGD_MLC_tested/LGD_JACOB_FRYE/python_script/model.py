import pandas as pd
import numpy as np
import sys
from scipy.stats import norm
import statsmodels.api as sm
import scipy.stats
import warnings
from sklearn.metrics import r2_score, mean_squared_error
import json
warnings.filterwarnings('ignore')
from random import uniform
import argparse
import os


def dataload(path,pool_id):
    pool_dict = {
                 7 : [2019,2020,2021,2022],
                 8 : [2019,2020,2021,2022],
                 9 : [2019,2020,2021,2022],
                 11 : [2019,2020,2021,2022],
                 12 : [2019,2020,2021,2022],
                }
    
#     default_rates = pd.read_csv("Defaulter_rate.csv")
    default_rates = pd.read_json(os.path.join(path, 'data1.json'))
#     macro_eco_fact = pd.read_csv("Economic_Factor_value_vasicek.csv")
    macro_eco_fact = pd.read_json(os.path.join(path, 'data2.json'))
    #1) additional file read for jacob frye
#     jacob_frye = pd.read_csv("jacob_frye_demo.csv",encoding = 'cp1252')
    jacob_frye = pd.read_json(os.path.join(path, 'data3.json'))
    #2) subset for jacob frye using 3 columns needed
    # jacob_frye = jacob_frye[['ACCOUNT_NO','PD','LGD','COLLECTIVE_POOL_ID']]
#     print(jacob_frye)
    default_rates['ASSESSMENT_DATE'] = pd.to_datetime(default_rates['ASSESSMENT_DATE'])
    default_rates['ASSESSMENT_DATE'] = default_rates['ASSESSMENT_DATE'].dt.year
    #1.1) selecting subset of required column from macroeconomic factor table
    macro_eco_fact = macro_eco_fact[['ECF Year','Economic Factor Param','Parameter Value']]
    #1.2) renaming the column according to requirements
    macro_eco_fact.rename(columns = {'ECF Year':'ASSESSMENT_DATE','Economic Factor Param':'Macro_economic_factor'},inplace = True)
    GDP = macro_eco_fact[macro_eco_fact['Macro_economic_factor'] == 'GDP Growth']
    Inflation = macro_eco_fact[macro_eco_fact['Macro_economic_factor'] == 'Inflation (YoY) (CCPI)']
    GDP = GDP.drop_duplicates(subset=['ASSESSMENT_DATE'])
    Inflation = Inflation.drop_duplicates(subset=['ASSESSMENT_DATE'])
    GDP = GDP[GDP['ASSESSMENT_DATE'].isin(pool_dict[pool_id])]
    Inflation = Inflation[Inflation['ASSESSMENT_DATE'].isin(pool_dict[pool_id])]
    yearly_macro_factors = GDP.merge(Inflation[['ASSESSMENT_DATE','Parameter Value','Macro_economic_factor']],left_on='ASSESSMENT_DATE', right_on='ASSESSMENT_DATE',
          suffixes=('_GDP', '_Inflation'))
    default_rates = default_rates[default_rates['COLLECTIVE_POOL_ID'] == int(pool_id)]
    result_df = default_rates.merge(yearly_macro_factors, on ='ASSESSMENT_DATE')
    result_df['PIT_score'] = pd.Series(dtype='float')
    for i in range(0,len(result_df)):
        result_df['PIT_score'][i] = norm.ppf(result_df['DEFAULT_RATE'][i])
    mean = result_df['PIT_score'].mean()
    sd = result_df['PIT_score'].std()
    rho = ((sd ** 2)/(1 + sd ** 2))
    k = mean * np.sqrt(1 - rho)
    result_df['Z'] = (k -(result_df['PIT_score']*np.sqrt(1 - rho))/np.sqrt(rho))
    reg = result_df[['Z','Parameter Value_GDP','Parameter Value_Inflation']]
    X=reg[['Parameter Value_GDP','Parameter Value_Inflation']]
    Y =reg[['Z']]
    x = sm.add_constant(X,has_constant='add')
    model = sm.OLS(Y, x).fit()
    parameter_output = model.params
    parameter = pd.DataFrame(parameter_output) 
    intercept = model.params[0]
    beta_1 = model.params[1]
    beta_2 = model.params[2]  
    future = pd.DataFrame()
    years=[]
    for i in range(len(result_df)):
        val = max(result_df['ASSESSMENT_DATE']) + i
        years.append(val)
    future['Years'] = years   
    future['Future_GDP'] = [round(uniform(0.4, 0.7), 1) for i in range(len(future))]
    future['Future_Inflation'] = [round(uniform(0.5, 0.9), 1) for i in range(len(future))]  
    future['Future_GDP_Lag'] = pd.Series(dtype = 'float')
    future['Future_Inflation_Lag'] = pd.Series(dtype = 'float')
    future['Future_GDP_Lag'][0] = result_df['Parameter Value_GDP'].tail(1).values[0]
    future['Future_Inflation_Lag'][0] = result_df['Parameter Value_Inflation'].tail(1).values[0]
    for i in range (1,len(future)):
        future['Future_GDP_Lag'][i] = future['Future_GDP'][i-1] 
        future['Future_Inflation_Lag'][i] = future['Future_Inflation'][i-1] 
    future['future_Z'] = intercept + (beta_1 * future['Future_GDP_Lag'] )+ (beta_2 * future['Future_Inflation_Lag'])
    future['PIT_score'] = (k-np.sqrt(rho*future['future_Z'])/np.sqrt(1 - rho))
    future['PIT_PD'] = pd.Series(dtype='float')
    for i in range(0,len(future)):
        future['PIT_PD'][i] = (scipy.stats.norm.sf(abs(future['PIT_score']))[i])
        future['survival_probability'] = 1 - ((future['PIT_PD']).cumsum())
    future['unconditional_PD'] = pd.Series(dtype='float')
    for i in range(1,len(future)):
        future['unconditional_PD'][0] = future['PIT_PD'][0]
        future['unconditional_PD'][i] = future['PIT_PD'][i] * future['survival_probability'][i-1]   
    future['PIT_PD'] = future['PIT_PD'].apply(lambda x: "{:.5%}".format(x))
    future['PIT_PD'] = future['PIT_PD'].str.replace('%', '')
    future['PIT_PD'] = future['PIT_PD'].astype('float')
    future = future[['Years','unconditional_PD','PIT_PD']]

## PART 2 -JACOB FRYE
    jacob_frye = jacob_frye[jacob_frye['COLLECTIVE_POOL_ID'] == int(pool_id)]
    TTC_pd = jacob_frye['PD'].mean()/100
    TTC_lgd = jacob_frye['LGD'].mean()/100
    given_TTC_PD = TTC_pd
    given_TTC_LGD = TTC_lgd
    rho = rho
    k = (norm.ppf(given_TTC_PD)-norm.ppf(given_TTC_PD*given_TTC_LGD))/np.sqrt(1-0.1)
    future['PIT_PD'] = future['PIT_PD']/100
    future["PIT_lgd"] = norm.cdf(norm.ppf(future['PIT_PD'])-k)/future['PIT_PD']
    future = future[['Years','PIT_lgd']]
    future_prediction = future.to_json(orient = 'records')
    #print("finalOutput",future_prediction)
#     future.to_csv('jacob_frye.csv')
    return future

if __name__ == '__main__': 
    path = os.getcwd()   
    Pool_ids = [7,8,9,11,12]
    all_output = pd.DataFrame()
    for pool_id in Pool_ids:
        #dataload(pool_id)
        output = dataload(path, pool_id)
        output['Pool_ID'] = pool_id
        all_output = pd.concat([all_output, output], ignore_index=True)
    all_output.reset_index(drop=True, inplace=True)    
    json_output = all_output.to_json(orient='records')
    print("modelOutput: ",json_output)
    
        #line 150 : alternative for f-string
#         Filename = f"jacob_frye_output_{pool_id}.csv"
#         Filename = "jacob_frye_output_{}.csv".format(pool_id)
        #Filename = "jacob_frye_output_" + str(pool_id) + ".csv"
#         output.to_csv(Filename,index=False)