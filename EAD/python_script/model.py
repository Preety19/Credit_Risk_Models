import numpy as np
import pandas as pd
import json
import os
import sys
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
import category_encoders as ce
import warnings
warnings.filterwarnings("ignore")


def read_pickle(encoder_file_path):
    with open(encoder_file_path, "rb") as fp:
        encoder = pickle.load(fp)
        fp.close()
    return encoder

# def read_json_input(path):
#     with open(path, "r") as inputFile:
#         data = json.load(inputFile)
#         inputFile.close()
#     return data

def predict(encoder_path, scaler_path, model, df_path):
    encoders = pd.read_pickle(os.path.join(encoder_path, 'EAD_leaveoneoutencoder.pkl'))
    scaler = pd.read_pickle(os.path.join(scaler_path, 'EAD_standardscaler.pkl'))
    ead_lr = pd.read_pickle(os.path.join(model, 'EAD_LinearRegression.pkl'))
    lr = pd.read_pickle(os.path.join(model, 'le_el.pkl'))
    lr_lgd = pd.read_pickle(os.path.join(model, 'LGD_LogisticRegression.pkl'))
    linearreg_lgd =  pd.read_pickle(os.path.join(model, 'LGD_LinearRegression.pkl'))
    df_list = pd.read_json(df_path)
    result = []
    customer_id = []
    Loan_id = []
    region = []
    #sequence = []
    
    for index, row in df_list.iterrows():
        data = pd.DataFrame(row).transpose()
        Cust_id = data['CUSTOMER_ID'].values
        loan_ids = data['LOAN_ID'].values
        region_c = data['REGION'].values
        customer_id.append(Cust_id[0])
        Loan_id.append(loan_ids[0])
        region.append(region_c[0])
        
        data.drop(['CUSTOMER_ID','LOAN_ID','REGION'], axis=1, inplace = True)
    # Convert the dictionary to a DataFrame and append to the list
        row_df = pd.DataFrame(data)
    #for i, df in enumerate(df_list):
        #row_df = pd.DataFrame(df, index=[0])
        credit_amount = row_df['AMT_CREDIT'].values[0]
        encoded_vec = encoders.transform(row_df)
        encoded_data = encoded_vec.to_dict(orient='records')[0]
        Columns_to_be_scaled = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                                'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
                                'CNT_FAM_MEMBERS', 'DAYS_LAST_PHONE_CHANGE']
        subset = {key: encoded_data[key] for key in Columns_to_be_scaled}
        subset = pd.DataFrame(subset, index=[0])
        subset = scaler.transform(subset)
        scaled_values = {}
        for scaled_features in range(len(Columns_to_be_scaled)):
            scaled_values[Columns_to_be_scaled[scaled_features]] = subset[0][scaled_features]
        encoded_data.update(scaled_values)
        encoded_data = pd.DataFrame(encoded_data, index=[0])
    
        #PD = lr.predict_proba(encoded_data)[:, 1]
        
        #recovery_rate_1 = lr_lgd.predict(encoded_data)
        #recovery_rate_2 = linearreg_lgd.predict(encoded_data)
        #recovery_rate = recovery_rate_1 * recovery_rate_2
        #LGD_EL = 1 - recovery_rate
        
        EAD = ead_lr.predict(encoded_data)
        EAD = np.where(EAD < 0, 0, EAD)
        EAD = np.where(EAD > 1, 1, EAD)
        EAD = EAD * credit_amount
        
        #EAD_EL = np.clip(EAD_EL, 0, 1)
        #EAD_EL = credit_amount
        #sequence.append(i)
        result.append(EAD[0])
    #sequence = list(range(len(result)))    
    final_df = pd.DataFrame({'Customer_ID': customer_id,'Loan_ID': Loan_id,'Region': region,'EAD_value': result})
        #print("modelOutput :",EAD_EL)
        
        
#         EL = EAD_EL * PD * LGD_EL
#         EL = pd.DataFrame(EL, columns=[i])
#         result.append(EL)
    
#     result_json = pd.concat(result, axis=1).to_json(orient='records')
#     result_json = pd.concat(EAD_EL, axis=1).to_json(orient='records')
    return final_df

if __name__ == '__main__':
 
    # Fetching parent directory
    path = os.getcwd()
    parent = os.path.dirname(path)
    data_file = "data.json"
    data_path = os.path.join(path,data_file)
    # Print the JSON output
    resulting_values = predict(os.path.join(parent, "encoders"), os.path.join(parent, "std_scaler"), os.path.join(parent, "model"), data_path)
    
    json_output = resulting_values.to_json(orient='records')
    print('modelOutput: ',json_output)