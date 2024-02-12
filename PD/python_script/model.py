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
    return encoder

def read_json_input(path):
    with open(path, "r") as inputFile:
        data = json.load(inputFile)
    return data

def predict(encoder_path, scaler_path, model_path, df_path):
    encoders = pd.read_pickle(os.path.join(encoder_path, 'encoder.pkl'))
    scaler = pd.read_pickle(os.path.join(scaler_path, 'std_scaler.pkl'))
    lr = pd.read_pickle(os.path.join(model_path, 'lr.pkl'))
    #df_list = read_json_input(df_path)
    df_list = pd.read_json(df_path)
    
    predictions = []
    customer_id = []
    Loan_id = []
    region = []
    Columns_to_be_scaled = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                            'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
                            'CNT_FAM_MEMBERS', 'DAYS_LAST_PHONE_CHANGE']
    
     
    #for record in df_list:
    for index, row in df_list.iterrows():
        data = pd.DataFrame(row).transpose()
        #print(data)
        Cust_id = data['CUSTOMER_ID'].values
        loan_ids = data['LOAN_ID'].values
        region_c = data['REGION'].values
        customer_id.append(Cust_id[0])
        Loan_id.append(loan_ids[0])
        region.append(region_c[0])
        #print(customer_id)
        data.drop(['CUSTOMER_ID','LOAN_ID','REGION'], axis=1, inplace = True)
        encoded_data = encoders.transform(pd.DataFrame(data))    
        #encoded_data = encoders.transform(pd.DataFrame(record, index=[0]))
        encoded_data = encoded_data.to_dict(orient='records')[0]

        subset = {key: encoded_data[key] for key in Columns_to_be_scaled}
        subset = pd.DataFrame(subset, index=[0])
        subset = scaler.transform(subset)

        scaled_values = {}
        for scaled_features in range(len(Columns_to_be_scaled)):
            scaled_values[Columns_to_be_scaled[scaled_features]] = subset[0][scaled_features]

        encoded_data.update(scaled_values)
        encoded_data = pd.DataFrame(encoded_data, index=[0])
        pred = lr.predict(encoded_data)
        pred_prob = lr.predict_proba(encoded_data)[:, 1]
        predictions.append(pred_prob[0])
    #sequence = list(range(len(predictions)))    
    final_df = df = pd.DataFrame({'Customer_ID': customer_id,'Loan_ID': Loan_id,'Region': region,'PD_value': predictions})
    
    return final_df

if __name__ == '__main__':
    
    path = os.getcwd()
    parent = os.path.dirname(path)
    data_file = 'data.json'
    data_path = os.path.join(path,data_file)
    resulting_values = predict(os.path.join(parent, "encoders"), os.path.join(parent, "std_scaler"),
                          os.path.join(parent, "model"), data_path)
    
    json_output = resulting_values.to_json(orient='records')
    print('modelOutput: ',json_output)
    
    