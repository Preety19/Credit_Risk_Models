import pandas as pd
import numpy as np
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

def predict(encoder_path, scaler_path, model_path, df_path):
    #PD
    encoders_PD = pd.read_pickle(os.path.join(encoder_path, 'encoder_PD.pkl'))
    scaler_PD = pd.read_pickle(os.path.join(scaler_path, 'std_scaler_PD.pkl'))
    lr_PD = pd.read_pickle(os.path.join(model_path, 'lr_PD.pkl'))
    
    #LGD
    encoders_LGD = pd.read_pickle(os.path.join(encoder_path,'LGDLogistic_leaveoneoutencoder.pkl'))
    scaler_LGD = pd.read_pickle(os.path.join(scaler_path,'LGDLogistic_standardscaler.pkl'))
    encoders_linear_LGD = pd.read_pickle(os.path.join(encoder_path,'LGDLinear_leaveoneoutencoder.pkl'))
    scaler_linear_LGD = pd.read_pickle(os.path.join(scaler_path,'LGDLinear_standardscaler.pkl'))
    linear_LGD = pd.read_pickle(os.path.join(model_path,'LGD_LinearRegression.pkl'))
    logistic_LGD = pd.read_pickle(os.path.join(model_path,'LGD_LogisticRegression.pkl'))
    
    #EAD
    encoders_EAD = pd.read_pickle(os.path.join(encoder_path, 'EAD_leaveoneoutencoder.pkl'))
    scaler_EAD = pd.read_pickle(os.path.join(scaler_path, 'EAD_standardscaler.pkl'))
    lr_EAD = pd.read_pickle(os.path.join(model_path, 'EAD_LinearRegression.pkl'))
        
    df_list = pd.read_json(df_path)
    df_list['PORTFOLIO'] = 'Home Loan'
    predictions_PD = []
    predictions_LGD = []
    predictions_EAD = []    
    customer_id = []
    Loan_id = []
    region = []
    portfolio = []
    Columns_to_be_scaled = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                            'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
                            'CNT_FAM_MEMBERS', 'DAYS_LAST_PHONE_CHANGE']
    for index, row in df_list.iterrows():
        data = pd.DataFrame(row).transpose()
        Cust_id = data['CUSTOMER_ID'].values
        loan_ids = data['LOAN_ID'].values
        region_c = data['REGION'].values
        port = data['PORTFOLIO'].values
        customer_id.append(Cust_id[0])
        Loan_id.append(loan_ids[0])
        region.append(region_c[0])
        portfolio.append(port[0])
        data.drop(['CUSTOMER_ID','LOAN_ID','REGION','PORTFOLIO'], axis=1, inplace = True)
        encoded_data_PD = encoders_PD.transform(pd.DataFrame(data))    
        encoded_data_PD = encoded_data_PD.to_dict(orient='records')[0]
        subset_PD = {key: encoded_data_PD[key] for key in Columns_to_be_scaled}
        subset_PD = pd.DataFrame(subset_PD, index=[0])
        subset_PD = scaler_PD.transform(subset_PD)
        scaled_values_PD = {}
        for scaled_features_PD in range(len(Columns_to_be_scaled)):
            scaled_values_PD[Columns_to_be_scaled[scaled_features_PD]] = subset_PD[0][scaled_features_PD]
        encoded_data_PD.update(scaled_values_PD)
        encoded_data_PD = pd.DataFrame(encoded_data_PD, index=[0])
        pred = lr_PD.predict(encoded_data_PD)
        pred_prob = lr_PD.predict_proba(encoded_data_PD)[:, 1]
        predictions_PD.append(pred_prob[0])
                
    #####################################################  EAD  ########################################################
        row_df_EAD = pd.DataFrame(data)
        credit_amount = row_df_EAD['AMT_CREDIT'].values[0]
        encoded_vec_EAD = encoders_EAD.transform(row_df_EAD)
        encoded_data_EAD = encoded_vec_EAD.to_dict(orient='records')[0]        
        subset_EAD = {key: encoded_data_EAD[key] for key in Columns_to_be_scaled}
        subset_EAD = pd.DataFrame(subset_EAD, index=[0])
        subset_EAD = scaler_EAD.transform(subset_EAD)
        scaled_values_EAD = {}
        for scaled_features_EAD in range(len(Columns_to_be_scaled)):
            scaled_values_EAD[Columns_to_be_scaled[scaled_features_EAD]] = subset_EAD[0][scaled_features_EAD]
        encoded_data_EAD.update(scaled_values_EAD)
        encoded_data_EAD = pd.DataFrame(encoded_data_EAD, index=[0])   
        EAD = lr_EAD.predict(encoded_data_EAD)
        EAD = np.where(EAD < 0, 0, EAD)
        EAD = np.where(EAD > 1, 1, EAD)
        EAD = EAD * credit_amount        
        predictions_EAD.append(EAD[0])       
      
    #####################################################  LGD  ########################################################    
        row_df_LGD = pd.DataFrame(data)    
        encoded_vec_LGD = encoders_LGD.transform(row_df_LGD)
        encoded_data_LGD = encoded_vec_LGD.to_dict(orient='records')[0]
        subset_LGD = {key: encoded_data_LGD[key] for key in (Columns_to_be_scaled)}
        subset_LGD = pd.DataFrame(subset_LGD,index=[0])
        subset_LGD = scaler_LGD.transform(subset_LGD)
        scaled_values_LGD = {}
        for scaled_features_LGD in range(len(Columns_to_be_scaled)):
            scaled_values_LGD[Columns_to_be_scaled[scaled_features_LGD]]=subset_LGD[0][scaled_features_LGD]    
        # converting the encoded data from dictionary into dataframe and predicting the test data
        encoded_data_LGD.update(scaled_values_LGD)
        encoded_data_LGD = pd.DataFrame(encoded_data_LGD,index=[0])
        results_LGD = encoded_data_LGD.items()
        logistic_prediction_LGD = logistic_LGD.predict(encoded_data_LGD)
               
        row_df_linear_LGD = pd.DataFrame(data)#.transpose()    
        encoded_vec_linear_LGD = encoders_linear_LGD.transform(row_df_linear_LGD)
        encoded_data_linear_LGD = encoded_vec_linear_LGD.to_dict(orient='records')[0]
        subset_linear_LGD = {key: encoded_data_linear_LGD[key] for key in (Columns_to_be_scaled)}
        subset_linear_LGD = pd.DataFrame(subset_linear_LGD,index=[0])
        subset_linear_LGD = scaler_linear_LGD.transform(subset_linear_LGD)
        scaled_values_linear_LGD = {}
        for scaled_features_linear_LGD in range(len(Columns_to_be_scaled)):
            scaled_values_linear_LGD[Columns_to_be_scaled[scaled_features_linear_LGD]]=subset_linear_LGD[0][scaled_features_linear_LGD]    
        encoded_data_linear_LGD.update(scaled_values_linear_LGD)
        encoded_data_linear_LGD = pd.DataFrame(encoded_data_linear_LGD,index=[0])
        results_linear_LGD = encoded_data_linear_LGD.items()
        linear_prediction_LGD = linear_LGD.predict(encoded_data_linear_LGD)
        recovery_rate_LGD = logistic_prediction_LGD * linear_prediction_LGD
        recovery_rate_LGD = np.where(recovery_rate_LGD < 0, 0, recovery_rate_LGD)
        recovery_rate_LGD = np.where(recovery_rate_LGD > 1, 1, recovery_rate_LGD)
        LGD = 1 - recovery_rate_LGD[0]        
        predictions_LGD.append(LGD)
    ECL_df = pd.DataFrame({'Customer_ID': customer_id,'Loan_ID': Loan_id,'Region': region,'Portfolio': portfolio,'PD': predictions_PD, 'LGD': predictions_LGD,'EAD': predictions_EAD})
    ECL_df['ECL'] = ECL_df['PD'] * ECL_df['LGD'] * ECL_df['EAD']
    
    return ECL_df


if __name__ == '__main__':
    
    path = os.getcwd()
    parent = os.path.dirname(path)
    data_file = 'data.json'
    data_path = os.path.join(path,data_file)
    resulting_values = predict(os.path.join(parent, "encoders"), os.path.join(parent, "std_scaler"),
                          os.path.join(parent, "model"), data_path)
    
    json_output = resulting_values.to_json(orient='records')
    print('modelOutput: ',json_output)
    
    