import matplotlib.pyplot as plt 
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from evaluate import load

metric_1 = load("mse")
metric_2 = load("r_squared")
metric_3 = load("mae")

def compute_metrics(p,y):
    predictions, labels = p,y
    m1 = metric_1.compute(predictions=predictions, references=labels)
    m2 = metric_2.compute(predictions=predictions, references=labels)
    m3 = metric_3.compute(predictions=predictions, references=labels)
    return {"mse":m1["mse"], "r_quared":m2, "mae":m3["mae"]}

df_feat=pd.read_csv('./features/All_kfolds_feat_all_data.csv')
df_feat=df_feat.dropna()
df_feat = df_feat.loc[~df_feat.index.duplicated(keep='first')]
df_feat

folds_name = ["K_fold_1", "K_fold_2","K_fold_3","K_fold_4","K_fold_5"]

results= []

Prediction = []
GT = []
for k_n in folds_name:
    print(k_n)
    Training_data = df_feat[df_feat['Fold_name'] == k_n][ df_feat['Split'] != 'Test']
    Testing_data = df_feat[df_feat['Fold_name'] == k_n][ df_feat['Split'] == 'Test']
    
    t = Training_data.loc[:, df_feat.columns != 'labels']
    t1 = t.loc[:,t.columns != 'Fold_name' ] 
    X = t1.loc[:,t1.columns != 'Split' ]
    X = X.loc[:, ~X.columns.str.contains('^Unnamed')]
    y_train = Training_data['labels'].tolist()
    
    t = Testing_data.loc[:, df_feat.columns != 'labels']
    t1 = t.loc[:,t.columns != 'Fold_name' ] 
    X_test = t1.loc[:,t1.columns != 'Split' ]
    X_test = X_test.loc[:, ~X_test.columns.str.contains('^Unnamed')]
    y_test = Testing_data['labels'].tolist()
    
    regr = RandomForestRegressor(max_depth=8, random_state=42)
    regr.fit(X, y_train)
    r = regr.predict(X_test)
    joblib.dump(regr, './weights/random_forest_'+k_n+'.joblib')
    print(compute_metrics(r,y_test))
    results.append(compute_metrics(r,y_test))
    print()
    
print("mse ",(results[0]["mse"]+results[1]["mse"]+results[2]["mse"]+results[3]["mse"]+results[4]["mse"])/5)
print("r_quared ",(results[0]["r_quared"]+results[1]["r_quared"]+results[2]["r_quared"]+results[3]["r_quared"]+results[4]["r_quared"])/5)
print("mae ",(results[0]["mae"]+results[1]["mae"]+results[2]["mae"]+results[3]["mae"]+results[4]["mae"])/5)