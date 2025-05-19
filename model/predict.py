import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ct = joblib.load(os.path.join(BASE_DIR, "column_transformer.pkl"))
sc = joblib.load(os.path.join(BASE_DIR, "Standard_Scaler.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "ridge_best.pkl"))


def predict(carat,cut,clarity,table,x,y,z):
    df = pd.DataFrame({'carat':[carat],'table':[table],'x':[x],'y':[y],'z':[z],'Volume':[x*y*z],'cut':[cut],'clarity':[clarity],'id':[0],'depth':[0],'color':['F']})

    X = ct.transform(df)
    X.drop(['remainder__color'],axis=1,inplace=True)

    X = sc.transform(X)

    return model.predict(X)[0]

# print(predict(carat=2.03,cut=np.nan,clarity='SI2',table=58.0,x=8.06,y=8.12,z=5.05))
