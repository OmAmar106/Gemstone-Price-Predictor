import joblib
import pandas as pd
import numpy as np

ct = joblib.load("column_transformer.pkl")
model = joblib.load("ridge_best.pkl")

def predict(carat,cut,clarity,table,x,y,z):
    df = pd.DataFrame({'carat':[carat],'table':[table],'x':[x],'y':[y],'z':[z],'Volume':[x*y*z],'cut':[cut],'clarity':[clarity]})

    X = ct.transform(df)

    return model.predict(X)[0]

# print(predict(carat=2.03,cut=np.nan,clarity='SI2',table=58.0,x=8.06,y=8.12,z=5.05))
