import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import streamlit as st 
import joblib
from sklearn.preprocessing import LabelEncoder

# load the data 

df = pd.read_csv(r'C:\Users\Fathy mohammed\Desktop\finall project\Healthcare\ECG_to_model_data.csv')

df =  df[['heart_rate', 'Pwave', 'PQseg', 'QRScomplex', 'QRseg',
       'QTinterval', 'RSseg', 'STseg', 'Twave', 'TPseg', 'ECGseg',
       'R-Rinterval', 'P-Pinterval', 'PQinterval',  'PRinterval','QRinterval','Type_of_disease']]

# Data Segmentation 
x = df[['heart_rate', 'Pwave', 'PQseg', 'QRScomplex', 'QRseg',
       'QTinterval', 'RSseg', 'STseg', 'Twave', 'TPseg', 'ECGseg',
       'R-Rinterval', 'P-Pinterval', 'PQinterval',  'PRinterval','QRinterval']]

y = df[['Type_of_disease']]

X_train ,X_test ,y_train ,y_test = train_test_split(x,y ,test_size=0.2 ,random_state=42)

# Model training
tree_model = DecisionTreeClassifier(criterion = 'entropy',
    max_depth= 20, 
    max_features= 'sqrt',
    min_samples_leaf= 1,
    min_samples_split= 2, 
    splitter='best')
tree_model.fit(X_train, y_train)

# Save the model 
model_path ="ECG_model.pkl"
joblib.dump(tree_model, model_path)

# Streamlit interface

st.title('ECG Prediction APP')
st.image("https://www.researchgate.net/profile/Gionata-Fragomeni/publication/366038962/figure/fig1/AS:11431281104981492@1670333175837/ECG-signal-Adapted-from-29.png", caption="ECG signal", use_column_width=True)

user_input = {}
for col in X_train.columns:
    user_input[col] = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()))


# load the model
tree_model = joblib.load(model_path)

if st.button("Start Prediction"):
        input_data = pd.DataFrame([user_input])
        input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
        # prediction
        prediction =tree_model.predict(input_data)

        doc ={
              'NSR' : 'Normal Sinus Rhythm' ,
              'AFF': 'Atrial Fibrillation' ,
              'ARR': 'Arrhythmia',
              'CHF': 'Congestive Heart Failure'

        }
        
        if isinstance(prediction, np.ndarray):
             prediction = prediction.item()  # تحويل المصفوفة إلى قيمة مفردة

        if prediction in doc:
            if prediction == 'NSR':
                st.success(f"{prediction}:{doc[prediction]}")
            else:
                st.error(f"{prediction}: {doc[prediction]}")
        else:
            st.warning("Unknown Prediction")