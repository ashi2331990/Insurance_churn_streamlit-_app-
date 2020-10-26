import streamlit as st 
import numpy as np 
import pandas as pd 
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
#---------------------------------------------------------------------------------------------------------------------------------
df=pd.read_csv(r"Train.csv")
dffeature=df.copy()
dff=dffeature.drop(['feature_7','feature_10'],axis=1)
from sklearn.model_selection import train_test_split
X=dff.drop("labels",axis=1)
y=dff.labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model2=AdaBoostClassifier(learning_rate=0.5, n_estimators=200)
model2.fit(X_train,y_train) 
#--------------------------------------------------------------------------------------------------------------------------------------------------        
def predict_forest(feature_0,feature_1,feature_2,feature_3,feature_4,feature_5,
                    feature_6,feature_8,feature_9,feature_11,feature_12, feature_13, 
                    feature_14, feature_15):
    input=np.array([[feature_0,feature_1,feature_2,feature_3,feature_4,feature_5,
                        feature_6,feature_8,feature_9,feature_11,feature_12, feature_13, 
                        feature_14, feature_15]]).astype(np.float64)
    a=pd.DataFrame({"feature_0":float(feature_0),"feature_1":float(feature_1),"feature_2":float(feature_2),"feature_3":float(feature_3),
                    "feature_4":float(feature_4),"feature_5":float(feature_5),"feature_6":float(feature_6),"feature_8":float(feature_8),
                    "feature_9":float(feature_9),"feature_11":float(feature_11),"feature_12":float(feature_12),"feature_13":float(feature_13),
                    "feature_14":float(feature_14),"feature_15":float(feature_15)},index=[1])
    prediction=model2.predict(a)
    return (prediction)
#---------------------------------------------------------------------------------------------------------------------------------------------------
def main():

    st.title("Hackathon Project")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Insurance Customer Churn Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")
    st.header("Enter the Features in the side bar")

    feature_0 = st.sidebar.text_input("feature_0",key=0)
    feature_1 = st.sidebar.text_input("feature_1",key=1)
    feature_2 = st.sidebar.text_input("feature_2",key=2)
    feature_3 = st.sidebar.text_input("feature_3",key=3)
    feature_4 = st.sidebar.text_input("feature_4",key=4)
    feature_5 = st.sidebar.text_input("feature_5",key=5)
    feature_6 = st.sidebar.text_input("feature_6",key=6)
    feature_8 = st.sidebar.text_input("feature_8",key=7)
    feature_9 = st.sidebar.text_input("feature_9",key=8)
    feature_11 = st.sidebar.text_input("feature_11",key=9)
    feature_12 = st.sidebar.text_input("feature_12",key=10)
    feature_13 = st.sidebar.text_input("feature_13",key=11)
    feature_14 = st.sidebar.text_input("feature_14",key=12)
    feature_15 = st.sidebar.text_input("feature_15",key=13)
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Consumer will stay </h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Consumer will churn </h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_forest(feature_0,feature_1,feature_2,feature_3,feature_4,feature_5,
                                 feature_6,feature_8,feature_9,feature_11,feature_12, feature_13, 
                                feature_14, feature_15)
        st.success('The consumer status will be  {}'.format(output))

        if output == 1:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)
if __name__=='__main__':
    main()