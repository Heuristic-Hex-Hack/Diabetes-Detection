import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Sidebar Configuration
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#99ffcc,#99ffcc);
    color: purple;
}
</style>
""",
    unsafe_allow_html=True,
)

# create a title and a sub-title
st.write("""
# Diabetes Detection
""")

image=Image.open('diabetes.jpg')
st.image(image,caption='Machine Learning Project by Hack Inversion',use_column_width=True)

st.subheader("Gender:M/F/other")
text1=st.text_input("Enter your Gender: M/F/Others")
#if not option1:
if not text1:
    st.warning('Please enter your Gender')
    st.stop()
if text1=='M' or text1=='F' or text1=='Others':

    df=pd.read_csv('diabetes.csv')

    st.subheader('Dataset Used: ')
    
    # show data as table
    if text1=='F':
        st.dataframe(df)
        print('')
        st.write(df.describe())

        # visualize data
        st.header('Display Graphs')

        radio2=st.radio('',('Bar Chart','Line Chart'))


        if radio2=='Bar Chart':
                
            st.bar_chart(df)
        else:
            
            st.line_chart(df)

        check3=st.checkbox("Show Area Chart")
        if check3:
            st.area_chart(df,height=900,width=1300,use_container_width=True)


        # split data
    
        X=df.iloc[:,0:8].values
        Y=df.iloc[:,-1].values

        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

        # get feature input from the user
        def get_user_input():

            st.sidebar.title('USER INPUTS')
            pregnancies=st.sidebar.slider('Pregnancies',0,15,2) #range0-15 and default is 2
            glucose=st.sidebar.slider('Glucose',0,200,110)
            blood_pressure=st.sidebar.slider('Blood Pressure',0,100,70)
            skin_thickness=st.sidebar.slider('Skin Thickness',0,80,30)
            insulin=st.sidebar.slider('Insulin',0.0,700.0,220.0)
            BMI=st.sidebar.slider('BMI',0.0,60.0,30.0)
            DFF=st.sidebar.slider('DFF',0.0,2.0,0.08)
            age=st.sidebar.slider('Age',10,90,35)

            # store a dictionary into a variable
            user_data={'Pregnancies':pregnancies,'Glucose':glucose,'Blood Pressure':blood_pressure,'Skin Thickness':skin_thickness,'Insulin':insulin,'BMI':BMI,'DFF':DFF,'Age':age}

            # transform data into data frame
            features=pd.DataFrame(user_data,index=[0])
            return features


        # store user input into  a variable
        user_input=get_user_input()

        # set subheader and display user input
        st.subheader('User Input: ')

        st.write(user_input)

        # create and train model
        RandomForestClassifier=RandomForestClassifier()

        RandomForestClassifier.fit(X_train,Y_train)

        # show the model metrics
        st.subheader('Model Test Accuracy score: ')
        st.write(str(metrics.accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

        st.subheader('Mean absolute Error: ')
        st.write(str(metrics.mean_absolute_error(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

        #st.subheader('Squared Error:')
        #st.write(str(metrics.mean_squared_error(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

        st.subheader('R2-score:')
        st.write(str(metrics.r2_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')
        
        st.subheader('Confusion Matrix:')
        st.write(metrics.confusion_matrix(Y_test,RandomForestClassifier.predict(X_test)))

        # store the model predictions in a variable
        prediction=RandomForestClassifier.predict(user_input)

        # set a  subheader and display the classifications
        st.subheader('For diabetic person output is 1 else 0')
        if st.button('Show Prediction'):
            st.subheader('Classification: ')
            st.write(prediction)

            if prediction==0:
                st.success('You are Healthy :) ')
            if prediction==1:
                st.warning('You are Diabetic :( ')
        
    if text1=="M" or text1=="Others":
        df.drop(columns=['Pregnancies'],axis=1,inplace=True)
        st.dataframe(df)
        print('')
        st.write(df.describe())

        # visualize data
        st.header('Display Graphs')

        radio2=st.radio('',('Bar Chart','Line Chart'))


        if radio2=='Bar Chart':
                
            st.bar_chart(df)
        else:
            
            st.line_chart(df)

        check3=st.checkbox("Show Area Chart")
        if check3:
            st.area_chart(df,height=900,width=1300,use_container_width=True)


        # split data
    
        X=df.iloc[:,0:7].values
        Y=df.iloc[:,-1].values

        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

        # get feature input from the user
        def get_user_input():

            st.sidebar.title('USER INPUTS')
            #pregnancies=st.sidebar.slider('Pregnancies',0,15,2) #range0-15 and default is 2
            glucose=st.sidebar.slider('Glucose',0,200,110)
            blood_pressure=st.sidebar.slider('Blood Pressure',0,100,70)
            skin_thickness=st.sidebar.slider('Skin Thickness',0,80,30)
            insulin=st.sidebar.slider('Insulin',0.0,700.0,220.0)
            BMI=st.sidebar.slider('BMI',0.0,60.0,30.0)
            DFF=st.sidebar.slider('DFF',0.0,2.0,0.08)
            age=st.sidebar.slider('Age',10,90,35)

            # store a dictionary into a variable
            user_data={'Glucose':glucose,'Blood Pressure':blood_pressure,'Skin Thickness':skin_thickness,'Insulin':insulin,'BMI':BMI,'DFF':DFF,'Age':age}

            # transform data into data frame
            features=pd.DataFrame(user_data,index=[0])
            return features


        # store user input into  a variable
        user_input=get_user_input()

        # set subheader and display user input
        st.subheader('User Input: ')

        st.write(user_input)

        # create and train model
        RandomForestClassifier=RandomForestClassifier()

        RandomForestClassifier.fit(X_train,Y_train)

        # show the model metrics
        st.subheader('Model Test Accuracy score: ')
        st.write(str(metrics.accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

        st.subheader('Mean absolute Error: ')
        st.write(str(metrics.mean_absolute_error(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

        #st.subheader('Squared Error:')
        #st.write(str(metrics.mean_squared_error(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

        st.subheader('R2-score:')
        st.write(str(metrics.r2_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')
        
        st.subheader('Confusion Matrix:')
        st.write(metrics.confusion_matrix(Y_test,RandomForestClassifier.predict(X_test)))

        # store the model predictions in a variable
        prediction=RandomForestClassifier.predict(user_input)

        # set a  subheader and display the classifications
        st.subheader('For diabetic person output is 1 else 0')
        if st.button('Show Prediction'):
            st.subheader('Classification: ')
            st.write(prediction)

            if prediction==0:
                st.success('You are Healthy :) ')
            if prediction==1:
                st.warning('You are Diabetic :( ')

