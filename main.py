import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
matplotlib.use("Agg")

# Helper function to download data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

def main():
    st.title("Data Science Project with Streamlit")

    activities = ['EDA', 'PLOTS', 'DATA CLEANING', 'ML', 'DOWNLOAD']
    choices = st.sidebar.selectbox("Select Activity", activities)
    
    data = st.file_uploader("Upload a dataset", type=["csv", "txt", "xlsx"])
    
    if data is not None:
        if data.name.endswith('.csv'):
            df = pd.read_csv(data)
        elif data.name.endswith('.txt'):
            df = pd.read_csv(data, delimiter='\t')
        else:
            df = pd.read_excel(data)
        st.dataframe(df.head())
        
        if choices == "EDA":
            st.subheader("Exploratory Data Analysis")
            if st.checkbox("Show Shape"):
                st.write(df.shape)
                
            if st.checkbox("Show Describe"):
                st.write(df.describe())
                
            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)
                
            if st.checkbox("Show Selected Column"):
                selected_columns = st.multiselect("Select Columns", all_columns)
                if selected_columns:
                    new_df = df[selected_columns]
                    st.dataframe(new_df)
                    
            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:, -1].value_counts())
                
            if st.checkbox("Correlation Matrix"):
                plt.figure(figsize=(10, 6))
                sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
                st.pyplot()
                
            if st.checkbox("Scatter Plot"):
                selected_x = st.selectbox("Select X-Axis", all_columns)
                selected_y = st.selectbox("Select Y-Axis", all_columns)
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=df[selected_x], y=df[selected_y])
                st.pyplot()
        
        elif choices == 'PLOTS':
            st.subheader("Data Visualization")
            if st.checkbox("Show Value Counts"):
                value_counts_plot = df.iloc[:, -1].value_counts().plot(kind="bar")
                st.pyplot(value_counts_plot.figure)
                
            all_columns = df.columns.to_list()
            type_of_plot = st.selectbox("Select plot type", ["area", "bar", "line", "hist", "box", "kde", "pairplot", "scatter_matrix"])
            selected_columns = st.multiselect("Select columns to plot", all_columns)
            
            if st.button("Generate Plot"):
                if selected_columns:
                    new_extracted_df = df[selected_columns]
                    
                    if type_of_plot == 'area':
                        st.area_chart(new_extracted_df)
                    elif type_of_plot == 'bar':
                        st.bar_chart(new_extracted_df)
                    elif type_of_plot == 'line':
                        st.line_chart(new_extracted_df)
                    elif type_of_plot == 'pairplot':
                        st.write(sns.pairplot(df[selected_columns]))
                        st.pyplot()
                    elif type_of_plot == 'scatter_matrix':
                        fig = px.scatter_matrix(df[selected_columns])
                        st.write(fig)
                    else:
                        plot = new_extracted_df.plot(kind=type_of_plot)
                        st.pyplot(plot.figure)
        
        elif choices == 'DATA CLEANING':
            st.subheader("Data Cleaning")
            if st.checkbox("Handle Missing Values"):
                all_columns = df.columns.to_list()
                selected_columns = st.multiselect("Select columns to handle missing values", all_columns)
                if st.button("Fill Missing Values"):
                    df[selected_columns] = df[selected_columns].fillna(df[selected_columns].mean())
                    st.write("Missing values filled with mean")
                    st.dataframe(df.head())
                    
            if st.checkbox("Drop Columns"):
                all_columns = df.columns.to_list()
                selected_columns = st.multiselect("Select columns to drop", all_columns)
                if st.button("Drop Selected Columns"):
                    df.drop(selected_columns, axis=1, inplace=True)
                    st.write("Selected columns dropped")
                    st.dataframe(df.head())
                    
        elif choices == 'ML':
            st.subheader("Machine Learning")
            target = st.selectbox("Select Target Variable", df.columns.to_list())
            features = st.multiselect("Select Feature Variables", [col for col in df.columns if col != target])
            
            if st.button("Train Model"):
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                st.write("Classification Report:")
                st.text(classification_report(y_test, predictions))
                
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(y_test, predictions))
                
        elif choices == 'DOWNLOAD':
            st.subheader("Download Processed Data")
            if st.button("Download CSV"):
                csv = convert_df_to_csv(df)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='processed_data.csv',
                    mime='text/csv',
                )
                
if __name__ == "__main__":
    main()
