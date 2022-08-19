import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from ngboost import NGBRegressor
import plotly.graph_objects as go
from feature_Engineering import LowVarianceFeaturesRemover, ScalePerEngine
from feature_Engineering import (RollTimeSeries, TSFreshFeaturesExtractor, CustomPCA, TSFreshFeaturesSelector, tsfresh_calc)
import plotly.express as px
from target_metrics_baseline import calculate_RUL
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Prediction of RUL of Engines
This predicts the **Engines' Remaining Useful Life**!
""")
st.write('---')

st.subheader('**Description of Dataset**')
st.write('**Unit** - Unit Number')
st.write('**Time** - Time in cycles')
st.write('**OperationalSetting1** - operational setting 1')
st.write('**OperationalSetting1** - operational setting 2')
st.write('**OperationalSetting1** - operational setting 3')
st.write('**Sensor-1Measurement** -  Fan inlet temperature')
st.write('**Sensor-2Measurement** -  LPC outlet temperature')
st.write('**Sensor-3Measurement** -  HPC outlet temperature')
st.write('**Sensor-4Measurement** -  LPT outlet temperature')
st.write('**Sensor-5Measurement** -  Fan inlet Pressure')
st.write('**Sensor-6Measurement** -  bypass-duct pressure')
st.write('**Sensor-7Measurement** -  HPC outlet pressure')
st.write('**Sensor-8Measurement** -  Physical fan speed')
st.write('**Sensor-9Measurement** -  Physical core speed')
st.write('**Sensor-10Measurement** -  Engine pressure ratio(P50/P2')
st.write('**Sensor-11Measurement** -  HPC outlet Static pressure')
st.write('**Sensor-12Measurement** -  Ratio of fuel flow to Ps30')
st.write('**Sensor-13Measurement** -  Corrected fan speed')
st.write('**Sensor-14Measurement** -  Corrected core speed')
st.write('**Sensor-15Measurement** -  Bypass Ratio')
st.write('**Sensor-16Measurement** -  Burner fuel-air ratio')
st.write('**Sensor-17Measurement** -  Bleed Enthalpy')
st.write('**Sensor-18Measurement** -  Required fan speed')
st.write('**Sensor-19Measurement** -  Required fan conversion speed')
st.write('**Sensor-20Measurement** -  High-pressure turbines Cool air flow')
st.write('**Sensor-21Measurement** -  Low-pressure turbines Cool air flow')

train = pd.read_csv('train_FD001.csv')
test = pd.read_csv('test_FD001.csv')
#train['rul'] = calculate_RUL(train)
train_df = train
train_df.rename(columns={
        'Sensor-1Measurement':'Sensor_1Measurement',
        'Sensor-2Measurement':'Sensor_2Measurement',
        'Sensor-3Measurement':'Sensor_3Measurement',
        'Sensor-4Measurement':'Sensor_4Measurement',
        'Sensor-5Measurement':'Sensor_5Measurement',
        'Sensor-6Measurement':'Sensor_6Measurement',
        'Sensor-7Measurement':'Sensor_7Measurement',
        'Sensor-8Measurement':'Sensor_8Measurement',
        'Sensor-9Measurement':'Sensor_9Measurement',
        'Sensor-10Measurement':'Sensor_10Measurement',
        'Sensor-11Measurement':'Sensor_11Measurement',
        'Sensor-12Measurement':'Sensor_12Measurement',
        'Sensor-13Measurement':'Sensor_13Measurement',
        'Sensor-14Measurement':'Sensor_14Measurement',
        'Sensor-15Measurement':'Sensor_15Measurement',
        'Sensor-16Measurement':'Sensor_16Measurement',
        'Sensor-17Measurement':'Sensor_17Measurement',
        'Sensor-18Measurement':'Sensor_18Measurement',
        'Sensor-19Measurement':'Sensor_19Measurement',
        'Sensor-20Measurement':'Sensor_20Measurement',
        'Sensor-21Measurement':'Sensor_21Measurement',
        }, inplace = True)
st.write(train_df)

df_stat = train.groupby('Unit')['Time'].max().describe()
st.write('#### Number of engines - ', df_stat['count'])
st.write('#### Minimum life cycle - ', df_stat['min'])
st.write('#### Average life cycle - ', df_stat['mean'])
st.write('#### Maximum life cycle - ', df_stat['max'])
st.write('---')

st.header('Choose File')
uploaded_file = st.file_uploader('')
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    df_stat_test = dataframe.groupby('Unit')['Time'].max().describe()
    c = df_stat_test['count']
    test_long_h_fts = pickle.load(open('test_long_h_ftrs_1.pkl','rb'))
    test_short_h_fts = pickle.load(open('test_short_h_ftrs_1.pkl','rb'))
    test_ftrs = test_long_h_fts.merge(test_short_h_fts, how='inner', right_index=True, left_index=True)
    test_ftrs.index = test_ftrs.index.set_names(['Unit', 'Time'])

    X_test = test_ftrs.reset_index().drop(columns=['Unit'])
    test_units = test_ftrs.index.to_frame(index=False)

    model = pickle.load(open('model.pkl','rb'))
    engines = test_units['Unit']
    X_test_last = X_test.groupby(engines, as_index=False).last()
    pred_rul = model.predict(X_test_last)
    pred = pred_rul.transpose()
    st.header('Remaining Usable Life of given Engines - ')
    st.write('#### Number of engines - ', c)
    color1 = 'lightgreen'
    color2 = 'lightblue'
    no_eng = []
    for i in range(1,int(c+1)):
        no_eng.append(i)
    fig = go.Figure(data=[go.Table(
    # Ratio for column width
        columnwidth=[1, 5],

        header=dict(values=['Engine', 'Predicted RUL']),
        cells=dict(values=[no_eng,
                          pred_rul.round()],
                   fill_color=[[color1, color2]*(int(c/2))],))
    ])
    fig.update_layout(
        autosize=False,
        width=700,
        height=400,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        ),
        paper_bgcolor="LightSteelBlue",)
    fig.update_layout(font_size=14)
    fig.update_layout(font_family='sans-serif')
    fig.update_layout(font_color="#233")
    st.write(fig)
    st.write('---')

st.write('### Visualization on train set')
chart_select = st.sidebar.selectbox(
    label = "Type of chart",
    options=['Scatterplots','Lineplots']
)

numeric_columns = list(train_df.select_dtypes(['float','int']).columns)
sensor_names = ['Sensor_{}Measurement'.format(i) for i in range(1,22)] 

if chart_select == 'Scatterplots':
    st.sidebar.subheader('Scatterplot Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=['Time'])
        y_values = st.sidebar.selectbox('Y axis',options=sensor_names)
        plot = px.scatter(data_frame=train_df,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Lineplots':
    st.sidebar.subheader('Lineplots Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=['Unit'])
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        plot = px.line(train_df,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
st.write('---')

result = pickle.load(open('result.pkl','rb'))
result['HI'] = result.groupby('Unit').RUL.transform(lambda x: minmax_scale(x))
st.subheader('Health Indicator of first 30 engines in train set')
hi = px.line(result[result.Unit<31],x='Time',y="HI", color='Unit')
st.write(hi)
st.write('---')
        

    




