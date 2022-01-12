import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go

def show_dashboard_page():
    yol = "container_background.html"
    f = open(yol, 'r')
    contents = f.read()
    f.close()
    contents = contents.replace('smth', 'Pneumonia Dashboard')
    st.markdown(contents, unsafe_allow_html=True)

    # st.title("Pneumonia Dashboard")
    st.subheader("1. Number of Pneumonia Deaths by Age")
    num_of_death_areachart()

    st.subheader("2. Pneumonia Death Rate by Age")
    death_rate_linechart()

    st.subheader("3. Choropleth Map of Global Pneumonia Death Rate")
    selected_year = st.slider("Year", min_value=1990, max_value=2017, step=1, key=1)
    death_rate_map(selected_year)

    st.subheader("4. Pneumonia Risk Factor in Children Under Age 5")
    col1, col2 = st.columns(2)
    with col1:
        selected_year2 = st.slider("Year", min_value=1990, max_value=2017, step=1, key=2)
    risk_factor_child(selected_year2,col2)

    st.subheader("5. Pneumonia Risk Factor in Elderly Above Age 60")
    col3, col4 = st.columns(2)
    with col3:
        selected_year3 = st.slider("Year", min_value=1990, max_value=2017, step=1, key=3)
    risk_factor_elderly(selected_year3,col4)

    # st.subheader("6. Percentage of one-year-olds vaccinated against pneumococcal conjugate (PCV3), 2017")

@st.cache
def load_data(selected_dataset):
    df_death = pd.read_csv("pneumonia-and-lower-respiratory-diseases-deaths.csv")
    df_death_rate = pd.read_csv("pneumonia-mortality-by-age.csv")
    df_death_rate_global = pd.read_csv("pneumonia-death-rates-age-global.csv")
    df_riskFactor_child = pd.read_csv("pneumonia-risk-factors-child.csv")
    df_riskFactor_elderly = pd.read_csv("pneumonia-risk-factors-elderly.csv")
    df_careseeking = pd.read_csv("pneumonia-careseeking.csv")

    if(selected_dataset=="number_of_deaths"):
        return df_death
    elif(selected_dataset=="death_rate_by_age"):
        return df_death_rate
    elif(selected_dataset=="death_rate_global"):
        return df_death_rate_global
    elif(selected_dataset=='risk_factor_child'):
        return df_riskFactor_child
    elif(selected_dataset=='risk_factor_elderly'):
        return df_riskFactor_elderly
    else:
        print("Dataset not selected.")
        return df_riskFactor_elderly ############

def filter_year(df, year):
    df_year = df[df["Year"] == year]
    return df_year

def filter_country(df, country):
    df_country = df[df['Entity'] == country]
    df_country.drop(df_country.columns[[0, 1, 2]], axis=1, inplace=True)
    df_country.reset_index(drop=True)
    return df_country

def num_of_death_areachart():
    # Load dataset
    df_death = load_data('number_of_deaths')
    # Select country and filter with selected country
    countries = sorted(df_death['Entity'].unique())
    selected_country = st.selectbox('Country', countries, key=3)
    df_death = df_death[df_death['Entity'] == selected_country]
    # Rename columns and drop index
    df_death.rename(columns={'Deaths - Lower respiratory infections - Sex: Both - Age: Under 5 (Number)': 'Age under 5',
                             'Deaths - Lower respiratory infections - Sex: Both - Age: 50-69 years (Number)': 'Age 50-69',
                             'Deaths - Lower respiratory infections - Sex: Both - Age: 15-49 years (Number)': 'Age 15-49',
                             'Deaths - Lower respiratory infections - Sex: Both - Age: 5-14 years (Number)': 'Age 5-14',
                             'Deaths - Lower respiratory infections - Sex: Both - Age: 70+ years (Number)': 'Age above 70'
                             },
                    inplace=True)
    df_death.reset_index(drop=True, inplace=True)
    # Plot area chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_death['Year'], y=df_death['Age under 5'],
        name='Age under 5',
        mode='lines',
        line=dict(width=0.5, color='orange'),
        stackgroup='one'))
    fig2.add_trace(go.Scatter(
        x=df_death['Year'], y=df_death['Age 5-14'],
        name='Age 5-14',
        mode='lines',
        line=dict(width=0.5, color='lightgreen'),
        stackgroup='one'))
    fig2.add_trace(go.Scatter(
        x=df_death['Year'], y=df_death['Age 15-49'],
        name='Age 15-49',
        mode='lines',
        line=dict(width=0.5, color='blue'),
        stackgroup='one'))
    fig2.add_trace(go.Scatter(
        x=df_death['Year'], y=df_death['Age 50-69'],
        name='Age 50-69',
        mode='lines',
        line=dict(width=0.5, color='yellow'),
        stackgroup='one'))
    fig2.add_trace(go.Scatter(
        x=df_death['Year'], y=df_death['Age above 70'],
        name='Age above 70',
        mode='lines',
        line=dict(width=0.5, color='darkred'),
        stackgroup='one'))
    # fig2.update_layout(
    #     title="Death from pneumonia, by age")
    fig2.update_xaxes(
        title_text='Year')
    fig2.update_yaxes(
        title_text="Count")
    st.plotly_chart(fig2)

def death_rate_linechart():
    # Load dataset
    df_death_rate = load_data('death_rate_by_age')
    # Select country and filter with selected country
    countries = sorted(df_death_rate['Entity'].unique())
    selected_country = st.selectbox('Country', countries, key=4)
    df_death_rate = df_death_rate[df_death_rate['Entity'] == selected_country]
    # Rename columns and drop index
    df_death_rate.rename(columns={'Deaths - Lower respiratory infections - Sex: Both - Age: Under 5 (Rate)': 'Age under 5',
                                  'Deaths - Lower respiratory infections - Sex: Both - Age: 50-69 years (Rate)': 'Age 50-69',
                                  'Deaths - Lower respiratory infections - Sex: Both - Age: 15-49 years (Rate)': 'Age 15-49',
                                  'Deaths - Lower respiratory infections - Sex: Both - Age: 5-14 years (Rate)': 'Age 5-14',
                                  'Deaths - Lower respiratory infections - Sex: Both - Age: 70+ years (Rate)': 'Age above 70'
                                 },
                         inplace=True)
    df_death_rate.reset_index(drop=True, inplace=True)
    # Plot line chart
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_death_rate['Year'], y=df_death_rate['Age under 5'],
        name='Age under 5',
        mode='lines+markers',
        line=dict(width=0.5, color='orange')))
    fig3.add_trace(go.Scatter(
        x=df_death_rate['Year'], y=df_death_rate['Age 5-14'],
        name='Age 5-14',
        mode='lines+markers',
        line=dict(width=0.5, color='pink')))
    fig3.add_trace(go.Scatter(
        x=df_death_rate['Year'], y=df_death_rate['Age 15-49'],
        name='Age 15-49',
        mode='lines+markers',
        line=dict(width=0.5, color='blue')))
    fig3.add_trace(go.Scatter(
        x=df_death_rate['Year'], y=df_death_rate['Age 50-69'],
        name='Age 50-69',
        mode='lines+markers',
        line=dict(width=0.5, color='darkgreen')))
    fig3.add_trace(go.Scatter(
        x=df_death_rate['Year'], y=df_death_rate['Age above 70'],
        name='Age above 70',
        mode='lines+markers',
        line=dict(width=0.5, color='darkred')))
    # fig3.update_layout(
    #     title="Pneumonia Death Rate by Age")
    fig3.update_xaxes(
        title_text='Year')
    fig3.update_yaxes(
        title_text="Count")
    st.plotly_chart(fig3)

def death_rate_map(selected_year):
    # Load data
    df_death_rate_global = load_data('death_rate_global')
    # Filter with selected year
    df_death_rate_global = filter_year(df_death_rate_global, selected_year)
    # Rename column and drop index
    df_death_rate_global.rename(columns={'Deaths - Lower respiratory infections - Sex: Both - Age: Age-standardized (Rate)': 'Death rate'},
                                inplace=True)
    df_death_rate_global.reset_index(drop=True, inplace=True)
    # Load geojson file
    geo_world = json.load(open("custom.geo.json", "r"))

    # Instanciating necessary lists
    found = []
    missing = []
    countries_geo = []

    # For simpler acces, setting "zone" as index in a temporary dataFrame
    tmp = df_death_rate_global.set_index('Entity')

    # Looping over the custom GeoJSON file
    for country in geo_world['features']:

        # Country name detection
        country_name = country['properties']['name']

        # Checking if that country is in the dataset
        if country_name in tmp.index:

            # Adding country to our "Matched/found" countries
            found.append(country_name)

            # Getting information from both GeoJSON file and dataFrame
            geometry = country['geometry']

            # Adding 'id' information for further match between map and data
            countries_geo.append({
                'type': 'Feature',
                'geometry': geometry,
                'id': country_name
            })

        # Else, adding the country to the missing countries
        else:
            missing.append(country_name)

    # Displaying metrics
    print(f'Countries found    : {len(found)}')
    print(f'Countries not found: {len(missing)}')
    geo_world_ok = {'type': 'FeatureCollection', 'features': countries_geo}

    # Create the log count column
    df_death_rate_global['rate_color'] = df_death_rate_global['Death rate'].apply(np.log10)

    # Get the maximum value to cap displayed values
    max_log = df_death_rate_global['rate_color'].max()
    max_val = int(max_log) + 1

    # Prepare the range of the colorbar
    values = [i for i in range(max_val)]
    ticks = [10 ** i for i in values]

    # Create figure
    fig4 = px.choropleth_mapbox(
        df_death_rate_global,
        geojson=geo_world_ok,
        locations='Entity',
        color=df_death_rate_global['rate_color'],
        color_continuous_scale='YlOrRd',
        range_color=(0, df_death_rate_global['rate_color'].max()),
        hover_name='Entity',
        hover_data={'rate_color': False, 'Entity': False, 'Death rate': True},
        mapbox_style='open-street-map',
        zoom=0,
        center={'lat': 19, 'lon': 11},
        opacity=0.8,
        title="Pneumonia Death Rate of All Ages"
    )

    # Define layout specificities
    fig4.update_layout(
        margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
        coloraxis_colorbar={
            # 'title': 'Pneumonia Death Rate',
            'tickvals': values,
            'ticktext': ticks
        }
    )
    st.plotly_chart(fig4)

def risk_factor_child(selected_year,col):
    df_riskFactor_child = load_data('risk_factor_child')
    df_riskFactor_child = filter_year(df_riskFactor_child, selected_year)
    # show_data_button(df_riskFactor_child,1)
    countries = sorted(df_riskFactor_child['Entity'])
    with col:
        selected_country = st.selectbox('Country', countries, key=1)
    df_riskFactor_child = filter_country(df_riskFactor_child,selected_country)
    # show_data_button(df_riskFactor_child,2)
    count = np.array(df_riskFactor_child.iloc[0])
    count = np.round(count)
    count = count.astype(int)
    riskFactors = ['Outdoor air pollution',
                   'Child underweight',
                   'Indoor air pollution from solid fuels',
                   'No access to handwashing facility',
                   'Secondhand smoke',
                   'Vitamin A deficiency',
                   'Zinc deficiency',
                   'Short gestation for birth weight',
                   'Non-exclusive breastfeeding',
                   'Low birth weight for gestation',
                   'Child wasting',
                   'Child stunting']
    new_df = create_new_df(riskFactors, count)
    # title = 'Child deaths from pneumonia by risk factor, '+selected_country+', '+str(selected_year)
    plotBar(new_df,'count','riskFactors') #,title)

def risk_factor_elderly(selected_year, col):
    df_riskFactor_elderly = load_data('risk_factor_elderly')
    df_riskFactor_elderly = filter_year(df_riskFactor_elderly, selected_year)
    # show_data_button(df_riskFactor_elderly,2)
    countries = sorted(df_riskFactor_elderly['Entity'])
    with col:
        selected_country = st.selectbox('Country', countries, key=2)
    df_riskFactor_elderly = filter_country(df_riskFactor_elderly,selected_country)
    # st.dataframe(df_riskFactor_elderly)
    count = np.array(df_riskFactor_elderly.iloc[0])
    count = np.round(count)
    count = count.astype(int)
    riskFactors = ['No access to handwashing facility',
                   'Secondhand smoke',
                   'Smoking',
                   'Outdoor air pollution']
    new_df = create_new_df(riskFactors, count)
    # title = 'Elderly people deaths from pneumonia by risk factor, '+selected_country+', '+str(selected_year)
    plotBar(new_df,'count','riskFactors') #,title)

def show_data_button(df, key):
    show_data = st.button('Show raw data', key=key)
    if (show_data):
        st.dataframe(df)

def create_new_df(riskFactors, count):
    data_arr = []
    j = 0
    for i in riskFactors:
        data_arr.append([i, count[j]])
        j += 1
    new_df = pd.DataFrame(data=data_arr,
                          columns=['riskFactors', 'count'])
    return new_df

def plotBar(df, x, y):  #, title
    fig = px.bar(df, x=x, y=y, orientation='h',
                 #title=title,
                 labels={'count': 'Count',
                         'reasons': 'Reasons'}, text='count')
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig)