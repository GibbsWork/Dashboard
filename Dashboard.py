import json
from math import comb
import requests
import pandas as pd
import streamlit as st
import numpy as np
from twelvedata import TDClient
import plotly.express as px
import requests
import json
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
from dateutil.relativedelta import relativedelta
from datetime import timedelta
st.set_page_config(layout="wide")





apikey=st.secrets["APIKEY"]
td = TDClient(apikey=apikey)
fmpAPI = st.secrets["FMPAPI"]

from alpaca_trade_api.rest import REST, TimeFrame

base = 'https://data.alpaca.markets/v2'
ALPACA_API_KEY = st.secrets["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
BASE_URL = 'https://data.alpaca.markets/v2'
BASE_URLClock = 'https://paper-api.alpaca.markets'

from alpaca_trade_api.rest import REST, TimeFrame
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)

today = (datetime.datetime.now((datetime.timezone.utc)) - relativedelta(minutes=16)).isoformat()

maxRating = 21



def chart_data(asset):

    symbol = asset

    URL = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?limit=500&apikey={fmpAPI}"


    data = requests.get(URL)
    data = data.json()


    lSymbol = []
    lDate = []
    lEps =[]
    lEbitda =[]

    for value in data:

        date = value['date']
        eps = value['eps']
        ebitda = value['ebitda']

        lDate.append(date)
        lEps.append(eps)
        lEbitda.append(ebitda)


    df = pd.DataFrame()


    df['date'] = lDate 
    df['eps'] = lEps
    df['ebitda'] = lEbitda 
    df = df.set_index('date')
    print(df)

    return df

def dcf(asset):
    URL = f"https://financialmodelingprep.com/api/v3/discounted-cash-flow/{asset}?limit=500&apikey={fmpAPI}"


    data = requests.get(URL)
    data = data.json()

    lDate = []
    lDcf =[]
    lPrice =[]

    for value in data:

        date = value['date']
        currentPrice = value['Stock Price']
        dcfPrice = value['dcf']

        lDate.append(date)
        lPrice.append(currentPrice)
        lDcf.append(dcfPrice)


    df = pd.DataFrame()


    df['date'] = lDate 
    df['DCF'] = lDcf
    df['Price'] = lPrice 
    df = df.set_index('date')

    print(df)
    return df

def sectorPerformance():
    URL = f"https://financialmodelingprep.com/api/v3/sector-performance?apikey={fmpAPI}"
    data = requests.get(URL)
    data = data.json()
    print(data)

    sectorl =[]
    changesPercentagel =[]


    for value in data:
        sector = value['sector']
        perChange = value['changesPercentage']  

        perChange = perChange.replace("%", "")
        perChange = pd.to_numeric(perChange)

        

        sectorl.append(sector)
        changesPercentagel.append(perChange)


    df = pd.DataFrame()


    df['Sector'] = sectorl 
    df['Movement (%)'] = changesPercentagel

    df = df.set_index('Sector')

    print(df)
    return df

def tickerGraph(asset):
    ts = td.time_series(
    symbol=asset,
    outputsize=500,
    interval="1day",
)

    tickerDf = ts.as_pandas()
    print(tickerDf)

    ts.as_plotly_figure()
    chart = ts.with_ema(time_period=200).with_ema(time_period=50).with_mom().with_macd().as_plotly_figure()
    return chart, tickerDf

def sectorgraph():
    df = sectorPerformance()
    fig = px.bar(df, x=df.index, y='Movement (%)')
    # fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=True)
    fig.update_layout(yaxis={'visible': True, 'showticklabels': True})
    sectorGraphObject = fig
    return sectorGraphObject

def gainerslosers():

    loserSymbolsl = []
    gainersSymbolsl = []
    losernamel = []
    gainernamel = []
    loserchangel=[]
    gainerchangel = []
    typeLoser = []
    typeGainer = []

    fmpAPI = 'aa005c9f1003c4b4d396cc1e7037272f'



    URL = f"https://financialmodelingprep.com/api/v3/stock_market/losers?apikey={fmpAPI}"
    URLGainer = f"https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey={fmpAPI}"
    data = requests.get(URL)
    data = data.json()

    dataGainer = requests.get(URLGainer)
    dataGainer = dataGainer.json()


    for value in data:
        symbol = value['symbol']
        name = value['name']
        changeLoser = np.absolute(value['changesPercentage'])

        loserSymbolsl.append(symbol)
        losernamel.append(name)
        loserchangel.append(changeLoser)
        typeLoser.append('Loser')

    for value in dataGainer:
        symbol = value['symbol']
        name = value['name']
        changeGainer = value['changesPercentage']

        gainersSymbolsl.append(symbol)
        gainernamel.append(name)
        gainerchangel.append(changeGainer)
        typeGainer.append('Gainer')

    dfLoser = pd.DataFrame()
    dfGainer = pd.DataFrame()

    dfLoser['Symbol'] = loserSymbolsl 
    dfLoser['Name'] = losernamel
    dfLoser['Type'] = typeLoser
    dfLoser['Change'] = loserchangel

    dfGainer['Symbol'] = gainersSymbolsl 
    dfGainer['Name'] = gainernamel
    dfGainer['Type'] = typeGainer
    dfGainer['Change'] = gainerchangel

    dfLoser = dfLoser.set_index('Symbol')
    dfGainer = dfGainer.set_index('Symbol')


    frames = [dfGainer, dfLoser]

    result = pd.concat(frames)
    result = result.sort_values('Name')

    fig = px.scatter(result, x='Name', y="Change", color="Type",
                hover_name="Name",size="Change")

    return dfLoser, dfGainer, result, fig

def incomeStatement(asset):
    URL = f'https://financialmodelingprep.com/api/v3/income-statement/{asset}?limit=240&apikey={fmpAPI}'
    data = requests.get(URL)
    data = data.json()

    datel= []
    revenuel = []
    grossProfitl = []
    netIncomel = []


    for value in data:
        date = value['date']
        revenue = value['revenue']
        grossProfit = value['grossProfit']
        netIncome = value['netIncome']

        datel.append(date)
        revenuel.append(revenue)
        grossProfitl.append(grossProfit)
        netIncomel.append(netIncome)

    df = pd.DataFrame()

    df['Date'] = datel
    df['Revenue'] = revenuel
    df['Gross profit'] = grossProfitl
    df['Net income'] = netIncomel

    df = df.set_index('Date')

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Revenue'],
                        mode='lines+markers',
                        name='Revenue'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Gross profit'],
                        mode='lines+markers',
                        name='Gross profit'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Net income'],
                        mode='lines+markers',
                        name='Net income'))


    return df, fig


def bgSentiment(asset1):

    URL1 = f'https://financialmodelingprep.com/api/v3/income-statement/{asset1}?limit=240&apikey={fmpAPI}'
    URL2 = f'https://financialmodelingprep.com/api/v3/rating/{asset1}?apikey={fmpAPI}'
    data = requests.get(URL1)
    data = data.json()

    scoreData = requests.get(URL2)
    scoreData = scoreData.json()


    datel= []
    revenuel = []
    grossProfitl = []
    netIncomel = []
    epsl = []
    finalScore = 0

    for value in scoreData:
        score = value['rating']

        if score == 'S+':
            finalScore = 5
        elif score =='S':
            finalScore = 4
        elif score =='S-':
            finalScore = 3
        elif score =='A+':
            finalScore = 2
        elif score =='A':
            finalScore = 1
        else:
            finalScore = 0







    for value in data:
        date = value['date']
        revenue = value['revenue']
        grossProfit = value['grossProfit']
        netIncome = value['netIncome']
        eps = value['eps']


        datel.append(date)
        revenuel.append(revenue)
        grossProfitl.append(grossProfit)
        netIncomel.append(netIncome)
        epsl.append(eps)

    df = pd.DataFrame()

    df['Date'] = datel
    df['Revenue'] = revenuel
    df['Gross profit'] = grossProfitl
    df['Net income'] = netIncomel
    df['EPS'] = epsl

    df = df.set_index('Date')


    incomeStatementDelta = df.iloc[::-1]
    incomeStatementDelta = incomeStatementDelta.apply(pd.to_numeric)
    incomeStatementDelta = incomeStatementDelta.pct_change()
    incomeStatementDelta = incomeStatementDelta.apply(lambda x: x*100)
    incomeStatementDelta = incomeStatementDelta.dropna()


    revenueLength = len(incomeStatementDelta.loc[incomeStatementDelta['Revenue'] > 0])
    gpLength = len(incomeStatementDelta.loc[incomeStatementDelta['Gross profit'] > 0])
    netICLength = len(incomeStatementDelta.loc[incomeStatementDelta['Net income'] > 0])
    epsLength = len(incomeStatementDelta.loc[incomeStatementDelta['EPS'] > 0])



    finalRating = (revenueLength + gpLength + netICLength + epsLength + finalScore)


    closePrice = finalRating
    highPrice = maxRating

    confidenceLevel = 0.8

    topRange = highPrice
    midRange = highPrice*confidenceLevel
    lowRange = 0

    color = 'crimson'

    if closePrice > midRange:
        color = 'forestgreen'
    if closePrice == midRange:
        color = 'darkslateblue'
    

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = closePrice,
        gauge={"axis": {"range": [lowRange, topRange]},
            'steps' : [
                    {'range': [lowRange, topRange], 'color': "lightgray"}],
            'bar': {'color': color},

            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': midRange
                }
            

            
            
            },
        title = {'text': "Investment confidence"},
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))

    return fig

def bgSentiment2(asset1):

    URL1 = f'https://financialmodelingprep.com/api/v3/income-statement/{asset1}?limit=240&apikey={fmpAPI}'
    URL2 = f'https://financialmodelingprep.com/api/v3/rating/{asset1}?apikey={fmpAPI}'
    data = requests.get(URL1)
    data = data.json()

    scoreData = requests.get(URL2)
    scoreData = scoreData.json()


    datel= []
    revenuel = []
    grossProfitl = []
    netIncomel = []
    epsl = []
    finalScore = 0

    for value in scoreData:
        score = value['rating']

        if score == 'S+':
            finalScore = 5
        elif score =='S':
            finalScore = 4
        elif score =='S-':
            finalScore = 3
        elif score =='A+':
            finalScore = 2
        elif score =='A':
            finalScore = 1
        else:
            finalScore = 0







    for value in data:
        date = value['date']
        revenue = value['revenue']
        grossProfit = value['grossProfit']
        netIncome = value['netIncome']
        eps = value['eps']


        datel.append(date)
        revenuel.append(revenue)
        grossProfitl.append(grossProfit)
        netIncomel.append(netIncome)
        epsl.append(eps)

    df = pd.DataFrame()

    df['Date'] = datel
    df['Revenue'] = revenuel
    df['Gross profit'] = grossProfitl
    df['Net income'] = netIncomel
    df['EPS'] = epsl

    df = df.set_index('Date')


    incomeStatementDelta = df.iloc[::-1]
    incomeStatementDelta = incomeStatementDelta.apply(pd.to_numeric)
    incomeStatementDelta = incomeStatementDelta.pct_change()
    incomeStatementDelta = incomeStatementDelta.apply(lambda x: x*100)
    incomeStatementDelta = incomeStatementDelta.dropna()


    revenueLength = len(incomeStatementDelta.loc[incomeStatementDelta['Revenue'] > 0])
    gpLength = len(incomeStatementDelta.loc[incomeStatementDelta['Gross profit'] > 0])
    netICLength = len(incomeStatementDelta.loc[incomeStatementDelta['Net income'] > 0])
    epsLength = len(incomeStatementDelta.loc[incomeStatementDelta['EPS'] > 0])



    finalRating = (revenueLength + gpLength + netICLength + epsLength + finalScore)


    closePrice = finalRating
    highPrice = maxRating

    confidenceLevel = 0.8

    topRange = highPrice
    midRange = highPrice*confidenceLevel
    lowRange = 0

    color = 'crimson'

    if closePrice > midRange:
        color = 'forestgreen'
    if closePrice == midRange:
        color = 'darkslateblue'
    

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = closePrice,
        gauge={"axis": {"range": [lowRange, topRange]},
            'steps' : [
                    {'range': [lowRange, topRange], 'color': "lightgray"}],
            'bar': {'color': color},

            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': midRange
                }
            

            
            
            },
        title = {'text': "Investment confidence"},
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))

    return fig

def bgSentiment3(asset1):

    URL1 = f'https://financialmodelingprep.com/api/v3/income-statement/{asset1}?limit=240&apikey={fmpAPI}'
    URL2 = f'https://financialmodelingprep.com/api/v3/rating/{asset1}?apikey={fmpAPI}'
    data = requests.get(URL1)
    data = data.json()

    scoreData = requests.get(URL2)
    scoreData = scoreData.json()


    datel= []
    revenuel = []
    grossProfitl = []
    netIncomel = []
    epsl = []
    finalScore = 0

    for value in scoreData:
        score = value['rating']

        if score == 'S+':
            finalScore = 5
        elif score =='S':
            finalScore = 4
        elif score =='S-':
            finalScore = 3
        elif score =='A+':
            finalScore = 2
        elif score =='A':
            finalScore = 1
        else:
            finalScore = 0







    for value in data:
        date = value['date']
        revenue = value['revenue']
        grossProfit = value['grossProfit']
        netIncome = value['netIncome']
        eps = value['eps']


        datel.append(date)
        revenuel.append(revenue)
        grossProfitl.append(grossProfit)
        netIncomel.append(netIncome)
        epsl.append(eps)

    df = pd.DataFrame()

    df['Date'] = datel
    df['Revenue'] = revenuel
    df['Gross profit'] = grossProfitl
    df['Net income'] = netIncomel
    df['EPS'] = epsl

    df = df.set_index('Date')


    incomeStatementDelta = df.iloc[::-1]
    incomeStatementDelta = incomeStatementDelta.apply(pd.to_numeric)
    incomeStatementDelta = incomeStatementDelta.pct_change()
    incomeStatementDelta = incomeStatementDelta.apply(lambda x: x*100)
    incomeStatementDelta = incomeStatementDelta.dropna()


    revenueLength = len(incomeStatementDelta.loc[incomeStatementDelta['Revenue'] > 0])
    gpLength = len(incomeStatementDelta.loc[incomeStatementDelta['Gross profit'] > 0])
    netICLength = len(incomeStatementDelta.loc[incomeStatementDelta['Net income'] > 0])
    epsLength = len(incomeStatementDelta.loc[incomeStatementDelta['EPS'] > 0])



    finalRating = (revenueLength + gpLength + netICLength + epsLength + finalScore)


    closePrice = finalRating
    highPrice = maxRating

    confidenceLevel = 0.8

    topRange = highPrice
    midRange = highPrice*confidenceLevel
    lowRange = 0

    color = 'crimson'

    if closePrice > midRange:
        color = 'forestgreen'
    if closePrice == midRange:
        color = 'darkslateblue'
    

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = closePrice,
        gauge={"axis": {"range": [lowRange, topRange]},
            'steps' : [
                    {'range': [lowRange, topRange], 'color': "lightgray"}],
            'bar': {'color': color},

            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': midRange
                }
            

            
            
            },
        title = {'text': "Investment confidence"},
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))

    return fig



def marketNews():
    URL  = f"https://financialmodelingprep.com/api/v3/fmp/articles?page=0&size=30&apikey={fmpAPI}"
    data = requests.get(URL)
    data = data.json()

    titleL = []
    datel = []

    for value in data['content']:
        date = (value['date']) 
        title = (value['title']) 
        content = (value['content']) 

        titleL.append(title)
        datel.append(date)

    return titleL, datel


def stockComparision(asset1, asset2, asset3,starttime):
    stockOne = api.get_bars(asset1, TimeFrame.Day, starttime, today, adjustment='split').df
    stockTwo = api.get_bars(asset2, TimeFrame.Day, starttime, today, adjustment='split').df
    stockThree = api.get_bars(asset3, TimeFrame.Day, starttime, today, adjustment='split').df

    fig1 = go.Figure(data=[go.Candlestick(x=stockOne.index,
                open=stockOne['open'],
                high=stockOne['high'],
                low=stockOne['low'],
                close=stockOne['close'],
                increasing_line_color= 'slateblue', decreasing_line_color= 'lightsalmon'                
                )])

    fig2 = go.Figure(data=[go.Candlestick(x=stockTwo.index,
            open=stockTwo['open'],
            high=stockTwo['high'],
            low=stockTwo['low'],
            close=stockTwo['close'],
            increasing_line_color= 'slateblue', decreasing_line_color= 'lightsalmon'
            )])

    fig3 = go.Figure(data=[go.Candlestick(x=stockThree.index,
            open=stockThree['open'],
            high=stockThree['high'],
            low=stockThree['low'],
            close=stockThree['close'],
            increasing_line_color= 'slateblue', decreasing_line_color= 'lightsalmon'
            )])
    



    logPrice1 = 100/(stockOne['open'][0])
    stockOneComp = stockOne.apply(lambda x: x*logPrice1)
    stockOneComp['symbol'] = f'{asset1}'
    logPrice2 = 100/(stockTwo['open'][0])
    stockTwoComp = stockTwo.apply(lambda x: x*logPrice2)
    stockTwoComp['symbol'] = f'{asset2}'
    logPrice3 = 100/(stockThree['open'][0])
    stockThreeComp = stockThree.apply(lambda x: x*logPrice3)
    stockThreeComp['symbol'] = f'{asset3}'

    frames = [stockOneComp, stockTwoComp, stockThreeComp]
  
    stockComp = pd.concat(frames)



    combinedGraph = px.line(stockComp, x=stockComp.index, y='close', color='symbol')




    return stockOne, stockTwo, stockThree ,fig1, fig2, fig3, combinedGraph








add_selectbox = st.sidebar.selectbox(
    "Dashboard type?",
    ("Dashboard", "Comparision")
)

if add_selectbox == "Comparision":
    st.title('Stock price comparision dashboard')
    st.subheader('Run a comparision')

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker1 = st.text_input('Input ticker one', value='MSFT')
            ticker1 = ticker1.upper()

        with col2:
            ticker2 = st.text_input('Input ticker two', value='AMZN')
            ticker2 = ticker2.upper()

        with col3:
            ticker3 = st.text_input('Input ticker three', value='AAPL')
            ticker3 = ticker3.upper()



    start_time = st.slider(
        "When do you start?",
        min_value = datetime.datetime.now().date() - timedelta(days=(365*5 + 1)),
        max_value = datetime.datetime.now().date()- timedelta(days=1),
        format="DD/MM/YY")

    runComp = st.button('Run comparision')

    if runComp:
        st.success('running comparision')
        
        stockOneDF, stockTwoDF, stockThreeDF ,tickerOneChart, tickerTwoChart,tickerThreeChart ,combinedGraph = stockComparision(ticker1, ticker2, ticker3, start_time)
        ticker1ConfidenceGraph = bgSentiment(ticker1)
        ticker2ConfidenceGraph = bgSentiment2(ticker2)
        ticker3ConfidenceGraph = bgSentiment3(ticker3)



        st.plotly_chart(combinedGraph, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        

        with col1:
            st.subheader(ticker1)

            currentPrice = (stockOneDF['close'][-1])
            StartPrice = (stockOneDF['close'][0])
            movement = round((((currentPrice - StartPrice) / StartPrice)*100),2)

            st.metric(f'Share price & Movement since start date', currentPrice, delta=f'{movement}%')
            st.plotly_chart(ticker1ConfidenceGraph, use_container_width=True)
            st.plotly_chart(tickerOneChart, use_container_width=True)




        with col2:
            st.subheader(ticker2)


            currentPrice = (stockTwoDF['close'][-1])
            StartPrice = (stockTwoDF['close'][0])
            movement = round((((currentPrice - StartPrice) / StartPrice)*100),2)

            st.metric(f'Share price & Movement since start date', currentPrice, delta=f'{movement}%')
            st.plotly_chart(ticker2ConfidenceGraph, use_container_width=True)
            st.plotly_chart(tickerTwoChart, use_container_width=True)





        with col3:
            st.subheader(ticker3)

            currentPrice = (stockThreeDF['close'][-1])
            StartPrice = (stockThreeDF['close'][0])
            movement = round((((currentPrice - StartPrice) / StartPrice)*100),2)

            st.metric(f'Share price & Movement since start date', currentPrice, delta=f'{movement}%')
            st.plotly_chart(ticker3ConfidenceGraph, use_container_width=True)
            st.plotly_chart(tickerThreeChart, use_container_width=True)




    

if add_selectbox == "Dashboard":
    
    st.title('Market Dashboard')
    st.header('Enter a ticker, run market summary or view the latest headlines')

    ticker = st.text_input('Enter a ticker', value='MSFT')
    option = st.selectbox(
            'Summary graph',
            ('QQQ', 'SPY', 'DIA'))
    ticker = ticker.upper()

    button1, button2, button3 = st.columns(3, gap='large')
    with button1:
        buttonPressed = st.button('Run ticker')
    with button2:
        marketSummaryPressed = st.button('Run market summary')
    
    with button3:
        newsButton = st.button('Market headlines')





    if buttonPressed:
        if ticker != "":
            st.success(f'Ticker found: {ticker}')
            incState, incomeStatementChart = incomeStatement(ticker)
            chart_data = chart_data(ticker)
            dcfData = dcf(ticker)
            tickerChart, tickerDataFrame = tickerGraph(ticker)
            
            upside = round((((round((dcfData['DCF'][-1]),2) - round((dcfData['Price'][-1]),2))/round((dcfData['Price'][-1]),2))*100),2)

            curretStockPrice = round((dcfData['Price'][-1]),2)

            
            epsData = chart_data['eps']


            currenEPS = round(chart_data['eps'][0], 2)
            priorYearEPS = round(chart_data['eps'][1], 2)
            priorTwoEPS = round(chart_data['eps'][2], 2)

            yearOneMovement = round((((chart_data['eps'][0] - chart_data['eps'][1])/chart_data['eps'][1])*100),2)
            yearTwoMovement = round((((chart_data['eps'][1] - chart_data['eps'][2])/chart_data['eps'][2])*100),2)
            yearThreeMovement = round((((chart_data['eps'][2] - chart_data['eps'][3])/chart_data['eps'][3])*100),2)


            st.plotly_chart(tickerChart, use_container_width=True)

            st.header('DCF Valuation')
        

            co1, co2, co3 = st.columns(3)   
            co1.metric('Date', dcfData.index[-1], delta=None)
            co2.metric("Share Price", curretStockPrice, delta=None)
            co3.metric("DCF / Upside", round((dcfData['DCF'][-1]),2), f'{upside}%')

            st.header('EPS Movement')
            col1, col2, col3 = st.columns(3)
    
            col1.metric("Current Year EPS Movement", currenEPS, f'{yearOneMovement}%')
            col2.metric("Prior Year EPS Movement", priorYearEPS, f'{yearTwoMovement}%')
            col3.metric("2 Years Prior EPS Movement", priorTwoEPS, f'{yearThreeMovement}%')


            st.plotly_chart(incomeStatementChart, use_container_width= True)

    
            col1, col2 = st.columns(2)

            with col1:
                df = epsData
                fig = px.bar(df, x=df.index, y='eps', text_auto='.2s', title="EPS Growth")
                fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=True)
                fig.update_layout(yaxis={'visible': True, 'showticklabels': False})
                epsBarChart = fig
                st.plotly_chart(epsBarChart, use_container_width=True)


            with col2:
                ebitdaGrowth = chart_data['ebitda']
                fig = px.bar(ebitdaGrowth, x=df.index, y='ebitda', text_auto='.2s', title="Ebitda Movement")
                fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=True)
                fig.update_layout(yaxis={'visible': True, 'showticklabels': False})
                edbitdaChart = fig
                st.plotly_chart(edbitdaChart, use_container_width= True)









            st.header('Raw data')
            st.table(chart_data)
            st.table(incState)
            st.table(tickerDataFrame)
        else:
            st.text('ticker cant be blank')

    if marketSummaryPressed:
        st.success('Running market summary')

        sectorData = sectorPerformance()
        tickerChart, tickerDataFrame = tickerGraph(option)
        losers, gainers, full, gainerloserChart = gainerslosers()
        st.plotly_chart(tickerChart, use_container_width=True)
        st.subheader('Gainer / Losers')
        st.plotly_chart(gainerloserChart, use_container_width=True)
        st.subheader('Sector performance')
        st.plotly_chart(sectorgraph(), use_container_width=True)
        st.subheader('Raw Data')
        st.table(sectorData)
        st.table(losers)
        st.table(gainers)
        st.table(tickerDataFrame)

    if newsButton:
        data, date = marketNews()
        st.markdown(f'◼ {date[0]}: {data[0]}')
        st.markdown(f'◼ {date[1]}: {data[1]}')
        st.markdown(f'◼ {date[2]}: {data[2]}')
        st.markdown(f'◼ {date[3]}: {data[3]}')
        st.markdown(f'◼ {date[4]}: {data[4]}')
        st.markdown(f'◼ {date[5]}: {data[5]}')
        st.markdown(f'◼ {date[6]}: {data[6]}')
        st.markdown(f'◼ {date[7]}: {data[7]}')
        st.markdown(f'◼ {date[8]}: {data[8]}')
        st.markdown(f'◼ {date[9]}: {data[9]}')
        st.markdown(f'◼ {date[10]}: {data[10]}')
        st.markdown(f'◼ {date[11]}: {data[11]}')
        st.markdown(f'◼ {date[12]}: {data[12]}')
        st.markdown(f'◼ {date[13]}: {data[13]}')
        st.markdown(f'◼ {date[14]}: {data[14]}')
        st.markdown(f'◼ {date[15]}: {data[15]}')
        st.markdown(f'◼ {date[16]}: {data[16]}')
        st.markdown(f'◼ {date[17]}: {data[17]}')
        st.markdown(f'◼ {date[18]}: {data[18]}')
        st.markdown(f'◼ {date[19]}: {data[19]}')
        st.markdown(f'◼ {date[20]}: {data[20]}')
        st.markdown(f'◼ {date[21]}: {data[21]}')
        st.markdown(f'◼ {date[22]}: {data[22]}')
        st.markdown(f'◼ {date[23]}: {data[23]}')
        st.markdown(f'◼ {date[24]}: {data[24]}')
        st.markdown(f'◼ {date[25]}: {data[25]}')
        st.markdown(f'◼ {date[26]}: {data[26]}')
        st.markdown(f'◼ {date[27]}: {data[27]}')
        st.markdown(f'◼ {date[28]}: {data[28]}')
        st.markdown(f'◼ {date[29]}: {data[29]}')

