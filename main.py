#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 20:50:38 2022

@author: riyazpatel
"""
################ Projet Dash - Haris PATEL - Riyâz PATEL - M2 DSS / Dr.Zitouni
#pip install snscrape
#pip install pandas
#pip install dash==2.0.0
#pip install dash_extensions
#pip install plotly
#pip install nltk
#pip install wordcloud
#pip install dash_bootstrap_components
#pip install gensim
#pip install spacy
#pip install textblob
#pip install textblob-fr
######## Imports et donnees préliminaires
#### Imports
import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import snscrape.modules.twitter as twitterScraper
import dash_bootstrap_components as dbc
from dash import dash_table as dt
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from datetime import date
import dash_extensions
# from dash_extensions import send_data_frame
from dash import Dash, dcc, html, Input, Output
from collections import Counter
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import base64
import time
import nltk
import re
from nltk.corpus import stopwords
from base64 import b64encode

import gensim
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
import numpy as np  # numerical computation
import matplotlib.pyplot as plt  # basic data visualization

import spacy
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('punkt')

import os
import shutil
import numpy as np
import pandas as pd
import calendar
from datetime import datetime
# pip install fpdf
from fpdf import FPDF

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
import string
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer

tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
import plotly.graph_objects as go
import plotly.express as px

# pip install kaleido
# conda install -c conda-forge python-kaleido
# import kaleido
# import os

# if not os.path.exists("images"):
#     os.mkdir("images")

#### Stopwords - WC / Fréquence mots

regex_pattern = re.compile(pattern="["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)

url_pattern = re.compile(r'(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?')

new_stopwords = stopwords.words('english') + stopwords.words('french') + stopwords.words('arabic') + stopwords.words(
    'german') + stopwords.words('italian') + stopwords.words('spanish')
stopwords_perso = ["c'est", "cest", "quil", "co", "qd", "alors", "si", "tant", "qua", "cela", "tout", "dun", "va",
                   "cette", "ça"]
new_stopwords.extend(stopwords_perso)
stop_words = set(STOP_WORDS)
#### Valeur des langues la dropdownlist pour le choix des langues

######## Imports et donnees preliminaires

######## Application Dash

def create_sentiment_labels(df, feature, value):
    df.loc[df[value] > 0, feature] = 'positive'
    df.loc[df[value] == 0, feature] = 'neutral'
    df.loc[df[value] < 0, feature] = 'negative'


def sentiment_analysis(dataframe):
    dataframe['blob_polarity'] = dataframe['clean_tweet_nltk'].apply(lambda x: TextBlob(x).sentiment.polarity)
    dataframe['blob_subjectivity'] = dataframe['clean_tweet_nltk'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    create_sentiment_labels(dataframe, 'blob_sentiment', 'blob_polarity')

    return dataframe[['clean_tweet_nltk', 'blob_polarity', 'blob_subjectivity', 'blob_sentiment']].head()


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def clean_text(tweet, lemmatize='nltk'):
    tweet = tweet.lower()  # lowercase
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)  # remove urls
    tweet = re.sub(r'\@\w+|\#', '', tweet)  # remove mentions of other usernames and the hashtag character
    tweet = remove_stopwords(tweet)  # remove stopwords with Gensim

    if (lemmatize == 'spacy'):
        # Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
        nlp = spacy.load('en', disable=['parser', 'ner'])
        doc = nlp(tweet)
        tokenized = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    elif (lemmatize == 'nltk'):

        lemmatizer = WordNetLemmatizer()
        tokenized = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(tweet)]

    # remove left over stop words with nltk
    tokenized = [token for token in tokenized if token not in stopwords.words("english")]

    # remove non-alpha characters and keep the words of length >2 only
    tokenized = [token for token in tokenized if token.isalpha() and len(token) > 2]

    return tokenized


def combine_tokens(tokenized):
    non_tokenized = ' '.join([w for w in tokenized])
    return non_tokenized


lang_options = [{'label': 'Arabic | العربية', 'value': 'ar'},
                {'label': 'British English | British English', 'value': 'en-gb'},
                {'label': 'English | English', 'value': 'en'}, {'label': 'French | français', 'value': 'fr'},
                {'label': 'German | Deutsch', 'value': 'de'}, {'label': 'Italian | italiano', 'value': 'it'},
                {'label': 'Spanish | español', 'value': 'es'}]
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MORPH])

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

#### Création du Layout


#### Creation et stockage du dataframe de tweets scrape
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Dashboard Twitter Analytics",
                            style={"textAlign": "center"}), width=22)),
    dbc.Row(dbc.Col(html.H1("  No-code Twitter scrapping and analytics",
                            style={"textAlign": "center", "fontSize": 20}), width=22)),
    html.Hr(),
    html.Div(className='row1',
             style={'display': 'flex'},
             children=[
                 html.Div([
                     html.H1('Request', style={'fontSize': 20}),
                     dcc.Input(id='searchId', placeholder='Ex: vaccin, santé, covid', type='text'),
                     html.Hr(),
                     html.H1('Language', style={'fontSize': 20}),
                     dcc.Dropdown(id='twitter_search_lang', placeholder='Language',
                                  options=lang_options, value='fr'
                                  )], style={'width': '33%', 'display': 'inline-block'}),

                 html.Div([
                     html.H2('Time range', style={'fontSize': 20}),
                     dcc.DatePickerSingle(
                         id='date1_Id',
                         min_date_allowed=date(2007, 8, 5),
                         date=date(2022, 1, 1),
                         display_format='YYYY-MM-DD'
                     ),
                     dcc.DatePickerSingle(
                         id='date2_Id',
                         min_date_allowed=date(2007, 8, 5),
                         date=date.today(),
                         display_format='YYYY-MM-DD'
                     )], style={'width': '33%', 'display': 'inline-block'}),

                 html.Div([
                     html.H3('Number of tweets', style={'fontSize': 20}),
                     dcc.Input(id='countId',
                               placeholder='Ex: 100, 200', type='number', value=1000),
                     dcc.Store(id='memory'),
                     html.Hr(),
                     html.Hr(),
                     dbc.Button("Launch", id="button", n_clicks=0, outline=True, color="primary", className="me-1"),
                     dbc.Spinner(html.Div(id="loading-output"), color="primary")],
                     style={'width': '33%', 'display': 'inline-block'})
             ]),

    html.Hr(),
    html.Div([
        dcc.Tabs([
            dcc.Tab(label='Data base', children=[
                dt.DataTable(
                    columns=[{'id': c, 'name': c} for c in
                             ['URL', 'Text', 'Datetime', 'RetweetCount', 'ReplyCount', 'LikeCount', 'Username']],
                    # style_data={'whiteSpace': 'normal','height': 'auto'},
                    id='tweet_table',
                    style_cell_conditional=[
                        {
                            'if': {'column_id': c},
                            'textAlign': 'left'
                        } for c in ['Date', 'Region']
                    ],
                    style_data={
                        'color': 'black',
                        'backgroundColor': 'white', 'whiteSpace': 'normal', 'height': 'auto'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(180,206,255)',
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(180,206,255)',
                        'color': 'black',
                        'fontWeight': 'bold'
                    },
                    style_table={'overflowY': 'scroll', 'overflowX': 'scroll', 'height': 500}),
                html.Hr(),
                dbc.Button("Download CSV", id="btn_csv", size="sm", className="me-1", outline=True, color="success"),
                dcc.Download(id="download-dataframe-csv")
                # dbc.Button("Download pdf", id="btn_pdf", size="sm", className="me-1",outline=True, color="success"),
                # dcc.Download(id="download-pdf")
            ]),
            dcc.Tab(label='Temporal analysis', children=[
                dcc.Graph(id='graph-day'),
                dcc.Graph(id='graph-hour')
            ]),
            dcc.Tab(label='Textual analysis', children=[
                dcc.Graph(id='tweet_word_count'),
                html.Hr(),
                dcc.RangeSlider(
                    id='range_frequency_number',
                    min=1,
                    max=200,
                    step=1,
                    value=[1, 200],
                    marks={
                        1: {'label': 'Minimum'},
                        50: {'label': '50'},
                        100: {'label': '100 most frequent words'},
                        200: {'label': '200'}
                    },
                    pushable=1,
                    tooltip={"placement": "bottom", "always_visible": False},
                    allowCross=False
                ),
                html.Hr(),
                html.Img(id='image_wc', style={
                    'height': '100%',
                    'width': '100%'
                })
            ]),
            dcc.Tab(label='Sentiment analysis (V1)', children=[
                dcc.Graph(id="pie-chart"),
                html.Hr(),
                dt.DataTable(
                    columns=[{'id': c, 'name': c} for c in ['Text', 'blob_sentiment']],
                    # style_data={'whiteSpace': 'normal','height': 'auto'},
                    id='sentiment_table',
                    style_cell_conditional=[
                        {
                            'if': {'column_id': c},
                            'textAlign': 'left'
                        } for c in ['Date', 'Region']
                    ],
                    style_data={
                        'color': 'black',
                        'backgroundColor': 'white', 'whiteSpace': 'normal', 'height': 'auto'
                    },
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{sentiment} contains "Neutral"'
                            },
                            'backgroundColor': 'blue',
                            'color': 'white'
                        },
                        {
                            'if': {
                                'filter_query': '{sentiment} contains "Positive"'
                            },
                            'backgroundColor': 'green',
                            'color': 'white'
                        },
                        {
                            'if': {
                                'filter_query': '{sentiment} contains "Negative"'
                            },
                            'backgroundColor': 'red',
                            'color': 'white'
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(180,206,255)',
                        'color': 'black',
                        'fontWeight': 'bold'
                    },
                    style_table={'overflowY': 'scroll', 'overflowX': 'scroll', 'height': 500})])
        ]),
        html.Hr()

    ]),
    html.P([html.A("Code source", href="https://gitlab.univ-lille.fr/haris.patel.etu/dash_tw_analytics"),
            ".  Projet DASH : Haris PATEL - Riyâz PATEL - M2 DSS / Dr. Zitouni "
            ], style={"textAlign": "center"})

])

#### Création du Layout


#### Creation et stockage du dataframe de tweets scrape
@app.callback(
    [Output("loading-output", "children"),
     Output('memory', 'data')],
    [Input("button", "n_clicks")],
    [State('searchId', 'value'),
     State('twitter_search_lang', 'value'),
     State('date1_Id', 'date'),
     State('date2_Id', 'date'),
     State('countId', 'value')]

)
def update_Frame(n_clicks, searchId, twitter_search_lang, date_since, date_until, count):
    if searchId is None:
        raise PreventUpdate
    if n_clicks:
        time.sleep(1)
        tweet_list = []
        for i, tweet in enumerate(twitterScraper.TwitterSearchScraper(
                query=searchId + " since:" + date_since + " until:" + date_until + " lang:" + twitter_search_lang).get_items()):
            if i > count:
                break
            tweet_list.append(
                [tweet.content, tweet.date, tweet.replyCount, tweet.retweetCount, tweet.likeCount
                 ])
        # Creating a dataframe from the tweets list above
        df = pd.DataFrame(tweet_list,
                          columns=['Text', 'Datetime', 'RetweetCount', 'ReplyCount', 'LikeCount'
                                   ])
        return f"Output loaded ", df.to_dict(orient='records')

#### Creation et stockage du dataframe de tweets scrape


#### Affichage de la base de donnee
@app.callback(
    [
        Output(component_id='tweet_table', component_property='data'),
        Output(component_id='tweet_table', component_property='columns')],
    [Input(component_id='memory', component_property='data')]
)
def display_tweets(df):
    tweets = pd.DataFrame(df)
    columns = [{'name': col, 'id': col} for col in tweets.columns]
    data = tweets.to_dict(orient='records')
    return data, columns

#### Affichage de la base de donnee


#### Graph : Nombre de tweets par jour
@app.callback(
    Output(component_id='graph-day', component_property='figure'),
    [Input('memory', 'data')],
    [State('date1_Id', 'date'),
     State('date2_Id', 'date')]
)
def update_div1(df, min_date_range, max_date_range):
    if df is None:
        raise PreventUpdate

    data = pd.DataFrame(df)

    data['date'] = pd.DatetimeIndex(data['Datetime']).date
    data['count'] = 1
    # data_filtered = data[['hour', 'date', 'count']]
    data_filtered = data[['date', 'count']]

    df_tweets_date = data_filtered.groupby(["date"]).sum().reset_index()

    df_tweets_date.set_index('date', inplace=True)

    df_tweets_date = df_tweets_date.reindex(pd.date_range(min_date_range, max_date_range), fill_value=0)

    fig1 = px.line(df_tweets_date, x=df_tweets_date.index, y='count',
                   title='The count of tweets for each day')

    fig1.update_layout(xaxis=dict(title='Date'),
                       yaxis=dict(title='Count Total'), title_x=.5,
                       margin=dict(r=0))

    return fig1

#### Graph : Nombre de tweets par jour


#### Graph : Repartition des tweets en 24heures
@app.callback(
    Output(component_id='graph-hour', component_property='figure'),
    [Input('memory', 'data')]
)
def update_div2(df):
    if df is None:
        raise PreventUpdate

    data = pd.DataFrame(df)
    data['hour'] = pd.DatetimeIndex(data['Datetime']).hour  # 1

    data['count'] = 1  # 2

    data = data[['hour', 'count']]  # 3

    data['hour'] = pd.to_datetime(data['hour'], format='%H')  # 4

    df_tweets_hourly = data.groupby(["hour"]).sum().reset_index()  # 5

    df_tweets_hourly.set_index('hour', inplace=True)  # 6

    df_tweets_hourly = df_tweets_hourly.reindex(
        pd.date_range(start="1900-01-01 00:00:00", end="1900-01-01 23:00:00", freq="60min"), fill_value=0)  # 7

    df_tweets_hourly['datetime'] = df_tweets_hourly.index  # 8
    df_tweets_hourly['hour'] = pd.DatetimeIndex(df_tweets_hourly['datetime']).hour  # 9

    fig = px.bar(df_tweets_hourly, x='hour', y='count', range_x=[0, 23])

    fig = px.bar(df_tweets_hourly, x='hour', y='count',
                 title='The count of tweets for each hour')

    fig.update_layout(xaxis=dict(title='Hours'),
                      yaxis=dict(title='Count Total'), title_x=.5,
                      margin=dict(r=0))

    return fig

#### Graph : Repartition des tweets en 24heures

#### Bouton pour telecharger la base de donnees des tweets scrapes, memory, en csv
@app.callback(
    Output('download-dataframe-csv', 'data'),
    [Input('btn_csv', 'n_clicks')],
    [State('memory', 'data')],
    prevent_initial_call=True,
)
def func(n_clicks, df):
    data = pd.DataFrame(df)
    return dcc.send_data_frame(data.to_csv, "mydf.csv")

#### Graph : Frequence des mots des tweets et WordCloud

######## Application Dash

@app.callback(
    [Output(component_id='tweet_word_count', component_property='figure'),
     Output('image_wc', 'src')],
    [Input('memory', 'data'),
     Input('range_frequency_number', 'value')
     ]
)
def word_count_graph(df, count_value):
    if df is None:
        raise PreventUpdate

    data = pd.DataFrame(df)

    # convert to lower
    data['clean_text'] = data['Text'].apply(
        lambda x: ' '.join([word for word in x.lower().split() if word not in (new_stopwords)]))
    # enlever les emojis
    data['clean_text'] = data['clean_text'].str.replace(emoji_pattern, '', regex=True)
    # enlever les liens
    data['clean_text'] = data['clean_text'].str.replace(url_pattern, '', regex=True)
    # Remplacer la ponctuation par un espace
    data['clean_text'] = data['clean_text'].str.replace(r'[^\w\s]+', ' ', regex=True)
    # Filtrer les stopwords
    data['clean_text'] = data['clean_text'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (new_stopwords)]))
    # Count Words
    countFreq = data.clean_text.str.split(expand=True).stack().value_counts()[:100]

    newGraph = countFreq[count_value[0]:count_value[1]]

    fig = px.bar(x=newGraph.index, y=newGraph.values, title="Tweet Word Frequency",
                 labels={'x': 'Words', 'y': 'Count'})

    wc = WordCloud(background_color='#F0FFFE', width=800, height=600).generate_from_frequencies(newGraph)

    wc_img = wc.to_image()
    img = BytesIO()
    wc_img.save(img, format='PNG')

    return fig, "data:image/png;base64,{}".format(base64.b64encode(img.getvalue()).decode())


#### Pie CHART : Analyse des sentiments

@app.callback(
    Output("pie-chart", "figure"),
    [Input('memory', 'data')],
    [State('twitter_search_lang', 'value')])
def pie_chart_sentiment(df, langue):
    if df is None:
        raise PreventUpdate

    if langue == 'fr':

        data = pd.DataFrame(df)

        data["Text"] = data["Text"].str.lower()

        AComment = []
        AllfilteredComment = []
        senti_list = []

        for comment in data["Text"].apply(str):
            Word_Tok = []
            for word in re.sub("\W", " ", comment).split():
                Word_Tok.append(word)
            AComment.append(Word_Tok)
        data["Word_Tok"] = AComment

        stop_words = set(STOP_WORDS)

        deselect_stop_words = ['n\'', 'ne', 'pas', 'plus', 'personne', 'aucun', 'ni', 'aucune', 'rien']

        for w in deselect_stop_words:
            if w in stop_words:
                stop_words.remove(w)
            else:
                continue

        for comment in data["Word_Tok"]:
            filteredComment = [w for w in comment if not ((w in stop_words) or (len(w) == 1))]
            AllfilteredComment.append(' '.join(filteredComment))

        data["CommentAferPreproc"] = AllfilteredComment

        for i in data["CommentAferPreproc"]:
            vs = tb(i).sentiment[0]
            if (vs > 0):
                senti_list.append('Positive')
            elif (vs < 0):
                senti_list.append('Negative')
            else:
                senti_list.append('Neutral')

        data["sentiment"] = senti_list
        newdf = pd.DataFrame(data['sentiment'].value_counts())
        df2 = newdf.rename_axis('Sentiment_blob').reset_index()
        fig1 = px.pie(df2, values='sentiment', names='Sentiment_blob',
                      title='Pie chart : Sentiments analysis (Natural Language Processing)',
                      color='Sentiment_blob', color_discrete_map={'Neutral': 'blue',
                                                                  'Positive': 'green',
                                                                  'Negative': 'red'})
        return fig1

    if langue == 'en':

        data = pd.DataFrame(df)
        data['tokenized_tweet_nltk'] = data['Text'].apply(lambda x: clean_text(x, 'nltk'))
        data['clean_tweet_nltk'] = data['tokenized_tweet_nltk'].apply(lambda x: combine_tokens(x))
        sentiment_analysis(data)
        newdf = pd.DataFrame(data['blob_sentiment'].value_counts())
        df2 = newdf.rename_axis('sentiment').reset_index()
        fig2 = px.pie(df2, values='blob_sentiment', names='sentiment',
                      title='Pie chart : Sentiments analysis (Natural Language Processing)',
                      color_discrete_map={'Neutral': 'blue',
                                          'Positive': 'green',
                                          'Negative': 'red'})

        return fig2

    else:
        None


@app.callback(

    [Output(component_id='sentiment_table', component_property='data'),
     Output(component_id='sentiment_table', component_property='columns')],

    [Input(component_id='memory', component_property='data')],
    [State('twitter_search_lang', 'value')]

)
def display_sentiments(df, language):
    if language == 'en':
        data = pd.DataFrame(df)
        data['tokenized_tweet_nltk'] = data['Text'].apply(lambda x: clean_text(x, 'nltk'))
        data['clean_tweet_nltk'] = data['tokenized_tweet_nltk'].apply(lambda x: combine_tokens(x))
        sentiment_analysis(data)
        mycolumns = ['Text', 'blob_sentiment']
        data_aff = data[mycolumns]
        data_aff1 = data_aff.rename(columns={'blob_sentiment': 'sentiment'})
        columns = [{'name': col, 'id': col} for col in data_aff1.columns]
        data_aff_2 = data_aff1.to_dict(orient='records')
        return data_aff_2, columns

    if language == 'fr':

        data = pd.DataFrame(df)

        data["Text"] = data["Text"].str.lower()

        AComment = []
        AllfilteredComment = []
        senti_list = []

        for comment in data["Text"].apply(str):
            Word_Tok = []
            for word in re.sub("\W", " ", comment).split():
                Word_Tok.append(word)
            AComment.append(Word_Tok)
        data["Word_Tok"] = AComment

        stop_words = set(STOP_WORDS)

        deselect_stop_words = ['n\'', 'ne', 'pas', 'plus', 'personne', 'aucun', 'ni', 'aucune', 'rien']

        for w in deselect_stop_words:
            if w in stop_words:
                stop_words.remove(w)
            else:
                continue

        for comment in data["Word_Tok"]:
            filteredComment = [w for w in comment if not ((w in stop_words) or (len(w) == 1))]
            AllfilteredComment.append(' '.join(filteredComment))

        data["CommentAferPreproc"] = AllfilteredComment

        for i in data["CommentAferPreproc"]:
            vs = tb(i).sentiment[0]
            if (vs > 0):
                senti_list.append('Positive')
            elif (vs < 0):
                senti_list.append('Negative')
            else:
                senti_list.append('Neutral')

        data["sentiment"] = senti_list
        mycolumns = ['Text', 'sentiment']
        data_aff = data[mycolumns]
        columns = [{'name': col, 'id': col} for col in data_aff.columns]
        data_aff_1 = data_aff.to_dict(orient='records')
        return data_aff_1, columns
    else:
        None

if __name__ == '__main__':
    app.run_server()










