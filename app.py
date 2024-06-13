# Import Libraries
import pandas as pd

import matplotlib
matplotlib.use('agg')

import dash
from dash import Dash, dash_table
from dash import html
from dash import dcc
from dash.dependencies import Output, Input, State

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff

import numpy as np

'################################################################################'
import pathlib
import os

# heroku csv reading function
def load_data(data_file: str) -> pd.DataFrame:
    '''
    Load data from /data directory
    '''
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("data").resolve()
    return pd.read_csv(DATA_PATH.joinpath(data_file))
'################################################################################'

# Preprocessing
df = load_data("data.csv")
'################################################################################'
df.set_index('Unnamed: 0', inplace=True)
df.rename_axis(None, inplace=True) #remove index header

################################################################################

# Year List
unique_years = df.iloc[:,3:].T.index.str.extract(r'(\d{4})').squeeze().unique().astype(int)
yearlist = np.sort(unique_years)[::-1]

################################################################################

# Initialize the namelist
namelist = ['Panos', 'Ashish', 'Kartik', 'Akshaye', 'Chris', 'Sid', 'Tanish', 'Yufeng', 'Soumil']
# Sort the names by all-time profit
namelist = df.loc[namelist, df.columns[3:]].T.sum().sort_values(ascending=False).index.to_list()

################################################################################

# Main preprocessing function

def df_selected_year(df, selected_year):
    selected_year = str(selected_year)
    df = df.filter(like=selected_year, axis=1) # get only columns of the selected year
    df.insert(0, 'NET', df.sum(axis=1))
    df.insert(1, 'PPG', df.iloc[:,1:].mean(axis=1))
    df.insert(2, 'TABLES', df.iloc[:,2:].count(axis=1))

    dftime2 = df.iloc[:,3:].T.copy()
    df.fillna(0, inplace=True)
    dftime = df.iloc[:,2:].T.copy()
    


    # Preprocessing for profit/loss against time analysis
    other_players = [col for col in dftime.columns if col not in namelist] # Create a list of players not in the selected list
    dftime['Guests'] = dftime[other_players].sum(axis=1).to_frame() # sum all guest values
    dftime.drop(columns=other_players, inplace=True) # drop the other players

    if 'TABLES' in dftime.index:
        dftime.drop(index='TABLES', inplace=True)
        
    dftime = dftime.rename_axis('Date').reset_index()



    # Initialize variables to keep track of cumulative sums for each column
    cumulative_sums = {col: 0 for col in dftime.columns[1:]}

    # Iterate through the columns and update values so that the datum of the nth row is the sum of all data up to and including row n 
    for col in dftime.columns[1:]:
        for index, row in dftime.iterrows():
            cumulative_sums[col] += row[col]
            dftime.at[index, col] = cumulative_sums[col]

    # Convert 'Date' column to datetime format
    dftime['Date'] = pd.to_datetime(dftime['Date'], format='%d/%m/%Y')

    # Format the 'Date' column
    dftime['Date'] = dftime['Date'].dt.strftime('%d-%b')

    return df, dftime, dftime2

################################################################################

def melt_data(dftime):
    dftime_melt = dftime.melt(id_vars='Date', value_vars=list(dftime.columns[1:])).rename({'variable':'Player','value':'Net Profit/Loss'}, axis='columns') #melt the table and rename the new columns
    dftime_melt['Net Profit/Loss'] = round(dftime_melt['Net Profit/Loss'],2) # round all values to 2 decimal points
    return dftime_melt

################################################################################

# Helper function for plotly plots

def update_plotly_background(fig, x_zeroline=True, y_zeroline=True, x_grid=True, y_grid=True):
    fig.update_layout(
        plot_bgcolor='white',
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        zeroline=True if x_zeroline==True else False,
        zerolinecolor='lightgrey',
        showgrid=False if x_grid==False else True
    )

    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        zeroline=True if y_zeroline==True else False,
        zerolinecolor='lightgrey',
        showgrid=False if y_grid==False else True
    )

################################################################################

# Group Analysis

# Session Results
def make_table(row, selected_year, df_raw):
    # Get date of the selected year only
    df, dftime, dftime2, dftime3 = df_selected_year(df=df_raw, selected_year=selected_year)

    selected_data = dftime2.iloc[row:row + 1].dropna(axis=1).to_dict('records')
    fig = dash_table.DataTable(
        id='results-table',  # Specify the id for the DataTable
        columns=[{'name': col, 'id': col} for col in dftime2.iloc[row:row + 1].dropna(axis=1).columns],
        data=selected_data,
        style_table={'fontSize': 22},  # Set the font size
    )
    return fig

################################################################################

# Year Stats

def year_table_stats(selected_year, df_raw):
    df, _, _ = df_selected_year(df=df_raw, selected_year=selected_year)
    df = df.loc[namelist]
    df = df.iloc[:, :3]
    df = df.astype(float)
    df = df.reset_index().rename(columns={'index': 'Player', 'NET': 'Net Profit', 'PPG': 'Profit per Game', 'TABLES': 'Games Played'}).sort_values(by='Net Profit')
    df = df.sort_values(by='Net Profit', ascending=False)
    df['Profit per Game'] = df['Profit per Game'].astype(float).round(2)
    df['Net Profit'] = df['Net Profit'].astype(float).round(2)

    data = df.to_dict(orient='records')
    fig = dash_table.DataTable(
        #columns=[{'name': col, 'id': col} for col in df.columns],
        columns=[
            {'name': 'Player', 'id': 'Player'},
            {'name': 'Net Profit', 'id': 'Net Profit', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Profit per Game', 'id': 'Profit per Game', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Games Played', 'id': 'Games Played', 'type': 'numeric'}
        ],
        data=data,
        style_table={'fontSize': 22},
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '10px'},
        # allow sorting
        sort_action='native',
        sort_mode='multi',
    )
    return fig

################################################################################

# All Time Stats

def all_time_table_stats(df_raw):
    df2 = df_raw.iloc[:, 3:].copy()

    df2.insert(0, 'NET', df2.sum(axis=1))
    df2.insert(1, 'PPG', df2.iloc[:,1:].mean(axis=1))
    df2.insert(2, 'TABLES', df2.iloc[:,2:].count(axis=1))
    df2.fillna(0, inplace=True)

    df2 = df2.loc[namelist]
    df2 = df2.iloc[:,:3]
    df2 = df2.astype(float)
    df2 = df2.reset_index().rename(columns={'index':'Player', 'NET': 'Net Profit', 'PPG': 'Profit per Game', 'TABLES': 'Games Played'}).sort_values(by='Net Profit')
    df2 = df2.sort_values(by='Net Profit', ascending=False)
    df2['Profit per Game'] = df2['Profit per Game'].astype(float).round(2)
    df2['Net Profit'] = df2['Net Profit'].astype(float).round(2)
    data = df2.to_dict(orient='records')

    fig = dash_table.DataTable(
        #columns=[{'name': col, 'id': col} for col in df2.columns],
        columns=[
            {'name': 'Player', 'id': 'Player'},
            {'name': 'Net Profit', 'id': 'Net Profit', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Profit per Game', 'id': 'Profit per Game', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Games Played', 'id': 'Games Played', 'type': 'numeric'}
        ],
        data=data,
        style_table={'fontSize': 22},
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '10px'},
        # allow sorting
        sort_action='native',
        sort_mode='multi',
    )
    return fig

################################################################################

# Current Profit/Loss
def Status_bar(player_list, selected_year, df_raw):
    df, dftime, _ = df_selected_year(df=df_raw, selected_year=selected_year)
    # get the current p/l from the last row of the dataframe
    data = [round(value, 2) for value in dftime.loc[dftime.index[-1], player_list].tolist()]
    df = pd.DataFrame({'Player': player_list, 'Net Profit/Loss': data})
    fig = px.bar(df, x='Net Profit/Loss', y='Player', text='Net Profit/Loss', orientation='h')
    
    fig.update_layout(
        xaxis_title='Net Profit/Loss',
        yaxis_title='Player',
        xaxis=dict(
            tickformat="%b",
            showgrid=True,
            tickfont = {'size':16},
        ),
        title_x=0.5,
        title_font = {'size':20},
        yaxis=dict(
            tickfont = {'size':16},
        )
    )

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    update_plotly_background(fig, y_grid=False)
    
    return fig

################################################################################

# Time Lapse Bar Plot
def Time_Lapse(player_list, selected_year, df_raw):
    _, dftime, _ = df_selected_year(df=df_raw, selected_year=selected_year)

    dftime_melt = melt_data(dftime)
    dftime_melt = dftime_melt[dftime_melt['Player'].isin(player_list)] #update the dataframe to include only the names selected

    max_value = dftime_melt['Net Profit/Loss'].max()//50*50+50
    min_value = dftime_melt['Net Profit/Loss'].min()//50*50

    fig = px.bar(dftime_melt,  
        x='Net Profit/Loss', y = "Player", animation_frame="Date", height=600, hover_data=['Net Profit/Loss'],color='Net Profit/Loss', orientation= 'h', range_x=(min_value, max_value))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Net Profit/Loss",
        xaxis=dict(
            tickformat="%b",
            showgrid=True,
            tickfont = {'size':16},
        ),
        title_x=0.5,
        title_font = {'size':20},
        yaxis=dict(
            tickfont = {'size':16},
        )
    )

    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", overwrite=True,)  
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
    update_plotly_background(fig, y_zeroline=False, y_grid=False)
    
    return fig

################################################################################

# Yearly Rank Plot

def rank_over_session(selected_year, df_raw):
    _, dftime, _ = df_selected_year(df=df_raw, selected_year=selected_year)
    dftime_melt = melt_data(dftime)
    dftime_melt = dftime_melt[dftime_melt['Player'].isin(namelist)]

    dfc = dftime.loc[:, namelist]

    ranked_df = dfc.rank(axis=1, method='average', ascending=False)
    ranked_df.index = ranked_df.index + 1

    ranked_df_long = ranked_df.reset_index().melt(id_vars='index', var_name='Player', value_name='Rank').rename(columns={'index': 'Session Number'})

    color_palette = ['red', 'purple', 'orange', 'blue', 'gray', 'brown', 'green', 'black', 'lime']

    # Create the line plot
    fig = px.line(ranked_df_long, 
                x='Session Number', 
                y='Rank', 
                color='Player', 
                markers=True,
                height=600,
                color_discrete_sequence=color_palette,
                category_orders={'Rank': sorted(ranked_df_long['Rank'].unique())},
                hover_data={'Rank': True, 'Player': True, 'Session Number': True})

    # Update hover template to control the order and format
    fig.update_traces(
        hovertemplate='<b>Rank: %{y}</b><br>Player: %{fullData.name}<br>Session Number: %{x}<extra></extra>',
    )

    fig.update_layout(
        yaxis=dict(
            title='Rank',
            autorange='reversed',  # Reverse the order of the y-axis
            dtick=1
        ),
        legend_title='Player',
        xaxis=dict(
            title='Session Number',
            tickmode='linear',
            tick0=0,
            dtick=1
        )
    )
    update_plotly_background(fig, x_zeroline=False, y_grid=False)
    
    return fig

################################################################################

# Player Analysis
def Current_profit_loss(person, selected_year, df_raw):
    df, dftime, dftime2 = df_selected_year(df=df_raw, selected_year=selected_year)

    def get_ppg(person): # find the previous profit per game statistic
        ppg = df.loc[person]['PPG']
        games_played = df.loc[person]['TABLES']
        last_p_l = df.at[person, df.columns[-1]]
        net = dftime.loc[dftime.index[-1], person]
        (ppg_ref := ppg) if np.isnan(dftime2[person].tail(1).values) else (ppg_ref := (net - last_p_l) / (games_played - 1))
        return round(ppg_ref, 1)
    
    def get_position(dftime, person):
        position = dftime.iloc[-1,1:-1].rank(ascending=False)
        position = int(position[person])
        reference_pos = dftime.iloc[-2,1:-1].rank(ascending=False)
        reference_pos = int(reference_pos[person])
        return position, reference_pos
    
    fig = make_subplots(rows=1, cols=4)
    
    # First gauge chart
    trace1 = (go.Indicator(
        mode = "number+delta",
        value= get_position(dftime, person)[0],
        title={'text': "Ranking"},
        delta= {'reference': 2*get_position(dftime, person)[0] - get_position(dftime, person)[1], 'increasing': {'color': "#006C5B"}, 'decreasing': {'color': "red"}},
        )) # to reverse the sign of the reference number: -difference = (final-reference)+final = 2*final-reference

    # Second gauge chart
    trace2 = (go.Indicator(
        mode = "number+delta",
        value= int(dftime.loc[dftime.index[-1], person]), #Take the last value of the column
        title={'text': "Net Profit/Loss"},
        delta= {'reference': int(dftime.loc[dftime.index[-2], person]), 'increasing': {'color': "#006C5B"}, 'decreasing': {'color': "red"}},
        ))

    # Third gauge chart
    trace3 = (go.Indicator(
        mode = "number+delta",
        value= round(df.loc[person]['PPG'], 1), # ppg as defined by the first table
        title={'text': "Profit per Game"},
        delta= {'reference': get_ppg(person), 'increasing': {'color': "#006C5B"}, 'decreasing': {'color': "red"}},
        ))

    # Fourth gauge chart
    trace4 = (go.Indicator(
        mode = "number+delta",
        value= int(df.loc[person]['TABLES']), # ppg as defined by the first table
        title={'text': "Games Played"},
        delta={'reference': int(df.loc[person]['TABLES'] - df_raw.loc[person].tail(1).notna().values), 
               'increasing': {'color': "#006C5B"}, 'decreasing': {'color': "red"}},
        ))

    fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{'type' : 'indicator'}, {'type' : 'indicator'}, {'type': 'indicator'}, {'type' : 'indicator'}]],
        )

    fig.append_trace(trace1, row=1, col=1)
    fig.append_trace(trace2, row=1, col=2)
    fig.append_trace(trace3, row=1, col=3)
    fig.append_trace(trace4, row=1, col=4)

    return fig

################################################################################

# Session Progress Plot
def Progress_Session(player_list, person, selected_year, df_raw):
    # Get date of the selected year only
    _, dftime, _ = df_selected_year(df=df_raw, selected_year=selected_year)

    x_data = np.arange(1, len(dftime[person]) + 1)

    fig = px.line(height=600,) # Initialize figure

    # Add the player data line first
    player_line = go.Scatter(x=x_data, y=dftime[person], mode='lines', name='{}'.format(person), line=dict(color='blue'))
    fig.add_trace(player_line)

    # Add the 5-day average data
    if dftime.shape[0] >= 6:
        average_5_game = go.Scatter(x=np.arange(3, dftime[person].shape[0]-1, 1), y=dftime[person].rolling(window=5).mean().dropna().values, mode='lines', name='{} (5-game avg)'.format(person), line=dict(color='mediumblue'))
        fig.add_trace(average_5_game)
    
    # Add the progress lines of each other player
    color_palette = ['red', 'purple', 'orange', 'gray', 'brown', 'green', 'black', 'lime']

    for player in range(len(player_list[:8])):
        player_line = go.Scatter(x=x_data, y=dftime[player_list[player]], mode='lines', name=player_list[player], line=dict(color=color_palette[player]))
        fig.add_trace(player_line)

    fig.update_layout(
        xaxis_title='Session Number ({})'.format(selected_year),
        yaxis_title='Net Profit/Loss',
        xaxis=dict(
            tickformat="%b",
            showgrid=True,
            tickfont = {'size':16},
            tickmode='linear',
        ),
        title_x=0.5,
        title_font = {'size':20},
        yaxis=dict(
            tickfont = {'size':16},
        )
    )
    update_plotly_background(fig)

    return fig

################################################################################

# Monthly Progress Plot
def Progress_Monthly(player_list, person, selected_year, df_raw):  
    # Get date of the selected year only
    _, dftime, _ = df_selected_year(df=df_raw, selected_year=selected_year)

    x_data = pd.to_datetime(dftime['Date'] + f"-{selected_year}", format='%d-%b-%Y')

    fig = px.line(height=600,) #initialise figure

    # Add the player data line
    player_line = go.Scatter(x=x_data, y=dftime[person], mode='lines', name='({})'.format(person), line=dict(color='blue'))
    fig.add_trace(player_line)

    # Add the progress lines of each other player
    color_palette = ['red', 'purple', 'orange', 'gray', 'brown', 'green', 'black', 'lime']

    for player in range(len(player_list[:8])):
        player_line = go.Scatter(x=x_data, y=dftime[player_list[player]], mode='lines', name=player_list[player], line=dict(color=color_palette[player]))
        fig.add_trace(player_line)

    # Customize the layout, including the title
    fig.update_layout(
        xaxis_title='Month ({})'.format(selected_year),
        yaxis_title='Net Profit/Loss',
        xaxis=dict(
            tickformat="%b",
            showgrid=True,
            tickfont = {'size':16},
        ),
        title_x=0.5,
        title_font = {'size':20},
        yaxis=dict(
            tickfont = {'size':16},
        )
    )
    update_plotly_background(fig)
    
    return fig

################################################################################

# Player Distribution Plot
def get_distribution(player_list, person, df):
    def get_data(player):
        df_data_person = df.iloc[:,3:].T
        df_data_person = df_data_person[[player]].dropna()
        return df_data_person[player]

    hist_data = [get_data(person)]
    for player in player_list:
        hist_data += [get_data(player)]

    group_labels = [person] + player_list
    color_palette = ['blue', 'red', 'purple', 'orange', 'gray', 'brown', 'green', 'black', 'lime']

    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=color_palette)

    fig.update_layout(
        xaxis=dict(title='Profit/Loss', title_font=dict(size=16)),
        height=600,
    )
    update_plotly_background(fig)
    
    return fig

################################################################################

# Player History
def player_history(selected_player, df_raw):
    df_data_person = df.iloc[:,3:].T
    df_data_person.index = pd.to_datetime(df_data_person.index, dayfirst=True).strftime('%y%m%d') #convert to this format to sort the dates
    df_data_person = df_data_person.sort_index(ascending=False)
    df_data_person.index = pd.to_datetime(df_data_person.index, format='%y%m%d').strftime('%b-%d %Y')
    df_data_person = df_data_person[[selected_player]].dropna()
    df_data_person.reset_index(inplace=True)
    df_data_person.rename(columns={'index':'Date', selected_player:'Profit'}, inplace=True)

    # Convert the 'Profit' column to numeric and round to two decimal places
    df_data_person['Profit'] = df_data_person['Profit'].apply(lambda x: f'{x:.2f}')

    data = df_data_person.to_dict(orient='records')
    fig = dash_table.DataTable(
        # dont include ID here; it will prevent the page from refreshing
        columns=[
            {'name': 'Date', 'id': 'Date'},
            {'name': 'Profit', 'id': 'Profit', 'format': dash_table.Format.Format(precision=2)}
        ],
        data=data,
        style_table={'fontSize': 30},
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_data_conditional=[
            {
                'if': {'column_id': 'Profit', 'filter_query': '{Profit} < 0'},
                'color': 'red'
            },
            {
                'if': {'column_id': 'Profit', 'filter_query': '{Profit} > 0'},
                'color': '#006C5B'
            }
        ]
    )
    return fig

################################################################################

# Dash Board
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

'################################################################################'
# Declare server for Heroku deployment. Needed for Procfile.
server = app.server
'################################################################################'

# App layout
app.layout = html.Div([
    html.H1('5F Casino', style={'textAlign': 'center', 'fontSize': 70}),
    dcc.Dropdown(
        id='main-dropdown',
        options=[{'label': year, 'value': year} for year in yearlist],
        value=yearlist[0],  # Set the default value
        style={'width': '100%'}
    ),
    dcc.Tabs(
        id='tabs',
        value='personal',
        children=[
            dcc.Tab(label='Player Data', value='personal'),
            dcc.Tab(label='Table Data', value='global'),
        ],
        style={'font-size': '24px'},
    ),

    # Hidden div to store the DataFrame as JSON
    dcc.Store(id='df-store', data=df.to_json(orient='split')),
    html.Div(id='dummy-input', style={'display': 'none'}), # hidden layer for the dummy-input 
    html.Div(id='page-content'),
])

# Callback to switch between subpages
@app.callback(Output('page-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'personal':
        return html.Div([
            # Player core stats
            dcc.Dropdown(
                id='Player',
                options=[{'label': person, 'value': person} for person in namelist],
                value=namelist[0]  # Set the default value
            ),
            html.H2(id='player-name', children='Panos', style={'textAlign': 'center', 'fontSize': 50}),
            html.Div(id='current-profit'),

            # Session Progress line plot        
            dcc.Checklist(id='Comparison-Checklist', value=[], inline=True),
            html.H2("Session Progress", style={'text-align': 'center'}),
            html.Div(id='session-progress'),

            # Monthly Progress line plot
            html.H2("Monthly Progress", style={'text-align': 'center'}),
            html.Div(id='monthly-progress'),

            # Player Profit Distribution
            html.H2("All-Time Profit Distribution", style={'text-align': 'center'}),
            html.Div(id='player-profit-distribution'),

            # All-Time Player P/L
            html.H2("All-Time Player Data", style={'text-align': 'center'}),
            html.Div(id='player-history-table', style={'height':'600px', 'overflow': 'scroll', 'width':'40%', 'margin':'0 auto'}),
        ])
    elif tab == 'global':
        return html.Div([
            # Last game results
            dcc.Dropdown(
                id='row-dropdown',
                value=0,
                style={'width': '100%'}
            ),
            html.Div(id='results-table-container', style={'margin-bottom': '50px'}),

            # Year Stats table
            html.H2(id="selected-year-text", children='2024 Table Stats', style={'text-align': 'center'}),
            html.Div(id='year-table-stats', style={'width':'50%', 'margin':'0 auto'}),

            # All-Time Stats table
            html.H2("All-Time Table Stats", style={'text-align': 'center'}),
            html.Div(id='all-time-table-stats', style={'width':'50%', 'margin':'0 auto'}),

            # Rank Over Sessions line plot
            html.H2(id="rank-over-session-text", children='2024 Player Rankings', style={'text-align': 'center'}),
            html.Div(id='rank-over-sessions-line-plot'),   

            dcc.Checklist(
                id='Name-Checklist',
                options=[{'label': person, 'value': person} for person in namelist],
                value=namelist[:],
                labelStyle={'font-size': '24px',},
                inline=True,
            ),

            # Time Lapse plot
            html.H2("Time Lapse of Net Profit/Loss", style={'text-align': 'center'}),
            html.Div(id='time-lapse'),
        ])
    
# Callback to update the H1 header '<player name>'
@app.callback(Output('player-name', 'children'), Input('Player', 'value'))
def update_header_text(selected_person):
    return '{}'.format(selected_person)

# Callback to update the current profit/loss value
@app.callback(Output('current-profit', 'children'), [Input('Player', 'value'), Input('main-dropdown', 'value')])
def update_current_profit_loss_value(selected_person, selected_year):
    fig = Current_profit_loss(selected_person, selected_year, df)
    return dcc.Graph(figure=fig)

# Callback for updating the checklist
@app.callback(Output('Comparison-Checklist', 'options'), Input('Player', 'value'))
def update_checklist(selected_player):
    remaining_names = [name for name in namelist if name != selected_player] # namelist without the selected name from dcc.Dropdown
    checklist_options = [{'label': name, 'value': name} for name in remaining_names]
    return checklist_options

# Callback for setting the default value to an empty list
@app.callback(Output('Comparison-Checklist', 'value'), [Input('Player', 'value')])
def update_checklist_default_value(selected_player):
    return []

# Callback to update the progress per session chart based on the selected person from the dropdown and the player list from the checklist
@app.callback(Output('session-progress', 'children'), [Input('Comparison-Checklist', 'value'), Input('Player', 'value'), Input('main-dropdown', 'value')])
def update_progress_session_chart(selected_people_checklist, selected_person_dropdown, selected_year):
    fig = Progress_Session(selected_people_checklist, selected_person_dropdown, selected_year, df)
    return dcc.Graph(figure=fig)

# Callback to update the monthly progress chart based on the selected person    
@app.callback(Output('monthly-progress', 'children'), [Input('Comparison-Checklist', 'value'), Input('Player', 'value'), Input('main-dropdown', 'value')])
def update_progress_montly_chart(selected_people_checklist, selected_person_dropdown, selected_year):
    fig = Progress_Monthly(selected_people_checklist, selected_person_dropdown, selected_year, df)
    return dcc.Graph(figure=fig)

# Callback to update the monthly progress chart based on the selected person    
@app.callback(Output('player-profit-distribution', 'children'), [Input('Comparison-Checklist', 'value'), Input('Player', 'value')])
def update_player_disrtribution(selected_people_checklist, selected_person_dropdown):
    fig = get_distribution(selected_people_checklist, selected_person_dropdown, df)
    return dcc.Graph(figure=fig)

# Callback to update the player history based on the selected person  
@app.callback(Output('player-history-table', 'children'), Input('Player', 'value'))
def update_player_history(selected_person):
    fig = player_history(selected_person, df)
    return fig

################################################################################

# Callback for the game dropdown
@app.callback([Output('row-dropdown', 'options'), Output('row-dropdown', 'value')], [Input('main-dropdown', 'value')], [State('df-store', 'data')])
def update_game_dropdown(selected_year, df_json):

    # Convert the JSON data back to a DataFrame
    df = pd.read_json(df_json, orient='split')
    
    df, dftime, dftime2 = df_selected_year(df=df, selected_year=selected_year)
    updated_options = [{'label': '{1} - Session {0}'.format(i+1, pd.to_datetime(dftime2.index.values[i], format='%d/%m/%Y').strftime('%d-%b')), 'value': i} for i in range(dftime2.shape[0])]
    updated_value = dftime2.shape[0] - 1  # Default to the last game
    return updated_options, updated_value

# Callback to update the table based on the selected row
@app.callback(Output('results-table-container', 'children'), [Input('row-dropdown', 'value'), Input('main-dropdown', 'value')])
def update_table(selected_row, selected_year):
    fig = make_table(selected_row, selected_year, df)
    return fig

# Callback to update the current progress bar chart based on the selected people (from the checklist)
@app.callback(Output('bar-plot', 'children'), [Input('Name-Checklist', 'value'), Input('main-dropdown', 'value')])
def update_graph(selected_people, selected_year):
    fig = Status_bar(selected_people, selected_year, df)
    return dcc.Graph(figure=fig)

# Callback to update the current time lapse bar chart based on the selected people (from the checklist)
@app.callback(Output('time-lapse', 'children'), [Input('Name-Checklist', 'value'), Input('main-dropdown', 'value')])
def update_graph(selected_people, selected_year):
    fig = Time_Lapse(selected_people, selected_year, df)
    return dcc.Graph(figure=fig)

# Callback to update the H2 header '<year> Player Ranks'
@app.callback(Output('rank-over-session-text', 'children'), Input('main-dropdown', 'value'))
def update_header_text(selected_year):
    return '{} Player Rankings'.format(selected_year)   

# Callback to update the rank over sessions line plot 
@app.callback(Output('rank-over-sessions-line-plot', 'children'), Input('main-dropdown', 'value'))
def update_session_ranks(selected_year):
    fig = rank_over_session(selected_year, df)
    return dcc.Graph(figure=fig)     

# Callback to update the H2 header '<year> Table Stats'
@app.callback(Output('selected-year-text', 'children'), Input('main-dropdown', 'value'))
def update_header_text(selected_year):
    return '{} Table Stats'.format(selected_year)

# Callback to update the table stats based on the selected year  
@app.callback(Output('year-table-stats', 'children'), Input('main-dropdown', 'value'))
def update_yearly_table_stats(selected_year):
    fig = year_table_stats(selected_year, df)
    return fig

# Callback to update the all-time table stats
@app.callback(Output('all-time-table-stats', 'children'), Input('dummy-input', 'children'))
def update_all_time_table_stats(dummy_input):
    fig = all_time_table_stats(df)
    return fig

if __name__ == '__main__':
    app.run_server(debug=False, port=int(os.environ.get('PORT', 8050)))