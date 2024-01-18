# Poker-Game-Analysis-using-Plotly-Dash
Analysing profit/loss data from a series of poker games, Visualising the results of the analysis using Plotly Dash and deploying the model using Heroku!

Check the app [here!](https://casino-5f-958d8779bafe.herokuapp.com/)

## <u>Overview</u>
The main objective of this project is to develop a dynamic interface, utilizing Plotly Dash, that allows my friends to track their progress and that of fellow poker players through effective analytics deployed in the dashboard.

Due to the nature of the dataset, and the scope of this project the analysis is mostly exploratory and focused on making insightful visualisations. In this project, plotly dash is extensively utilized to generate various analytics and metrics from the dataset. Most of the code is therefore concerned with visualisation of data rather than analysis. In the notebook, I use various plotting Plotly tools to visualise the results of my analysis such as Gauge, line and bar charts. I also use an interractive bar chart from Plotly Express and a table from dash_tables. The dashboard created is very interractive, allowing the user to check for the performance of a specific player, or look over the results of a particular session.

## <u>Motivation for the Project</u>
Every now and then, I gather with my friends and play poker. On January 2023, I had an interesting idea and decided to start logging the profit/loss of each player after each session. This was an easy way to track the progress of each player. At the beggining of the year, I exclusively utilized Excel's built-in sum() and average() functions for the dataset analysis. However, as the year progressed I had accumulated sufficient data and decided to showcase my expertise in Exploratory Data Analysis (EDA) and leverage my analytical skills to create something better ... an interactive dashboard!

## <u>About the dataset</u>
Each session consists of usually 8 and in some cases 9 players. However, most of the sessions are played by the same 6 people. Why you ask? Because we are all game-commited and we want to be #1 on the stats. Therefore, in each session we usually invite 2-3 guests to fill the table. These guests rotate and most of them have only played a few games in total (<5). Analysis is therefore focused on the 6 of us and those ones that have played enough games (cuttoff number of games is adjustable in the code).

Sample of the excel table used for the analysis:

|      |    NET    |    PPG    |  TABLES   |  14-Jan   |  21-Jan   |   28-Jan  |   10-Feb  | 
|-----:|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|Panos|   43.85   |    1.9    |     23    |-0.95 |2.6 |-3.45 |-10.2 |
|Ashish|   94.80   |    3.8    |     25    |-5 |24.6 |-24 |28.7 |
|Chris|   82.00   |    3.6    |     23    |37.35 |1.15 |13.35 |7.8 |

Notice I have used the excel functions for the first 3 columns of the dataset. Dont worry if your excel or csv file data dont have these columns, it should only take a few lines of code to reach the format of my table. 

## <u>Deployment</u>
- Dashboard was deployed using Heroku through github. Automatic deployment was configured which updates the server after pushing new code to github.
- To deploy sucessfully you need your requirements.txt, runtime.txt and Procfile (no extension) in your top level directory.
- Make sure that the python version specified in runtime.txt is supported from Heroku. Check the supported python versions [here](https://devcenter.heroku.com/articles/python-support)
- Your procfile should specify what you are trying to achieve with your code. In my case, I wanted to deploy my app.py file to the web and run it on a server. Therefore the content on my Procfile was:
```python
web: gunicorn app:server
```
- Finally make sure to keep the following lines in your python file to be able to deploy to Heroku:
```python
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

# load the data
df = load_data("data.csv")
```
And after initiating the app using ```dash.Dash(__name__)```
```python
# Declare server for Heroku deployment. Needed for Procfile.
server = app.server
```

Files in the .gitignore include:
- data.xslx excel file where I write the new data after a poker session
- test.ipynb notebook that I use to test that the new code works before commiting to github
- data_update.ipynb that converts the excel to a csv file, data.csv, saving it in the directory of the original csv file and overwriting it (since they have the same name).
```python
import pandas as pd
df = pd.read_excel("data.xlsx", index_col=0) # read from excel file and use the first column as the index of the dataframe
df.to_csv("data/data.csv") # convert to csv and upload it to the relevant directory
```

## <u>The next steps</u>
- Read the cards dealt using MFRC522 RFID readers connected to the Raspberry Pi 4.
- Read the board using object detection techniques. An IP camera connected to the RPi 4 can be used for that purpose. 
- Once the poker card tracking system is implemented, supplementary information will be gathered, such as the cards held by each player, ATS, VPIP, C-Bet% and others. With access to this data, we can assess how closely players played to the game theory optimal (GTO).

## <u>Inspiration for your own Project</u>
If you're passionate about poker gatherings with friends, maybe its a good time to start tracking and crunching numbers! My personalized dashboard will indefinitely add a dynamic element to the poker experience and also offer insights into players' performance trends, fostering a more engaging and data-driven approach to the game. You will probably loose some poker games, but remember, if you are the poker maestro among your pals you will come on top on the long run! Best of luck.
