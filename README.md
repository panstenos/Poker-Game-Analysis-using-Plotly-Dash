# Poker-Game-Analysis-using-Plotly-Dash
Analysing profit/loss data from a series of poker games, Visualising the results of the analysis using Plotly Dash and deploying the model using Heroku!

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

## <u>The next steps</u>
- Dataset will be uploaded to an SQL database. A webpage will be set so every player will be able to have access to the analytics generated. The backend will fetch data from the database and operate on a server to uphold the dynamic nature of the analytics.
- Once the poker card tracking system is implemented, supplementary information will be gathered, such as the cards held by each player, ATS, VPIP, C-Bet% and others. With access to this data, we can assess how closely players played to the game theory optimal (GTO).

## <u>Inspiration for your own Project</u>
If you're passionate about poker gatherings with friends, maybe its a good time to start tracking and crunching numbers! My personalized dashboard will indefinitely add a dynamic element to the poker experience and also offer insights into players' performance trends, fostering a more engaging and data-driven approach to the game. You will probably loose some poker games, but remember, if you are the poker maestro among your pals you will come on top on the long run! Best of luck.
