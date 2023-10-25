# Poker-League-Game-Analysis
Analysing profit/loss statistics from home poker games

## <u>Overview</u>
Once every two weeks or so, I gather with my friends and play poker. On January 2023, I had an interesting idea and decided to start logging the profit/loss of each player after each session. This was an easy way to track the progress of each player, but we decided to make a bit more interesting. We all agreed to announce the prize for the player with the highest profit/session.

After 25 sessions, I decided to see what anaylics I can extract from the data I have been loging throughout the year and present the findings to my friends! 

## <u>About the dataset</u>
The 'core' players in most of the games are a total of 6 people (me and 5 of my friends). The 6 of us have played the majority of the games, so the analysis is based on us. The table has a capacity of 8 people so in each session we will invite some additional players. Those players have played at most 20% of the games, so in the analysis I group their data.

Sample of the excel table used for the analysis:

|      |    NET    |    PPG    |  TABLES   |  14-Jan   |  21-Jan   |   28-Jan  |   10-Feb  | 
|-----:|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|Panos|   43.85   |    1.9    |     23    |-0.95 |2.6 |-3.45 |-10.2 |
|Ashish|   94.80   |    3.8    |     25    |-5 |24.6 |-24 |28.7 |
|Chris|   82.00   |    3.6    |     23    |37.35 |1.15 |13.35 |7.8 |

## <u>Data analysis</u>
Due to the nature of the dataset, the analysis is mostly exploratory and focused on making insightful visualisations. Through the dashboard, the 6 main players will be able to view their personal progress and compare it to the other main players. 

## <u>The next step</u>
Once the RFID tracking system is established, it will begin to collect additional data, including information about the cards held by each player and the way these hands were played. After obtaining these data, we can find how close players play to GTO (game theory optimal) and make a 'skill' statistic for the main players. 
