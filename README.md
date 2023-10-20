# Poker-League-Game-Analysis
Analysing profit/loss statistics from home poker games with friends

## <u>Overview</u>
Once every two weeks or so, I gather with my friends and play poker. On January 2023, I had an interesting idea and decided to start logging the profit/loss of each player after each session. This was an easy way to track the progress of each player, but we decided to make a bit more interesting. We all agreed to announce the prize for the player with the highest profit/session.

After 25 sessions, I decided to see what anaylics I can extract from the data I have been loging throughout the year and present the findings to my friends! 

## <u>About the dataset</u>
The 'core' players in most of the games are a total of 6 people (me and 5 of my friends). The 6 of us have played the majority of the games, so the analysis is based on us. The table has a capacity of 8 people so in each session, we will invite some 'complementary' players. Those players have played maximum 20% of the games, so in the analysis I group them. 

The excel Data:

Column1: Player names

Column2: Net profit/loss of players (use the excel sum function)

Column3: Profit Per Game (Divide column2 by the number of tables played)

Column4: Number of Tables Played (use the excel count function)

Rest of the: 

  Column head: Session date
  
  Column content: Session profit (or loss), if player did not attend the session leave blank so is shows as NaN in pandas
