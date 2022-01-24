import sys
from predictor import predictRuns
batting_team = input("Enter name of batting team: ")
bowling_team=input("Enter name of bowling team: ")
runs = predictRuns(batting_team, bowling_team)
print('Predicted Runs: ',runs)
