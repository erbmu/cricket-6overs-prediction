def predictRuns(batting_team, bowling_team):
    #importing libraries
    import pandas as pd
    import numpy as np
    # Read the csv file
    df=pd.read_csv('all_matches.csv')

    # Calculating total after 6 overs of every match by creating an entirely new dataframe without unnecesary columns
    df['total']=df['runs_off_bat']+df['extras']

    xdf=df[(df['ball'] < 6.1) & (df['innings'] < 3)]

    prev=str(xdf.loc[0,'match_id']) + xdf.loc[0,'batting_team']+xdf.loc[0,'bowling_team']
    nRuns=0    
    record = []
    j=0
    for i in xdf.index:
        value = str(xdf.loc[i,'match_id']) + xdf.loc[i,'batting_team']+xdf.loc[i,'bowling_team']

        j=i
        if value == prev:
            nRuns=nRuns + xdf.loc[i,'total']
            pMatchId = xdf.loc[i,'match_id']
            pBatTeam = xdf.loc[i,'batting_team']
            pBowlTeam = xdf.loc[i,'bowling_team']
            pStartDate = xdf.loc[i,'start_date']
            pInnings = xdf.loc[i,'innings']
        else:

            prev = value
            rec = pMatchId, pBatTeam, pBowlTeam,pStartDate, pInnings, nRuns
            record.append(rec)
            nRuns=xdf.loc[i,'total']

    rec = xdf.loc[j,'match_id'],xdf.loc[j,'batting_team'],xdf.loc[j,'bowling_team'],xdf.loc[j,'start_date'],xdf.loc[j,'innings'], nRuns
    record.append(rec)

    #df1 will contain 6 over run summary for each team by match id

    df1 = pd.DataFrame(record, columns=['match_id', 'batting_team','bowling_team','start_date','innings','total'])

    # Keeping only consistent teams
    consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                        'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                        'Delhi Daredevils', 'Sunrisers Hyderabad']
    df1=df1[(df1['batting_team'].isin(consistent_teams)) & (df1['bowling_team'].isin(consistent_teams))]

    # Converting the column 'date' from string into datetime object
    from datetime import datetime
    df1['start_date'] = df1['start_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

    # --- Data Preprocessing ---
    # Converting categorical features using OneHotEncoding method
    encoded_df1 = pd.get_dummies(data=df1, columns=['batting_team', 'bowling_team'])

    encoded_df1 = encoded_df1[['start_date', 'batting_team_Chennai Super Kings', 'batting_team_Delhi Daredevils', 'batting_team_Kings XI Punjab',
                  'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals',
                  'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad',
                  'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Daredevils', 'bowling_team_Kings XI Punjab',
                  'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals',
                  'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad',
                  'total']]
     
    # Splitting the data into train and test set
    X_train = encoded_df1.drop(labels='total', axis=1)[encoded_df1['start_date'].dt.year <= 2016]
    X_test = encoded_df1.drop(labels='total', axis=1)[encoded_df1['start_date'].dt.year >= 2017]

    y_train = encoded_df1[encoded_df1['start_date'].dt.year <= 2016]['total'].values
    y_test = encoded_df1[encoded_df1['start_date'].dt.year >= 2017]['total'].values

    # Removing the 'date' column
    X_train.drop(labels='start_date', axis=True, inplace=True)
    X_test.drop(labels='start_date', axis=True, inplace=True)

    # --- Model Building ---
    # Different Regression Models:
    # 1.) Linear Regression Model
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    # Ridge Regression
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    ridge=Ridge()
    parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
    ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
    ridge_regressor.fit(X_train,y_train)
    prediction=ridge_regressor.predict(X_test)

    # Random Forest Regression
    from sklearn.ensemble import RandomForestRegressor
    lin_regressor = RandomForestRegressor(n_estimators=100,max_features=None)
    lin_regressor.fit(X_train,y_train)

    # Reading the input
    #inputdata=pd.read_csv(testinput)
    #batting_team=inputdata.loc[0,'batting_team']
    #bowling_team=inputdata.loc[0,'bowling_team']
    #batting_team=input("Enter name of batting team: ")
    #bowling_team=input("Enter name of bowling team: ")

    temp_array = list()
    if batting_team == 'Chennai Super Kings':
        temp_array = temp_array + [1,0,0,0,0,0,0,0]
    elif batting_team == 'Delhi Daredevils' or batting_team == 'Delhi Capitals':
        temp_array = temp_array + [0,1,0,0,0,0,0,0]
    elif batting_team == 'Kings XI Punjab' or batting_team == 'Punjab Kings':
        temp_array = temp_array + [0,0,1,0,0,0,0,0]
    elif batting_team == 'Kolkata Knight Riders':
        temp_array = temp_array + [0,0,0,1,0,0,0,0]
    elif batting_team == 'Mumbai Indians':
        temp_array = temp_array + [0,0,0,0,1,0,0,0]
    elif batting_team == 'Rajasthan Royals':
        temp_array = temp_array + [0,0,0,0,0,1,0,0]
    elif batting_team == 'Royal Challengers Bangalore':
        temp_array = temp_array + [0,0,0,0,0,0,1,0]
    elif batting_team == 'Sunrisers Hyderabad':
        temp_array = temp_array + [0,0,0,0,0,0,0,1]
                
                
    if bowling_team == 'Chennai Super Kings':
        temp_array = temp_array + [1,0,0,0,0,0,0,0]
    elif bowling_team == 'Delhi Daredevils' or bowling_team == 'Delhi Capitals':
        temp_array = temp_array + [0,1,0,0,0,0,0,0]
    elif bowling_team == 'Kings XI Punjab' or bowling_team == 'Punjab Kings':
        temp_array = temp_array + [0,0,1,0,0,0,0,0]
    elif bowling_team == 'Kolkata Knight Riders':
        temp_array = temp_array + [0,0,0,1,0,0,0,0]
    elif bowling_team == 'Mumbai Indians':
        temp_array = temp_array + [0,0,0,0,1,0,0,0]
    elif bowling_team == 'Rajasthan Royals':
        temp_array = temp_array + [0,0,0,0,0,1,0,0]
    elif bowling_team == 'Royal Challengers Bangalore':
        temp_array = temp_array + [0,0,0,0,0,0,1,0]
    elif bowling_team == 'Sunrisers Hyderabad':
        temp_array = temp_array + [0,0,0,0,0,0,0,1]
    data = np.array([temp_array])
    my_prediction = int(lin_regressor.predict(data)[0])
    return my_prediction 

