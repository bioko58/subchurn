import pandas as pd
import re



######## GLOBAL DATA FILES: #########
subs_file = './data/subscriptions.txt'
library_file = './data/library.txt'
sessions_file =  './data/sessions.txt'
purchases_file = './data/purchases.txt'
outfile = './data/cleaned_data_2.txt'

data_pull_date = pd.to_datetime('2017-05-16') + pd.Timedelta('1d')  # date that files were obtained:



def get_billings_features(df):

    # reads Billing data file
    subs = pd.read_csv(subs_file, sep="\t")

    # drops columns w/no signal (same value for every row)
    subs = subs.drop(subs.columns[(subs.apply(lambda x: x.nunique(), axis=0) <= 1)].tolist(), axis=1)

    #renames for readability/consistency
    subs.rename(columns={'SUBS_START_DATE':'subs_start_date'}, inplace=True)

    # fixes larges end-dates that cause Pandas errors.  (non-churners have large "end-dates" which Pandas cannot handle. replaces these w/ "data pull-date" +1)
    subs.loc[(subs['subs_end_date'].apply(lambda x: int(re.search('(\d{4}) ', x).group(1))) >= 2019), 'subs_end_date'] = data_pull_date  # assumes:  if year >=2019, user must NOT have scheduled end to their sub

    # formats dates to datetime
    subs[['subs_start_date', 'subs_end_date']] = subs[['subs_start_date', 'subs_end_date']].apply(pd.to_datetime, axis=0)

    # calcs membership_length (datetime)
    subs['membership_length'] = subs.apply(lambda row: row['subs_end_date'] - row['subs_start_date'], axis=1)  #for subscribers, data pull-date is their 'end-date'
    # calcs membership_length (float)
    subs['membership_length_as_float'] = subs['membership_length'] / pd.Timedelta(days=1)

    # identify "rejoiners" -- users having > 1 subscription
    # ... note: includes free trial converters (later:  dont mark these, not 'true' rejoiners)
    # ... note: includes users w/ 2+ active subs (later: ID these, prob. system error)
    subs['is_rejoiner'] = subs.duplicated(['User_Account_Id'],keep=False)  # arg 'False' : Mark all duplicates as ``True``.


    # excludes annual subscribers
    if 'is_Annual_Sub' in subs.columns:   #only present in the large data-file (otherwise dropped due to no signal)
        subs = subs[subs['is_Annual_Sub']==0]

    # (following three commands excludes all rejoiners  (userids appearing > once))

    # gets userid and membership_length
    df = subs[subs['is_rejoiner']==False][['User_Account_Id', 'membership_length_as_float']].copy()

    # gets churn indicator (target variable)
    df['has_churned'] = subs[subs['is_rejoiner']==False]['is_Active'].apply(lambda x: abs(x-1))

    # gets country that user signedup from
    df['country_of_signup'] = subs[subs['is_rejoiner']==False]['Country_Code']

    # another option for handling rejoiners: only keep most recent membership
    # subs = pd.merge(subs, subs.groupby('User_Account_Id').agg({'subs_start_date':'max'}).reset_index(), how='inner',on=['User_Account_Id','subs_start_date'])

    return (df, subs)



def get_sessions_features(df, subs):

    # reads Sessions data file
    sessions = pd.read_csv(sessions_file, sep="\t")

    # formats to datetime
    sessions['Logon_Date'] = pd.to_datetime(sessions['Logon_Date'])

    # adds subscription start/end dates ... only for non-rejoiners!
    #sessions = pd.merge(sessions,subs[subs['is_rejoiner']==False][['User_Account_Id','subs_start_date','subs_end_date']],on='User_Account_Id',how='inner')

    # adds subscription start/end dates (and rejoiner status)
    sessions = pd.merge(sessions, subs[['User_Account_Id','subs_start_date','subs_end_date', 'is_rejoiner']],on='User_Account_Id',how='inner')

    # limits sessions to when User was subscribed at time of Logon.  <-- only works for non-rejoiners!  otherwise a user will have multple sets of start/end
    sessions = sessions[sessions.apply(lambda x: (x['Logon_Date'] >= x['subs_start_date']) & (x['Logon_Date'] <= x['subs_end_date']), axis=1)]

    # calcs num unique titles played, for each user
    temp = sessions.groupby(['User_Account_Id'])['game_id'].nunique().reset_index()
    temp.rename(columns={'game_id':'num_titles_played'}, inplace=True)
    df = pd.merge(df, temp, how='inner') #inner: only keep Ids w/some titles played at all.  left: keeps Ids even w/no titles played  (both limits to whats in DF - non-rejoiners?)

    # calcs total sessions, for each user
    temp = sessions['User_Account_Id'].value_counts().reset_index()
    temp.columns=(['User_Account_Id','ttl_sessions'])
    df = pd.merge(df, temp, how='inner')  #inner: only keep Ids w/some sessions. left:  keep Ids even w/no sessions

    return (df, sessions)



def get_sessions_library_features(df, sessions):

    ### library ###

    # reads library data file
    library = pd.read_csv(library_file, sep="\t")

    # formats to datetime.  errors=coerce replaces the '?'s w/ 'NaT' (null)
    for c in ['trial_start_date', 'trial_end_date', 'library_start_date', 'library_end_date']:
        library[c] = pd.to_datetime(library[c], errors='coerce')

    # replaces NaT dates with the 'max_date'.
    #   (data file inserts this 'max_date' for library_end_dates that hasn't been reached/defined yet. makes other columns consistent..)
    max_date = pd.to_datetime('2018-01-01')
    # sets trial_end_date to 'max_date' for on-going trials ('NaT').  (if trial-start, but null trial-end)
    library['trial_end_date'] = library.apply(lambda row: max_date if (pd.notnull(row['trial_start_date']) and pd.isnull(row['trial_end_date'])) else row['trial_end_date'], axis=1)
    # sets library_start_date to 'max_date' for on-going trials ('NaT') (no library-start, because no trial-end)
    library['library_start_date'] = library.apply(lambda row: max_date if (row['trial_end_date']==max_date and pd.isnull(row['library_start_date'])) else row['library_start_date'], axis=1)

    # renames GAME_ID to game_id (consistency across tables)
    library.rename(columns={'GAME_ID':'game_id'}, inplace=True)


    ### SESSSIONS-library ###

    # combines library and sessions info
    sessions = pd.merge(sessions,library, how="left", on="game_id")  #left - include details on games that ARENT in library.

    # determines if session was a Trial game
    sessions['is_Trial_Session'] = (sessions['Logon_Date'] <= sessions['trial_end_date']) & (sessions['Logon_Date'] >= sessions['trial_start_date'])

    # determines if session was a library game
    sessions['is_library_Session'] = (sessions['Logon_Date'] <= sessions['library_end_date']) & (sessions['Logon_Date'] >= sessions['library_start_date'])

    # determines if session was a other_Session  (game not a ZZZ offering)
    sessions['is_other_Session'] = pd.isnull(sessions['master_title'])


    ### features ###

    # calcs total trial_ and library_ sessions for each user
    temp = sessions.groupby(['User_Account_Id'])[['is_library_Session','is_Trial_Session','is_other_Session']].sum().reset_index()
    temp.columns=(['User_Account_Id','ttl_library_sessions', 'ttl_trial_sessions', 'ttl_other_sessions'])
    df = pd.merge(df, temp, how='left')     #left and inner same - earlier we subsetted DF to users w/at least 1 session

    # % of sessions that are library plays
    df['pct_library_sessions'] = df['ttl_library_sessions']/df['ttl_sessions']

    # % sessions that are trial plays
    df['pct_trial_sessions'] = df['ttl_trial_sessions']/df['ttl_sessions']


    # calcs ttl sessions per game - for each user..
    sessions_per_game = sessions.groupby(['User_Account_Id','game_id'])['Logon_Date'].count().reset_index()
    # was user actively engaged with this game? (>5 sessions)
    sessions_per_game['is_engaged'] = sessions_per_game['Logon_Date'] > 5
    # was user just trying this game, and didn't stick with it?  (<= 5 sessions)
    sessions_per_game['has_experimented'] = sessions_per_game['Logon_Date'] <= 5

    # ...joins new engagement metrics with DF
    temp = sessions_per_game.groupby('User_Account_Id')['is_engaged','has_experimented'].sum()
    temp = temp.reset_index()
    temp.columns = ['User_Account_Id','num_titles_engaged', 'num_titles_experimented']
    df = pd.merge(df, temp, how='left')     #left and inner same - earlier we subsetted DF to users w/at least 1 game

    return (df, library, sessions)



def get_purchases_features(df, sessions, library):

    # reads Purchases data file
    purchases = pd.read_csv(purchases_file, sep="\t")

    # calcs num transactions, for each type: FG (Full-Game), MTX (MicroTransaction), EXP (Expansion Pack)
    temp = purchases.groupby(['User_Account_Id','Content_Type'])['Units'].sum().unstack()
    temp = temp.fillna(0)
    temp = temp.add_suffix("_buys")
    temp = temp.reset_index()
    df = pd.merge(df, temp, how='inner')  #left includes users with NO purchases at all (including SUB) how does that happen? 47 of these #TODO

    # calcs total transactions (sum across all types)
    df['ttl_buys'] = df['EP_buys'] +df['FG_buys'] + df['MTX_buys'] + df['SUB_buys']

    # identifies if FG purchase was a Trial game
    trial_games = get_top_trials(library,sessions)
    #trial_games = library[pd.notnull(library['trial_start_date'])].sort_values('trial_start_date')['game_id'].tolist()   # lists all trial-games
    purchases['is_trial_purchase'] = ((purchases['game_id'].isin(trial_games)) & (purchases['Content_Type']=='FG') & (purchases['Units']>0))

    # identifies if FG purchases was a TOP-3 Trial game
    top_trial_games = get_top_trials(library, sessions, 3)
    #top_trial_games = sessions[sessions['game_id'].isin(trial_games)].groupby('game_id')['Logon_Date'].count().sort_values(ascending=False).index.tolist()[:3]  # top-3 popular trial-games (by num sessions)  -> [853013, 979004, 1151001]  #TODO confirm list
    purchases['is_top_trial_purchase'] = ((purchases['game_id'].isin(top_trial_games)) & (purchases['Content_Type']=='FG') & (purchases['Units']>0))

    # counts num of trial/top-trial purchases for each user
    temp = purchases.groupby('User_Account_Id')[['is_trial_purchase', 'is_top_trial_purchase']].sum().reset_index()
    temp.columns = ['User_Account_Id', 'num_trial_buys', 'num_top_trial_buys']
    df = pd.merge(df, temp, how='left')
    df[['num_trial_buys','num_top_trial_buys']] = df[['num_trial_buys','num_top_trial_buys']].fillna(0)  # inserts 0 instead of null, when user didn't buy any trial games

    return (df, purchases)



def get_top_trials(library, sessions, num=None):

    trial_games = library[pd.notnull(library['trial_start_date'])].sort_values('trial_start_date')['game_id'].tolist()

    if num==None:
        return trial_games
    else:
        top_X_trial_games = sessions[sessions['game_id'].isin(trial_games)].groupby('game_id')['Logon_Date'].count().sort_values(ascending=False).index.tolist()[:num]  # top-X popular trial-games (by num sessions)
        return top_X_trial_games
    # TODO: add a sanity check on the 'num' argument, and that library and sessions are OK



def get_first_mo_stats(df, sessions):


    # limits users to those > 1 month old
    df = df[df['membership_length_as_float'] >= 28]

    # limits session data to activity within 1st month
    sessions_firstmo = sessions[sessions['Logon_Date'] < (sessions['subs_start_date'] + pd.Timedelta(days=32))] #approx first month..

    # calcs num sessions in first month
    temp = pd.DataFrame(sessions_firstmo['User_Account_Id'].value_counts().reset_index())
    temp.columns=(['User_Account_Id','firstmo_ttl_sessions'])
    df = pd.merge(df, temp, how='left')
    df = df.fillna(0)  #<-- users with no sessions??  #TODO: can we specify a particular column, thats a little better

    # calcs num titles played in first month
    temp = pd.DataFrame(sessions_firstmo.groupby(['User_Account_Id'])['game_id'].nunique()).reset_index()
    temp.columns=(['User_Account_Id','firstmo_num_titles_played'])
    df = pd.merge(df, temp, how='left') #inner: only keep Ids w/some sessions.  left: keeps Ids even w/NO sessions
    df = df.fillna(0)

    # calcs num library and trial sessions in first month
    temp = sessions_firstmo.groupby(['User_Account_Id'])[['is_library_Session','is_Trial_Session','is_other_Session']].sum().reset_index()
    temp.columns=(['User_Account_Id','firstmo_ttl_library_sessions', 'firstmo_ttl_trial_sessions', 'firstmo_ttl_other_sessions'])
    df = pd.merge(df, temp, how='left')
    df = df.fillna(0)

    # calcs % library sessions in first month
    df['firstmo_pct_library_sessions'] = df['firstmo_ttl_library_sessions']/df['firstmo_ttl_sessions']

    # calcs % trial sessions in first month
    df['firstmo_pct_trial_sessions'] = df['firstmo_ttl_trial_sessions']/df['firstmo_ttl_sessions']

    return (df, sessions_firstmo)



def get_prev_2w_stats(df, sessions):

    # Definition: previous 2 weeks = the two weeks before users subscription end date (or, data pull-date for non-churners)


    # limits users to those > 2 weeks old
    df = df[df['membership_length_as_float'] >= 14]

    # get the session activity in the previous 2 weeks
    sessions_prev_2w = sessions[sessions['Logon_Date'] >= (sessions['subs_end_date'] - pd.Timedelta(2,'W'))]

    # calcs num sessions in previous 2 wks
    temp = pd.DataFrame(sessions_prev_2w['User_Account_Id'].value_counts().reset_index())
    temp.columns=(['User_Account_Id','prev2w_ttl_sessions'])
    df = pd.merge(df, temp, how='left')     #needs to be left - see below comment
    df['prev2w_ttl_sessions'].fillna(0, inplace=True)  #replaces nulls w/zeros, since some users have 0 sessions in final/prev two weeks

    # calcs num titles played in previous 2 wks
    temp = pd.DataFrame(sessions_prev_2w.groupby(['User_Account_Id'])['game_id'].nunique()).reset_index()
    temp.columns=(['User_Account_Id','prev2w_num_titles_played'])
    df = pd.merge(df, temp, how='left')     #needs to be left - see below comment
    df['prev2w_num_titles_played'].fillna(0, inplace=True)  #replaces nulls w/zeros, since some users play 0 titles in final/prev two weeks

    # calcs num trial_ and library_ sessions in previous 2 wks
    temp = sessions_prev_2w.groupby(['User_Account_Id'])[['is_library_Session','is_Trial_Session','is_other_Session']].sum().reset_index()
    temp.columns=(['User_Account_Id','prev2w_ttl_library_sessions', 'prev2w_ttl_trial_sessions', 'prev2w_ttl_other_sessions'])
    df = pd.merge(df, temp, how='left')     #needs to be left - see below comment
    df = df.fillna(0)  #TODO:  fix this so we only fillna on the specific columns..   #replaces nulls w/zeros, since some users have 0 sessions in final/prev two weeks

    # calcs % library sessions in previous 2 wks
    df['prev2w_pct_library_sessions'] = df['prev2w_ttl_library_sessions']/df['prev2w_ttl_sessions']
    #df = df.fillna(0)  #need this?

    # calcs % trial sessions in previous 2 wks
    df['prev2w_pct_trial_sessions'] = df['prev2w_ttl_trial_sessions']/df['prev2w_ttl_sessions']
    #df = df.fillna(0)  #need this?


    return (df, sessions_prev_2w)



def dummify_country(df):

    #TODO: put this into the get_billings
    df['is_RU'] = (df['country_of_signup']=='RU')
    df['is_DE'] = (df['country_of_signup']=='DE')

    return df



def add_signup_near_trial(df, library, sessions, subs):

    trial_games = get_top_trials(library, sessions)
    top_trial_games = get_top_trials(library, sessions, 3)

    # indicates whether signup happened near major trial (# 1 week before, 3 weeks after)
    trial_start_dates = library[library['game_id'].isin(top_trial_games)].sort_values('trial_start_date')['trial_start_date']
    trial_start_dates = (trial_start_dates - pd.Timedelta(1,'W')).tolist()
    #trial_end_dates = library[library['game_id'].isin(top_trial_games)].sort_values('trial_start_date')['trial_end_date'].tolist()
    subs['is_signup_near_top_trial'] = subs['subs_start_date'].apply(lambda date: any([(t - pd.Timedelta(1,'W')) < date < (t + pd.Timedelta(2,'W'))  for t in trial_start_dates]))

    # indicates whether signup happened near any trial (# 1 week before, 3 weeks after)
    trial_start_dates = library[library['game_id'].isin(trial_games)].sort_values('trial_start_date')['trial_start_date']
    trial_start_dates = (trial_start_dates - pd.Timedelta(1,'W')).tolist()
    #trial_end_dates = library[library['game_id'].isin(trial_games)].sort_values('trial_start_date')['trial_end_date'].tolist()
    subs['is_signup_near_trial'] = subs['subs_start_date'].apply(lambda date: any([(t - pd.Timedelta(1,'W')) < date < (t + pd.Timedelta(2,'W'))  for t in trial_start_dates]))

    df = pd.merge(df, subs[['User_Account_Id', 'is_signup_near_top_trial', 'is_signup_near_trial']], how='left')

    return df



def output_features_csv(df):
    df.to_csv(outfile, index=False, sep='\t')



def pickle_tables():
    # saves DFs as pickles

    # experimental code....
    # not an especially safe /conventional way to go about it, but just learning whats possible.
    # easier to just spell-out the exact objects and filenames, but anyhow:


    import inspect
    local_vars = inspect.currentframe().f_back.f_locals.items()

    for var_name, var_val in local_vars:
        if type(var_val)==pd.core.frame.DataFrame:
            var_val.to_pickle('pickles2/'+var_name+'.pkl')



if __name__=='__main__':

    # collects useful or engineered features, for use during modeling
    df = pd.DataFrame()

    # reads data files, transforms and engineers features
    df, subs = get_billings_features(df)
    df, sessions = get_sessions_features(df, subs)
    df, library, sessions = get_sessions_library_features(df, sessions)
    df, purchases = get_purchases_features(df, sessions, library)
    df, sessions_first_mo = get_first_mo_stats(df, sessions)
    df, sessions_prev_2w = get_prev_2w_stats(df, sessions)
    df = dummify_country(df)
    df = add_signup_near_trial(df, library, sessions, subs)


    # outputs cleaned data into CSV
    output_features_csv(df)

    # saves pandas tables as pickles for fast loading as needed
    pickle_tables()
