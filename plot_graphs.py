import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from lifelines import KaplanMeierFitter



def get_dataframes():

    # pickle file locations
    pkl_file_dir = './pickles2/'
    df_pkl = pkl_file_dir + 'df.pkl'
    subs_pkl = pkl_file_dir + 'subs.pkl'
    sessions_pkl = pkl_file_dir + 'sessions.pkl'

    # reads pickle files, produced by clean_data.py
    try:
        df = pd.read_pickle(df_pkl)
        subs = pd.read_pickle(subs_pkl)
        sessions = pd.read_pickle(sessions_pkl)
    except IOError:
        print "IOError: Couldn't open pickle files within {}.  Has clean_data.py been run first, to produce the pickle files?".format(pkl_file_dir)
        sys.exit(exc.errno)

    return (df,subs,sessions)



def get_gaptime(subs):

    ## calculates the gap-time (length of time between a rejoiners' membership-end, and their next membership-start

    #TODO:  currently this includes free-trial converters.  need to cleanly ID and exclude these

    # selects the rejoiners, orders them by ID and Start-Date
    gaptime = subs[subs['is_rejoiner']==True].sort_values(['User_Account_Id','subs_start_date'])

    gaptime['prev_start'] = gaptime['subs_start_date'].shift(1)

    # adds next rejoin date info to each row
    gaptime['next_start'] = gaptime['subs_start_date'].shift(-1)

    # calculates time delta between each sub end-date, and the next sub start date (only when the next userID is the same)
    gaptime['next_user_is_same'] = (gaptime['User_Account_Id']==gaptime['User_Account_Id'].shift(-1))
    gaptime['gap_time'] = gaptime.apply(lambda x: x['next_start']-x['subs_end_date'] if x['next_user_is_same'] else np.NaN, axis=1)
    gaptime['gap_time_as_float'] = gaptime['gap_time'] / pd.Timedelta(days=1)

    return gaptime



def plot_membership_length_histogram_mos(subs,save=False):

    ## membership_length histogram (monthly buckets ONLY)

    plt.figure()
    mos = np.arange(0,390,32)   # not precisely per month, but a 31-day chunk for simplicity sake..
    plt.hist(subs.membership_length_as_float, bins=mos, alpha=.5)
    #plt.xticks(mos)  #<- this is more accurate
    plt.xticks(np.arange(0,370,30))
    plt.ylabel('count')
    plt.xlabel('membership length (Days)')
    plt.title('Distribution of Membership-Lengths')
    plt.draw()
    if save:
        plt.savefig('membership_length_histogram__months.png')



def plot_membership_length_histogram_wks_and_mos(subs, save=False):

    ## membership length histogram (with wks AND months buckets overlaid)

    plt.figure()
    mos = np.arange(0,390,32)   # not precisely per month, but a 31-day chunk for simplicity sake..
    wks = np.arange(0,390,8)
    plt.hist(subs.membership_length_as_float, bins=mos, alpha=.5)
    plt.hist(subs.membership_length_as_float, bins=wks)
    #plt.xticks(mos)  #<- this is more accurate
    plt.xticks(np.arange(0,370,30))  #<-- a little deceptive, but less Qs..
    plt.ylabel('count')
    plt.xlabel('membership length (Days)')
    plt.title('Distribution of Membership-Lengths')
    plt.draw()
    if save:
        plt.savefig('membership_length_histogram__months_weeks.png')



def plot_membership_survival_curve(df, save=False):

    plt.figure()
    kmf = KaplanMeierFitter()
    kmf.fit(df['membership_length_as_float'], event_observed=df['has_churned'])
    kmf.plot()
    plt.title('Membership Surival Curve')
    plt.ylabel('Probability of Retaining Membership')
    plt.xlabel('Days since Subscribing')
    plt.xticks(np.arange(min(df['membership_length_as_float']), max(df['membership_length_as_float'])+1, 30))
    plt.legend().set_visible(False)
    plt.tight_layout()
    plt.draw()
    if save:
        plt.savefig('survival_curve.png')


def plot_gaptime_less_1mo(gaptime, save=False):

    ## gaptime histogram - focuses only on gaptimes less than 1 month (in days)

    plt.figure()
    days = np.arange(0,32,1)
    plt.hist(gaptime[ (gaptime['gap_time_as_float'] < 32) & (gaptime['gap_time_as_float'] >= 0) ]['gap_time_as_float'], bins=days)
    plt.title('Distribution of "Gap Times" Shorter Than 1 Month')
    plt.ylabel('Count')
    plt.xlabel('Gaptime (in days)')
    plt.xticks(np.arange(0, 32, 5))
    plt.tight_layout()
    plt.draw()
    if save:
        plt.savefig('gaptime_dist_less_1mo.png')



def plot_gaptime(gaptime, save=False):

    #gaptime histogram

    plt.figure()
    #wks = np.arange(0,390,8)
    mos = np.arange(0,390,32)
    plt.hist(gaptime[ gaptime['gap_time_as_float'] >= 0 ]['gap_time_as_float'], bins=mos)
    plt.title('Distribution of "Gap Times"')
    plt.ylabel('Count')
    plt.xlabel('Gaptime (in days)')
    plt.xticks(np.arange(0, max(gaptime['gap_time_as_float'])+1, 30))
    plt.tight_layout()
    plt.draw()
    if save:
        plt.savefig('gaptime_dist.png')



def plot_signup_vs_trial(subs, save=False):

    # plot histograms of signups by date, overlaid with trial periods

    plt.figure()
    subs.groupby([subs.subs_start_date.dt.year, subs.subs_start_date.dt.month])['subs_start_date'].count().plot(kind='bar', width=.85, label='')
    xtick_labels = subs.groupby([subs.subs_start_date.dt.year, subs.subs_start_date.dt.month])['subs_start_date'].count().index.tolist()
    xtick_labels = map(lambda x: str(x[1]).zfill(2) + '-' + str(x[0]) , xtick_labels)
    plt.xticks(np.arange(len(xtick_labels)), xtick_labels, rotation=65)
    plt.plot(9,350,'yo',label='Game-B')
    plt.plot(8,350,'bo',label='Game-F')
    plt.plot(14,350,'mo',label='Game-M')
    plt.legend()
    plt.ylabel('num. of new subscriptions')
    plt.xlabel('month')
    plt.title('Num. of New Subscriptions per Month')
    plt.tight_layout()
    plt.draw()
    if save:
        plt.savefig('signups_per_month_w_trials.png')



def plot_game_popularity_by_sessions(sessions, save=False):

    ## game popularity by ttl sessions

    plt.figure()
    # gets ttl_session count for each game
    games_sessionct = sessions.groupby('master_title')['Logon_Date'].count().sort_values(ascending=False)
    fig,ax = plt.subplots()
    y = np.round(games_sessionct.values / float(np.sum(games_sessionct.values)) * 100)[:-2]  #(coverts the counts to % of total) (HACK: the last two are just zeros...)
    x = range(0,len(y))
    ax.bar(x,y)

    xtick_labels = games_sessionct.index.values
    xtick_labels = map(lambda x: x[:10] + '...' + x[-5:] if len(x)>17 else x, xtick_labels)  #trims the long titles
    plt.xticks(np.arange(len(xtick_labels)), xtick_labels, rotation=90)
    ax.tick_params(axis='x', which='major', labelsize=8)  #shrinks the font size a bit

    plt.ylabel('% of all sessions')
    #plt.xlabel('game')
    plt.title('Game Popularity by % of Sessions')
    plt.tight_layout()
    plt.draw()
    if save:
        plt.savefig('game_popularity_sessions.png')



def plot_game_popularity_by_users(sessions, save=False):

    ## game popularity by num. unique users (%)

    plt.figure()
    # gets unique user-count (%) for each game
    games_userct = sessions.groupby('master_title')['User_Account_Id'].nunique().sort_values(ascending=False)
    fig,ax = plt.subplots()
    y = np.round(games_userct.values / float(np.sum(games_userct.values)) * 100)[:-2]  #(coverts the counts to % of total) (HACK: just trim it to match above...) equiv to [:15]
    x = range(0,len(y))
    ax.bar(x,y)

    xtick_labels = games_userct.index.values
    xtick_labels = map(lambda x: x[:10] + '...' + x[-5:] if len(x)>17 else x, xtick_labels)  #trims the long titles
    plt.xticks(np.arange(len(xtick_labels)), xtick_labels, rotation=90)
    ax.tick_params(axis='x', which='major', labelsize=8)  #shrinks the font size a bit
    plt.ylabel('% of all users')
    #plt.xlabel('game')  #looks better w/o the x-label
    plt.title('Game Popularity by % of Num. Unique Users')
    plt.tight_layout()
    plt.draw()
    if save:
        plt.savefig('game_popularity_users.png')


if __name__=='__main__':

    #loads dataframes from the pickles produced in clean_data.py
    df, subs, sessions = get_dataframes()

    #calculates an additional 'gaptime' DF
    gaptime = get_gaptime(subs)

    plot_membership_length_histogram_mos(subs)
    plot_membership_length_histogram_wks_and_mos(subs)
    plot_membership_survival_curve(df)
    plot_gaptime_less_1mo(gaptime)
    plot_signup_vs_trial(subs)
    plot_game_popularity_by_sessions(sessions)
    plot_game_popularity_by_users(sessions)
