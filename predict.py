import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, mean_squared_error, r2_score, f1_score
import compare_models as gd
from sklearn.metrics import confusion_matrix
import numpy as np
import statsmodels.api as sm



def get_scores(model, X_test, y_test):

    y_pred = np.array(model.predict(X_test.astype(float)) > threshold).astype(int)

    # prints model scores for a variety of metrics
    metrics = [precision_score, recall_score, roc_auc_score, f1_score]
    print "Test Scores for Logistic Regression model:".format(model.__class__.__name__)
    print "------------------------------"
    for metric in metrics:
        print metric.__name__, "=", metric(y_test, y_pred)

    # reports precision, recall, f1, support for predicting each Target class
    print "\n\nClassification Report for {} model:".format(model.__class__.__name__)
    print "------------------------------"
    print classification_report(y_test, y_pred, target_names=['not-churned', 'churned']) #, target_names=target_names)

    # prints confusion matrix
    print "\n\nConfusion Matrix:"
    print "------------------------------"
    print confusion_matrix(y_test, y_pred),"\n"
    print np.array([['TN', 'FP'], ['FN', 'TP']])



def get_data(datafile):

    # reads cleaned data file
    df = pd.read_csv(datafile, sep='\t')

    # gets target variable
    target = 'has_churned'
    y = df.pop(target).values

    # gets features (predictor variables)
    features = df.columns.tolist()

    # removes features that don't belong in the model
    #collinear_cols = []
    collinear_cols = ['ttl_sessions','pct_library_sessions','firstmo_ttl_sessions','firstmo_pct_library_sessions','prev2w_ttl_sessions','prev2w_pct_library_sessions']
    unnecessary_cols = ['User_Account_Id', 'game_id', 'country_of_signup', 'SUB_buys', 'ttl_buys', 'membership_length_as_float']
    for x in (collinear_cols + unnecessary_cols):
        if x in features:
            features.remove(x)


    X = df[features].values

    return X,y,features



def build_log_model(X_train,y_train,features):

    logit = sm.Logit(y_train.astype(float), pd.DataFrame(X_train,columns=features).astype(float))
    results = logit.fit(optimizer='newton')
    return results



if __name__ == '__main__':

    datafile = './data/cleaned_data.txt'

    # gets X and Y variables
    X, y, features = get_data(datafile)

    # performs test-train split of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=58)

    # compares CV-scores of Logistic Regression VS. various non-parametric models (confirming LR is a reasonably competitive/comparable choice)
    gd.compare_models(X_train, y_train)

    # (after confirming above) builds logistic model using statsmodels, to print the summary
    results = build_log_model(X_train, y_train, features)
    print results.summary()

    # scores logistic model on the Test data
    get_scores(results, X_test, y_test)
