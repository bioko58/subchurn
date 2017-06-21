import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt



def compare_models(X_train, y_train):

    log_params = {
        'class_weight':['balanced', None, {1:.6, 0:.4}, {1:.75, 0:.25}],
        'penalty':['l2', 'l1'],
        'C':[1, 0.2, 0.5, 0.75, 1.25, 1.5] }

    lr2_params = {'solver':['newton-cg'], 'C':[1e8], 'fit_intercept':[False]}

    gboost_params = {
        'learning_rate':[0.10, 0.07, 0.13],
        'n_estimators':[100, 50, 500],
        'max_depth':[3, 2, 6] }

    svm_params = { 'kernel':['rbf'], 'C':[.5, 1, 1.5] }

    rfc_params = {'n_estimators':[50,200,500], 'max_features':[None,4,7], 'min_samples_leaf':[3, 8]}

    dtree_params = { 'max_features':[None,4,7], 'min_samples_leaf':[3, 8] }

    hyperparams = [log_params, lr2_params, gboost_params, svm_params, rfc_params, dtree_params]  #hyperparams to use
    models = [LogisticRegression(), LogisticRegression(), GradientBoostingClassifier(), SVC(), RandomForestClassifier(), DecisionTreeClassifier()]  #models to compare


    # ensures all models run on the same splits, by defining a consistent set of splits to use
    kf = KFold(n_splits=5, shuffle=False)

    #runs grid-search for each model, over it's hyperparam space
    scoring_metric = 'roc_auc'
    gs_results = []
    results = {}
    for model, hyperparam in zip(models, hyperparams):

        print "starting grid search for ",model.__class__.__name__,"..."
        gs = GridSearchCV(model, hyperparam, cv=kf, n_jobs=1 ,scoring=scoring_metric)
        #gs = GridSearchCV(model, hyperparam, n_jobs=-1, scoring=scoring_metric)  # runs faster (parallel), but doesnt use the same splits for each model
        gs.fit(X_train, y_train)
        gs_results.append(gs)


    #sorts the grid-search results by score (descending)
    scores = [gs.best_score_ for gs in gs_results]
    indices = np.argsort(scores)[::-1]
    gs_results = np.array(gs_results)[indices].tolist()


    print "\nGrid Search results:"
    print "------------------------------"
    for i,gs in enumerate(gs_results, start=1):
        print "{}. {}".format(i,gs.best_estimator_.__class__.__name__)
        print "  params:  ",gs.best_params_
        print ' ',gs.get_params()['scoring'], "score:", gs.best_score_,"\n"
    print ""


    # finds the Logistic model from the grid-search results
    # logit = next(gs.best_estimator_ for gs in gs_results if gs.best_estimator_.__class__.__name__ == 'LogisticRegression')


'''
BELOW IS NOT NEEDED ANYMORE, SINCE USING LOGISTIC REGRESSION RATHER THAN RANDOM FOREST


def report_feat_importance(rfc, X, features):

    importances = rfc.feature_importances_  # scores (unsorted)
    importances = 100.0 * (importances / importances.max()) #scales scores from 0-100, w/100 = most imp. feature
    indices = np.argsort(importances)  #ranks each feature, lists (by index) in ascending order.
    features = np.array(features)[indices]  #gets name corresponding to each feature

    num_features = X.shape[1]

    # is there any easy way to do scale this?
    # gets feature_importances for ea tree in the forest. then find the stdv across the trees (for each feature)
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_],axis=0)  #makes n_estimator (150) rows, num_predictor (11) columns.  does stdv on each column at a time (across all rows)


    # prints the feature importances
    print "\n\nFeature Importance Ranking:"
    print "------------------------------"
    for i,(f,s) in enumerate(zip(features[::-1], importances[indices][::-1]), start=1):
        print "{}. {} ({:.2f})".format(i,f,s)
    print "\n"


    # graphs the feature importances
    plt.figure()
    plt.barh(range(num_features), importances[indices], xerr=std[indices], align='center')
    plt.yticks(range(num_features), features)
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance Comparison')

    plt.tight_layout()
    #plt.savefig('feature_importances.png')
    plt.ion()
    plt.show()
'''
