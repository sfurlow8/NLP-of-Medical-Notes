# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:39:10 2022

@author: mn46
"""

#############################
# Modeling functions
#------------------------

#%% libraries #############################################################

import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt   
import itertools
import seaborn as sns
import re
from itertools import cycle
from sklearn.metrics import (confusion_matrix, average_precision_score,
                             accuracy_score, recall_score, f1_score, auc,
                             precision_recall_curve, roc_auc_score, roc_curve, 
                             precision_score) 
# cohen_kappa_score,confusion_matrix      

from numpy import median, percentile
from numpy.random import seed, randint
from nltk.probability import FreqDist
# from nltk import word_tokenize
# import re
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier   


save_path = '../results/sectioned/test/' # path to save figures to

def transform_top_features(df, top_features, feature):
    
    #------------------------------------------------------------------------
    # Transform data with Top features 
    #------------------------------------------------------------------------

   top_features = list(top_features.Features.values)

   from nltk import word_tokenize

   def top_words(words):
       new_word = []
       for word in words:
           if word in top_features:
               new_word.append(word)
       return new_word

   df[feature] = df[feature].apply(lambda x: word_tokenize(x))
   df[feature] = df[feature].apply(lambda x: top_words(x)) 
   df[feature] = df[feature].apply(lambda x: ' '.join((x)))
    
   import re
   df['Ntokens'] = df[feature].astype('str').apply(lambda x: len(re.findall(r'\w+',x)))
    
   return df
    

def vectorization_(X_train, X_test):
    
    #------------------------------------------------------------------------
    # Vectorization
    #------------------------------------------------------------------------
   
    from sklearn.feature_extraction.text import CountVectorizer
    
    # word level tf-idf
    tfidf_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=None, ngram_range=(1,1))
    tfidf_vect.fit(X_train)
    xtrain_tfidf =  tfidf_vect.transform(X_train)
    xvalid_tfidf =  tfidf_vect.transform(X_test)
     
    # Train data
    x_train = pd.DataFrame(xtrain_tfidf.todense(), columns = tfidf_vect.get_feature_names())
    
    # Test data
    x_test = pd.DataFrame(xvalid_tfidf.todense(), columns = tfidf_vect.get_feature_names())
    
    return x_train, x_test, xtrain_tfidf, xvalid_tfidf, tfidf_vect




def train_test_encode(df, outcome, feature):
    
    from sklearn.model_selection import train_test_split
    
    #------------------------------------------------------------------------
    # Train/test dataset
    #------------------------------------------------------------------------
    
    #indices = list(df.index)
    
    df.dropna(inplace=True)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df[outcome])
    
    # indices_train = list(df_train.index)
    # indices_test = list(df_test.index)
    
    y_train = df_train[outcome]
    X_train = df_train.drop(columns = [outcome])
        
    y_test = df_test[outcome]
    X_test = df_test.drop(columns = [outcome])
    
    #------------------------------------------------------------------------
    # Encoding
    #------------------------------------------------------------------------
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test  = encoder.fit_transform(y_test)
    X_train = pd.Series(X_train[feature])
    X_test = pd.Series(X_test[feature])
    
    return X_train, X_test, y_train, y_test

def new_data_encode(df, outcome, feature):
    
    new_labels = df[outcome]
    new_dat = df.drop(columns = [outcome])

    
    #------------------------------------------------------------------------
    # Encoding
    #------------------------------------------------------------------------
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()

    new_labels = encoder.fit_transform(new_labels)
    new_dat = pd.Series(new_dat[feature])
    
    return new_dat, new_labels



def filter_features(X_train, X_test, y_train, y_test, outcome, feature):

    #------------------------------------------------------
    # Remove numbers 
    #------------------------------------------------------

    r=re.compile(r'\d')

    X_train[feature] = X_train[feature].apply(lambda x: r.sub('', x))  

    X_test[feature] = X_test[feature].apply(lambda x: r.sub('', x))  
 
    #------------------------------------------------------------------------
    # Remove misrepresented features in train
    #------------------------------------------------------------------------
        
    # remove single letters
    
    l =['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
              'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    for c in l:
        X_train = X_train.astype('str').apply(lambda x: x.replace(' ' + c + ' ',' '))
         
    #vectorization
    x_train, x_train, xtrain_tfidf, xtrain_tfidf, tfidf_vect = vectorization_(X_train[feature], X_train[feature])
    
    #matrix vector of features
    matrix_features = pd.DataFrame(xtrain_tfidf.todense(), columns = tfidf_vect.get_feature_names_out())
      
    #sparsity
    a = (matrix_features == 0).astype(int).sum(axis=0)/len(matrix_features)*100
    a = a[a<90]
    a = pd.DataFrame(a.index,columns = ['Features'])
    
    for c in l:
        a = a[~(a.Features.astype(str) == c)]
              
    matrix_features = matrix_features[a.Features]
    
    top_features = pd.DataFrame(list(matrix_features.columns.values), columns=list({'Features'})) # added list(Features)
     
    X_train = transform_top_features(pd.DataFrame(X_train), top_features, feature)
    
    X_train = X_train.drop(columns='Ntokens')
    X_test = pd.DataFrame(X_test)
    
    return X_train, X_test, y_train, y_test


def ngram(X_train, X_test, feature, ngram_range):
    #------------------------------------------------------------------------
    # Vectorization
    #------------------------------------------------------------------------
        
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    
    # word level tf-idf
    
    tfidf_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=None, ngram_range=ngram_range)
    
    #tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=None, ngram_range=ngram_range)
    tfidf_vect.fit(X_train[feature])
    xtrain_tfidf =  tfidf_vect.transform(X_train[feature])
    xvalid_tfidf =  tfidf_vect.transform(X_test[feature])  
  
    # Train data
    x_train = pd.DataFrame(xtrain_tfidf.todense(), columns = tfidf_vect.get_feature_names_out())
    x_test = pd.DataFrame(xvalid_tfidf.todense(), columns = tfidf_vect.get_feature_names_out())

    #sparsity
    a = (x_train == 0).astype(int).sum(axis=0)/len(x_train)*100
    
    #a.hist()
    a = a[a<90]
    a = pd.DataFrame(a.index,columns = ['Features'])
    
    x_train = x_train[a.Features]
    x_test = x_test[a.Features]

    return x_train, x_test



def plot__scores(mean_test_score, std_test_score, Cs):
        
    scores = mean_test_score
    scores_std = std_test_score
    
    plt.figure().set_size_inches(8, 6)
    plt.semilogx(Cs, scores)
    
    # plot error lines showing +/- std. errors of the scores
    n_folds = 5
    std_error = scores_std / np.sqrt(n_folds)
    
    plt.semilogx(Cs, scores + std_error, 'b--')
    plt.semilogx(Cs, scores - std_error, 'b--')
    
    # alpha=0.2 controls the translucency of the fill color
    plt.fill_between(Cs, scores + std_error, scores - std_error, alpha=0.2)
    
    plt.ylabel('CV $R^2$') # \pm$ standard error')
    #plt.ylabel('CV neg MSE +/- std error')
    # plt.ylabel('CV neg RMSE +/- std error')
    plt.xlabel('C')
    plt.axhline(np.max(scores), linestyle='--', color='.5')
    plt.xlim([Cs[0], Cs[-1]])

    plt.savefig(save_path+'scores.png', bbox_inches="tight")
    
    

def modeling(x_train, y_train, C):
    
    get_numeric_data = FunctionTransformer(lambda x: x[x_train.columns], validate=False)
  
    # clf = OneVsRestClassifier(LogisticRegression(penalty='l1', C = C, solver='liblinear', random_state = 42, class_weight='balanced'))
    clf = OneVsRestClassifier(LogisticRegression(penalty='l2', C = C, solver='lbfgs', random_state = 42, class_weight='balanced'))
  
    num_pipe = Pipeline([
       ('select_num', get_numeric_data)
       ])

    #------------------------------------------------------------------------
    # Hyperparameters
    #------------------------------------------------------------------------
    
    
    lst_params =  {'clf__estimator__warm_start':[True,False]} 
             
    full_pipeline = Pipeline([
            ('feat_union', FeatureUnion(transformer_list=[
                  ('select_num', num_pipe)
                  ])),
            ('clf', clf)
            ])
      
    #------------------------------------------------------------------------
    # Hyperparameter tuning in 5 fold CV
    #------------------------------------------------------------------------

    #neg_mean_squared_error
    #neg_root_mean_squared_error
    
    random_search = RandomizedSearchCV(full_pipeline, scoring = 'r2', param_distributions=lst_params, n_iter=100, cv=5, refit = True, n_jobs=-1, verbose=1, random_state = 42)
    
    #------------------------------------------------------------------------
    # Train and test
    #------------------------------------------------------------------------
    
    import sys
    sys.setrecursionlimit(10000)
    
    random_search.fit(x_train, y_train) 
    
    clf = random_search.best_estimator_

    # coef = pd.DataFrame(clf.steps[1][1].coef_, columns = x_train.columns)
   
    # ind = (coef == 0).all()
    # ind = pd.DataFrame(ind,columns={'Bool'})
    # top_features = pd.DataFrame(ind.loc[(ind.Bool==False)].index, columns={'Features'})

    return random_search #, coef, top_features

    
def modeling_fnc(x_train, y_train):
    
    
    def rmse_cv(x_train, y_train, C, std):
        
        random_search = modeling(x_train, y_train, C)
    
        if std == 'y':
            result = random_search.cv_results_['std_test_score'] #rmse
        else:
            result = random_search.cv_results_['mean_test_score'] #rmse
            
        return result


    Cs = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    
    mean_test_score = [rmse_cv(x_train, y_train, C = C, std ='n').mean() for C in Cs]
    
    std_test_score =  [rmse_cv(x_train, y_train, C = C, std ='y').mean() for C in Cs]

    plot__scores(mean_test_score, std_test_score, Cs)
     
    return mean_test_score, std_test_score, Cs  


def plt_importance(clf, class_labels, matrix_features, features, N):
    
    if N == []:
        N = len(matrix_features.columns)  # total number of features
    
    def plot_topN(clf, class_labels,N,a):
        print(clf)
        print(type(clf))
        coef0 = clf.steps[1][1].coef_
        coef = abs(coef0)

        underlying_estimator = clf.estimators_[1]  # Replace 0 with the index of the desired estimator
        coef0 = underlying_estimator.coef_
        coef0 = clf.estimator.coef_
        coef0 = clf.named_steps['clf'].estimators_[0].named_steps[a].coef_

        # Find the OneVsRestClassifier step in the pipeline
        for name, step in clf.named_steps.items():
            if isinstance(step, OneVsRestClassifier):
                ovr = step
                break

        # Access the coefficients of the underlying estimator
        coef0 = ovr.estimator_.coef_
        
        # print(ovr_classifier.estimators_)
        
        # ovr_classifier = clf.named_steps['clf']
        # coef0 = ovr_classifier.estimators_[0].coef_

        coef = abs(coef0)
        
        # c = pd.DataFrame(clf.steps[1][1].coef_, columns = a)
        c = pd.DataFrame(coef0, columns = a)
        ind = (c == 0).all()
        ind = pd.DataFrame(ind) #columns={'Bool'})
        a = ind.loc[(ind.Bool==False)].index   

        coef0 = coef0[:, (coef0 != 0).any(axis=0)]
        coef = abs(coef0)
        
        for i, class_label in enumerate(class_labels):
            feature_importance = coef[class_label]
            feature_signal = coef0[class_label]
            sorted_idx = np.argsort(np.transpose(feature_importance))
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            topN = sorted_idx[-N:]
            cols_names = []
            
            x1 = pd.DataFrame(feature_signal[topN])
            get_indexes_neg = x1.apply(lambda x: x[x<0].index)
            
            for j in topN:
                cols_names.append(a[j])         
            pos = np.arange(topN.shape[0]) + .5
            featfig = plt.figure(figsize=(8, 10))
            featax = featfig.add_subplot(1, 1, 1)
            b = featax.barh(pos, feature_importance[topN], align='center', color = 'steelblue', height=0.6)
            
            for ind in range(len(get_indexes_neg)):
                b[get_indexes_neg[0][ind]].set_color('indianred')
            featax.set_yticks(pos)
            featax.set_yticklabels(np.array(cols_names), fontsize=24)
            featax.set_xlabel('Relative Feature Importance', fontsize=24)
            
            plt.rcParams.update({'font.size': 24})
            
            if (i == 0) | ((i == len(class_labels)-1) & (len(class_labels)>5)):
                from matplotlib.lines import Line2D
                if len(get_indexes_neg) > 0:
                    custom_lines = [Line2D([0], [0], color='steelblue', lw=4),
                                    Line2D([0], [0], color='indianred', lw=4)]
                    featax.legend(custom_lines, ['Coefficient estimate > 0', 'Coefficient estimate < 0'], loc='lower right', prop={'size': 22})
                else:
                    custom_lines = [Line2D([0], [0], color='steelblue', lw=4)]
                    featax.legend(custom_lines, ['Coefficient estimate > 0'], loc='lower right', prop={'size': 22})

    
    plot_topN(clf,class_labels,N,features) 
    plt.savefig(save_path+'top_features.png', bbox_inches="tight")
    

def plt_importance2(clf, class_labels, matrix_features, features, N):
    if N == []:
        N = len(matrix_features.columns)

    ovr_classifier = clf.named_steps['clf']
    coefficients = ovr_classifier.estimators_[0].coef_
    binary_classifiers = ovr_classifier.estimators_

    feat_coefs = {}
    for i, coef in enumerate(coefficients[0]):
        # print(f"Feature {i}: Coefficient = {coef}")
        if coef != 0:
            feat = features[i]
            feat_coefs[feat] = coef
    
    print(feat_coefs)
    feat_coefs = dict(sorted(feat_coefs.items(), key=lambda item: item[1]))
    feat_coefs = {key: value for key, value in feat_coefs.items() if abs(value) >= 0.4}


    # Extract keys and values from the dictionary
    categories = list(feat_coefs.keys())
    values = list(feat_coefs.values())

    # Create a bar chart
    fig = plt.figure(figsize=(12,15))

    green = '#85C234'
    pink = '#C42782'

    # Assign colors based on positive or negative values
    colors = [green if value >= 0 else pink for value in values]

    # Create a bar chart with assigned colors
    plt.barh(categories, values, color=colors)

    # for i, value in enumerate(values):
    #     plt.text(value, i, str(value), va='center')

    # # Add a line at 0
    # plt.axhline(0, color='black', linestyle='--', linewidth=2)


    # Add labels and title
    plt.xlabel('LogReg Coefficient')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Tokens')
    plt.title('Influential Terms on CPC Scoring')
    plt.savefig(save_path+'inf_terms.png', bbox_inches="tight")


        # Plot feature importances for each label
    # for label_idx, binary_classifier in enumerate(binary_classifiers):
    #     if isinstance(binary_classifier, LogisticRegression):
    #         # Check if the base classifier supports feature importances (e.g., DecisionTreeClassifier)
    #         feature_importances = binary_classifier.feature_importances_
    #         num_features = len(feature_importances)

    #         # Create a bar plot to visualize feature importances for the current label
    #         plt.figure(figsize=(8, 6))
    #         plt.bar(range(num_features), feature_importances, tick_label=range(num_features))
    #         plt.title(f"Feature Importances for Label {label_idx}")
    #         plt.xlabel("Feature Index")
    #         plt.ylabel("Feature Importance")
    #         plt.show()
    #     else:
    #         print(f"Base classifier for Label {label_idx} does not support feature importances.")

    # plt.savefig(save_path+'top_features.png', bbox_inches="tight")
    

def perf(y_true, y_pred, probs, labels):
        
    ##############################
    # Target and predicted labels
    #---------------------------

    num_labels = len(labels)
    
    # y target
    y_true = np.array(pd.get_dummies(y_true))
    
    # y predicted - hard labels
    y_pred = np.array(pd.get_dummies(y_pred))
    
    if len(probs) !=0:
        y_pred = probs
    
    ###------------------------------------------------------------------------
    ### ROC
    ###------------------------------------------------------------------------
    
    colors = cycle(['darkgoldenrod', 'firebrick', 'steelblue', 'aqua', 'salmon',  'mediumorchid', 'black'])
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_labels)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_labels):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= num_labels
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    if len(labels) > 5:
        plt.figure(figsize=(12,8))
    else:
        plt.figure(figsize=(12,12))
    plt.rcParams.update({'font.size': 24})
    plt.plot(fpr["micro"], tpr["micro"],
              label='micro-average AUROC = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
              color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
              label='macro-average AUROC = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
              color='navy', linestyle=':', linewidth=4)
    
    
    for i, color in zip(range(num_labels), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                  label='{0} (AUROC = {1:0.2f})'
                  ''.format(labels[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('AUROC multi-class - hard labels') #
    # if len(probs) !=0:
    #     plt.title('AUROC multi-class - soft labels')
    if len(labels) > 5:
        plt.legend(loc="lower right",fontsize = 20)
    else:
        plt.legend(loc="lower right",fontsize = 18)
    plt.show()
    
    specificity= dict()
    specificity['micro'] = 1-fpr['micro']
    # sensitivity = tpr['micro']

    plt.savefig(save_path+'ROC_curve.png', bbox_inches="tight")
    
    ###------------------------------------------------------------------------
    ### AUPR
    ###------------------------------------------------------------------------
    
    # For each class
    precision = dict()
    recall = dict() 
    average_precision = dict()
    for i in range(num_labels):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                            y_pred[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
        y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_pred,
                                                         average="micro")
    ###------------------------------------------------------------------------
    ### AUPR - discriminated
    ###------------------------------------------------------------------------
    
    # setup plot details
    plt.figure(figsize=(20, 12))
    plt.rcParams.update({'font.size': 40})
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels_ = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.7, y[45] + 0.02))
    
    lines.append(l)
    labels_.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels_.append('micro-average (AUPRC = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    
    
    for i, color in zip(range(num_labels), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels_.append('{0} (AUPRC = {1:0.2f})'
                      ''.format(labels[i], average_precision[i]))
    
    # fig = plt.gcf()
    # fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.title('Precision-Recall curve to multi-class - hard labels')
    # plt.title('Precision-Recall curve to multi-class - soft hot encoded labels')
    # plt.title('Precision-Recall curve to multi-class - soft labels')
    plt.legend(lines, labels_, loc=(1.05, 0.12), fontsize = 36)#prop=dict(size=20))
    # plt.show()
    plt.savefig(save_path+'AUPR.png', bbox_inches="tight")
    
    #------------------------------------------------------------------------
    # Confusion matrix
    #------------------------------------------------------------------------
    
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix', n = 'None',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        
            
        if normalize:
            if n == 'recall':
                axis = 1
                cm = cm.astype('float') / cm.sum(axis=axis)[:, np.newaxis]
            elif n == 'precision':
                axis = 0
                cm = cm.astype('float') / cm.sum(axis=axis)[np.newaxis,:]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
            
        print(cm)
    
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.rcParams.update({'font.size': 12}) # 18 for GOS, 14 for mRS
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        # plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=60)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(save_path+'cnfmat.png', bbox_inches="tight")
       
        
    
    cnf_matrix = confusion_matrix(np.argmax(np.array(y_true), axis=1), np.argmax(y_pred, axis=1))
    
    # if len(labels) > 5:
    #     labels = ['mRS 0','mRS 1','mRS 2','mRS 3','mRS 4','mRS 5','mRS 6']
    # else:
    #     if (len(labels) == 4) & (outcome == 'GOS'):
    #         labels = ['GOS 1', 'GOS 3', 'GOS 4', 'GOS 5']
    #     if (len(labels) == 3) & (outcome == 'MRS'):    
    #         labels = ['mRS 0-2','mRS 3-5','mRS 6']
    #     if (len(labels) == 4) & (outcome == 'MRS'):  
    #         labels = ['mRS 0-1','mRS 2-4','mRS 5','mRS 6']
       
    
    # Plot normalized by recall confusion matrix
    fig = plt.figure()
    plt.rcParams.update({'font.size': 14}) 
    plot_confusion_matrix(cnf_matrix, classes=np.asarray(labels), normalize=True, n = 'recall')
    #plt.grid(b=None)
    plt.savefig(save_path+'norm_by_recall_cnfmat.png', bbox_inches="tight")
                           
    # Plot normalized by precision confusion matrix
    fig = plt.figure()
    plt.rcParams.update({'font.size': 14}) 
    plot_confusion_matrix(cnf_matrix, classes=np.asarray(labels), normalize=True, n = 'precision')
    plt.savefig(save_path+'norm_by_prec_cnfmat.png', bbox_inches="tight")
    #plt.grid(b=None)
    
    # fig = plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=np.asarray(labels), normalize=False, n = 'None')
                         
    #plt.grid(b=None)  
    
    

    #%% Bootstrapping function for Multiclass #####################################
    
    def get_CI_boot(y_true,y_pred,metric,boot,metric_average):
    
        # bootstrap confidence intervals
        # seed the random number generator
        seed(1)
        i = 0
        # generate dataset
        dataset = y_pred
        real = y_true
        # bootstrap
        scores = list()
        while i < boot:
            # bootstrap sample
            indices = randint(0, len(y_pred) - 1, len(y_pred))
            sample = dataset[indices]
            real = y_true[indices]
            if metric == roc_auc_score: 
                    # Compute micro-average ROC curve and ROC area
                fpr[metric_average], tpr[metric_average], _ = roc_curve(real.ravel(), sample.ravel())
                roc_auc[metric_average] = auc(fpr[metric_average], tpr[metric_average])              
                scores.append(roc_auc[metric_average])
                i += 1
            elif metric == average_precision_score:
                    # A "micro-average": quantifying score on all classes jointly
                average_precision[metric_average] = average_precision_score(real, sample, average=metric_average)                
                scores.append(average_precision[metric_average])
                i += 1
            elif metric == precision_score:
                precision[metric_average] = precision_score(real, sample, average=metric_average)
                scores.append(precision[metric_average] )
                i += 1
            elif metric == recall_score:
                recall[metric_average] = recall_score(real, sample, average=metric_average)              
                scores.append(recall[metric_average])
                i += 1   
            elif metric == accuracy_score:
                scores.append(accuracy_score(real, sample))
                i += 1   
            elif metric == f1_score:
                scores.append(f1_score(real, sample, average=metric_average))
                i += 1          
        # calculate 95% confidence intervals (100 - alpha)
        alpha = 5.0
        # calculate lower percentile (e.g. 2.5)
        lower_p = alpha / 2.0
        # retrieve observation at lower percentile
        lower = max(0.0, percentile(scores, lower_p))
        # calculate upper percentile (e.g. 97.5)
        upper_p = (100 - alpha) + (alpha / 2.0)
        # retrieve observation at upper percentile
        upper = min(1.0, percentile(scores, upper_p))
        return (lower,upper)
    
    
    #%% Bootstrapping function ###################################################
        
    def get_CI_boot_outcome(y_true,y_pred,metric,boot):
        # bootstrap confidence intervals
        # seed the random number generator
        seed(1)
        i = 0
        # generate dataset
        dataset = y_pred
        real = y_true
        # bootstrap
        scores = list()
        while i < boot:
            # bootstrap sample
            indices = randint(0, len(y_pred) - 1, len(y_pred))
            sample = dataset[indices]
            real = y_true[indices]
            if len(np.unique(y_true[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
        	# calculate and store statistic 
            else:
                statistic = metric(real,sample)
                scores.append(statistic)
                i += 1
        # calculate 95% confidence intervals (100 - alpha)
        alpha = 5.0
        # calculate lower percentile (e.g. 2.5)
        lower_p = alpha / 2.0
        # retrieve observation at lower percentile
        lower = max(0.0, percentile(scores, lower_p))
        # calculate upper percentile (e.g. 97.5)
        upper_p = (100 - alpha) + (alpha / 2.0)
        # retrieve observation at upper percentile
        upper = min(1.0, percentile(scores, upper_p))
        return (lower,upper)
    
    
    #%% Bootsrapping results in test multiclass ###################################
    
    myp = []
    mye = []
    
    metrics = [roc_auc_score, accuracy_score,  recall_score,
               f1_score, average_precision_score, precision_score]
    
    boot=100
    metric_average = "micro"
       
    e = []
    for p in metrics:
        
        if (len(probs) !=0) & (p != roc_auc_score):
            y_pred_ = np.argmax(y_pred,axis=1)
            y_pred_ = np.array(pd.get_dummies(y_pred_))
        else:  
            y_pred_ = y_pred
        
        extremes = get_CI_boot(y_true,y_pred_,p,boot,metric_average) #atencao aqui ao boot
        
        e.append(extremes)
        if p == accuracy_score :
            myp.append((round(p(y_true,y_pred_), 2)))
            
        else:
            if (len(probs) !=0) & (p != roc_auc_score):
                y_pred_ = np.argmax(y_pred,axis=1)
                y_pred_ = np.array(pd.get_dummies(y_pred_))
            else:
                y_pred_ = y_pred   
            myp.append((round(p(y_true,y_pred_,average=metric_average), 2)))
        mye.append(str(' ['+ str(round(extremes[0], 2))  +'-'+  str(round(extremes[1], 2)) +']'))
    
    df1 = np.transpose(pd.DataFrame(myp, index=['AUROC','ACC','Recall','F1','AP','PPV']))
    df2 = np.transpose(pd.DataFrame(mye, index=['AUROC','ACC','Recall','F1','AP','PPV']))
    boot_all = pd.concat([df1,df2]) 
                        
    
       
    #%% Specificity and sensitivity (recall) ##################################
    
    def spec(y_true,y_pred):
        TN, FP, FN, TP = confusion_matrix(y_true,y_pred).ravel()
        return TN/(TN+FP)   
    
    def sens(y_true,y_pred):
        TN, FP, FN, TP = confusion_matrix(y_true,y_pred).ravel()
        return TP/(TP+FN) 
        
    #%% Bootsrapping results in test per outcome ##################################
        
    if (len(probs) !=0):
         y_pred_ = y_pred
         y_pred = np.argmax(y_pred,axis=1)
         y_pred = np.array(pd.get_dummies(y_pred))
    
    df1 = []
    df2 = []
    
    for outcome in range(0,len(labels)):
    
        y_pred_outcome = y_pred[:,outcome]
        
        y_true_outcome = y_true[:,outcome]
        
        myp = []
        mye = []
        
        metrics = [roc_auc_score, accuracy_score, sens, spec, f1_score, 
                   average_precision_score, precision_score] 
        
        y_true_outcome = np.array(y_true_outcome)
        
        for p in metrics:
            if (len(probs) !=0) & (p == roc_auc_score):
                y_pred_outcome = y_pred_[:,outcome]
            else:
                y_pred_outcome =  y_pred[:,outcome]   
            extremes = get_CI_boot_outcome(y_true_outcome,y_pred_outcome,p,boot) #atencao aqui ao boot
            myp.append((round(p(y_true_outcome,y_pred_outcome), 2)))
            mye.append(str(' ['+ str(round(extremes[0], 2))  +'-'+  str(round(extremes[1], 2)) +']'))
        
        df1.append(myp)                                                   
        df2.append(mye)
    
    df1 = pd.DataFrame(df1, columns = ['AUROC','ACC','Recall','Spec','F1','AP','PPV'])
    df2 = pd.DataFrame(df2, columns = ['AUROC','ACC','Recall','Spec','F1','AP','PPV'])
         
    boot_label = pd.concat([df1,df2], axis = 1)
     
    return boot_all, boot_label