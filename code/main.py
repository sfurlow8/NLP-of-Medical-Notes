
#------------------------------------------------------------------------
# Libraries for preprocessing
#------------------------------------------------------------------------

import numpy as np
import pandas as pd
import re
import os
import sys
path = '../data/' # user directory where data is stored
sys.path.insert(0, path) # insert path


path_ = '../results/' # user directory where result plots go
sys.path.insert(0, path_) # insert path

from preprocessing_functions import ds_prep, join_fields, lemma

#------------------------------------------------------------------------
# Libraries for modeling
#------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from modeling_functions import (ngram, filter_features, plot__scores,
modeling, modeling_fnc, perf, plt_importance, plt_importance2, train_test_encode, new_data_encode)

import random
random.seed(42)

#------------------------------------------------------
# MAIN preprocessing
#------------------------------------------------------

#------------------------------------------------------
# Deidentified examples of notes for preprocessing
#------------------------------------------------------

df = pd.read_csv(path+"harmonized.csv")
df = df.dropna(subset=["cpc_at_discharge"]) # drop notes without cpc score
df = df[~df['note_type'].fillna('').str.contains('ischarge')] # drop discharge summaries

len_bf = len(df)

df = df.drop_duplicates(subset=['deid_note'])
len_after = len(df)

print(f"\nDf before dropping: {len_bf} rows.\nDf after dropping: {len_after}\n")


cpc_5_out = 5.0
df = df[df['cpc_at_discharge'] != cpc_5_out]


# adjust this func depending on how many CPC groups you want
def get_cpc_group(cpc):
    if cpc in [1, 2]:
        return 'CPC1-2'
    elif cpc in [3, 4]:
        return 'CPC3-4'
    # if cpc == 1:
    #     return 'CPC 1'
    # elif cpc == 2:
    #     return 'CPC 2'
    # elif cpc == 3:
    #     return 'CPC 3'
    # elif cpc == 4:
    #     return 'CPC 4'
    # # else:
    # #     return 'CPC 5'

df['cpc_score'] = df['cpc_at_discharge'].apply(get_cpc_group)

note_text = 'deid_note'
subject_id = 'ptid'
 
#------------------------------------------------------
# Preprocessing
#------------------------------------------------------

def remove_ponctuation_spaces(df,column):
    
    df[col] = df[col].astype(str).str.lower()
    df[col] = df[col].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x)).apply(lambda x: " ".join(x.split()))

    return df

for col in list([note_text]):
    
    df = remove_ponctuation_spaces(df,col)

df_ds = ds_prep(df, col = note_text)
# print(df_ds)
# print(df_ds.columns)
# print(df_ds['Diagnosis'])
# print(df_ds['Addendum'])
# print(df_ds['Labs'])


field = 'Allfields'

modeling_field = 'lemma'

df_prep = join_fields(df_ds, field, flag='noDD') # flag: no discharge disposition

df_prep = lemma(df_prep, field, modeling_field)

df_modeling = df_prep[[subject_id,'lemma']]

df_modeling.to_csv(os.path.join(path_,'notes_sample_preprocessed.csv'), sep=',', index = False)

# df_modeling = pd.read_csv(path_+'notes_sample_preprocessed.csv')

#-------------------------------------------
df_72 = pd.read_excel(path+'bt_72_168.xlsx')
df_72 = df_72[df_72['cpc_at_discharge'] != cpc_5_out]
df_72['cpc_score'] = df_72['cpc_at_discharge'].apply(get_cpc_group)

df_ds_72 = ds_prep(df_72, col = note_text)
df_prep_72 = join_fields(df_ds_72, field, flag='noDD') # flag: no discharge disposition

df_prep_71 = lemma(df_prep_72, field, modeling_field)

df_modeling_72 = df_prep_72[[subject_id,'lemma']]

df_modeling_72.to_csv(os.path.join(path_,'notes_sample_preprocessed.csv'), sep=',', index = False)
#--------------------------------------------

###########################################################################

#------------------------------------------------------
# MAIN modeling
#------------------------------------------------------

outcome = 'CPC' # or 'GOS' user defined

n_patterns = 2 # number of patterns for MRS or CPC, user defined

feature = 'lemma' # text column
 

#------------------------------------------------------------------------
# Outcome
#------------------------------------------------------------------------
 
if outcome == 'GOS':
    
    ##0	Death
    ##-	Persistent vegetative state 
    ##1 Severe disability (lower & upper) 
    ##2	Moderate disability (lower & upper) 
    ##3 Good recovery (lower & upper) 
    
    labels = ['GOS 1', 'GOS 3', 'GOS 4', 'GOS 5']
      
    outcome_name='gos.csv'
    
    C = 0.05
    
elif outcome ==  'MRS': # (outcome == 'MRS')
   
    if n_patterns == 7:
        
        ##0	No symptoms 	
        ##1	No significant disability	
        ##2	Slight disability	
        ##3	Moderate disability	
        ##4	Moderately severe disability	
        ##5	Severe disability	
        ##6	Death
    
        labels = ['mRS 0','mRS 1','mRS 2','mRS 3','mRS 4','mRS 5','mRS 6']
        
        outcome_name='mrs.csv'
        
        C = 0.05
    
    if n_patterns == 4:
        
        # col = 'MRS' # df is the user dataset dataframe
        # df[col][(df[col] == 1)] = 0 # 'No sympt+Not significant'    
        # df[col][(df[col] == 2) | (df[col] == 3) | (df[col] == 4)] = 1 # 'Slight+Mod+Mod sev'        
        # df[col][(df[col] == 5)] = 2 # 'Severe'    
        # df[col][(df[col] == 6)] = 3 # 'Death'    
      
        labels = ['mRS 0-1','mRS 2-4','mRS 5','mRS 6']
        
        outcome_name = 'mrs_4patterns.csv'
        
        C = 0.06

    if n_patterns == 3:
        
        # col = 'MRS' # df is the user dataset dataframe
        # df[col][(df[col] == 1) | (df[col] == 2)] = 0 # 'No symptoms+Not significant+Slight'    
        # df[col][(df[col] == 3) | (df[col] == 4) | (df[col] == 5)] = 1 # 'Moderate+Moderately severe+ Severe'  
        # df[col][(df[col] == 6)] = 2 # 'Death'  
   
        labels = ['mRS 0-2','mRS 3-5','mRS 6']
        
        outcome_name = 'mrs_3patterns.csv'
        
        C = 0.08

    # use just 2 categories to test accuracy on MIMIC (alive or dead)
    if n_patterns == 2:
        
        # col = 'MRS' # df is the user dataset dataframe
        # df[col][(df[col] == 1) | (df[col] == 2)] = 0 # 'No symptoms+Not significant+Slight'    
        # df[col][(df[col] == 3) | (df[col] == 4) | (df[col] == 5)] = 1 # 'Moderate+Moderately severe+ Severe'  
        # df[col][(df[col] == 6)] = 2 # 'Death'  
   
        labels = ['mRS 0-5', 'mRS 6']
        
        outcome_name = 'mrs_2patterns.csv'
        
        C = 0.09

else:
    if n_patterns == 5:

        labels = ['CPC 1','CPC 2','CPC 3','CPC 4', 'CPC 5']
        
        outcome_name = 'cpc_5patterns.csv'
        
        C = 0.06

    if n_patterns == 4:

        labels = ['CPC 1','CPC 2','CPC 3','CPC 4']
        
        outcome_name = 'cpc_4patterns.csv'
        
        C = 0.06

    if n_patterns == 2:
        # labels = ['CPC1-4', 'CPC5']
        labels = ['CPC1-2', 'CPC3-4']
        
        outcome_name = 'cpc_2patterns.csv'
        
        C = 0.09

    
# Load data

# x_train = pd.read_csv(os.path.join(path,'x_train_' + outcome_name), sep=',')
# x_test = pd.read_csv(os.path.join(path,'x_test_' + outcome_name), sep=',')

# y_train = pd.read_csv(os.path.join(path,'y_train_' + outcome_name), sep=',').iloc[:,0].values
# y_test = pd.read_csv(os.path.join(path,'y_test_' + outcome_name), sep=',').iloc[:,0].values

# x_train = pd.read_csv(os.path.join('x_train_' + outcome_name), sep=',')
# x_test = pd.read_csv(os.path.join('x_test_' + outcome_name), sep=',')

# y_train = pd.read_csv(os.path.join('y_train_' + outcome_name), sep=',').iloc[:,0].values
# y_test = pd.read_csv(os.path.join('y_test_' + outcome_name), sep=',').iloc[:,0].values

#------------------------------------------------------
# Create train and test sets
#------------------------------------------------------

#stratified random sampling 
# mrs_vals = []
# for ii in df['DEATH_STATUS']:
#     if ii == 'Died': mrs_vals.append('mRS 6')
#     else: mrs_vals.append('mRS 0-5')

df_modeling['CPC'] = df['cpc_score']
df_modeling_72['CPC'] = df_72['cpc_score']


X_train, X_test, y_train, y_test = train_test_encode(df_modeling, outcome, feature) # changed from df

X_train = pd.DataFrame(X_train).reset_index().drop(columns='index')

X_test = pd.DataFrame(X_test).reset_index().drop(columns='index')

pd.DataFrame(y_train).to_csv('y_train_' + outcome_name, sep=',',index = False)
pd.DataFrame(y_test).to_csv('y_test_'+ outcome_name, sep=',',index = False)


# X_train_72, X_test_72, y_train_72, y_test_72 = train_test_encode(df_72, outcome, feature) # changed from df

new_dat, new_labels = new_data_encode(df_modeling_72, outcome, feature) # changed from df

new_dat_72 = pd.DataFrame(new_dat).reset_index().drop(columns='index')

#------------------------------------------------------------------------
# Remove misrepresented features in train
#------------------------------------------------------------------------
 
X_train, X_test, y_train, y_test = filter_features(X_train, X_test, y_train, y_test, outcome, feature)

garbage1, X_test_72, garbage2, y_test_72 = filter_features(X_train, new_dat_72, y_train, new_labels, outcome, feature)


#------------------------------------------------------------------------
# Vectorization
#------------------------------------------------------------------------
 
# vectorization for combinations of n-grams
n_gram_comb =  [(1,1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

x_tr = []
x_t = []
for ngram_range in n_gram_comb:
    x_train, x_test = ngram(X_train, X_test, feature, ngram_range)
    x_tr.append(x_train)
    x_t.append(x_test)

# # if memory problems occur, do one at a time
# # x_train33, x_test33 = ngram(X_train, X_test, feature, (3, 3))
# # x_tr.append(x_train33)
# # x_t.append(x_test33)  
    
#join all    
x_tr = pd.concat(x_tr, axis = 1)
x_t = pd.concat(x_t, axis = 1)   

#remove repeated columns
columns = x_tr.columns.drop_duplicates()
x_train = x_tr.loc[:,~x_tr.columns.duplicated()]
x_test = x_t.loc[:,~x_t.columns.duplicated()]

x_train[x_train >0] = 1 
x_test[x_test >0] = 1 

x_train.to_csv('x_train_' + outcome_name, sep=',',index = False)
x_test.to_csv('x_test_' + outcome_name, sep=',',index = False)
  
# We ran the above first part and saved the x and y matrices

#-------------------------------------------------------------
x_tr_72 = []
x_t_72 = []
for ngram_range in n_gram_comb:
    x_train_72, x_test_72 = ngram(X_train, X_test_72, feature, ngram_range)
    x_tr_72.append(x_train_72)
    x_t_72.append(x_test_72)

# # if memory problems occur, do one at a time
# # x_train33, x_test33 = ngram(X_train, X_test, feature, (3, 3))
# # x_tr.append(x_train33)
# # x_t.append(x_test33)  
    
#join all    
x_tr_72 = pd.concat(x_tr_72, axis = 1)
x_t_72 = pd.concat(x_t_72, axis = 1)   

#remove repeated columns
columns_72 = x_tr_72.columns.drop_duplicates()
x_train_72 = x_tr_72.loc[:,~x_tr_72.columns.duplicated()]
x_test_72 = x_t_72.loc[:,~x_t_72.columns.duplicated()]

x_train_72[x_train_72 >0] = 1 
x_test_72[x_test_72 >0] = 1 

#-------------------------------------------------------------


#------------------------------------------------------------------------
#  Modeling
#------------------------------------------------------------------------

# Run this code in case we don't have C 
# mean_test_score, std_test_score, Cs = modeling_fnc(x_train, y_train)
 
# # Highest mean - correspondent std
# maxpos = mean_test_score.index(max(mean_test_score)) 
# min_test_score = np.max(mean_test_score) - std_test_score[maxpos]

# mean_test_score_ = list(np.round(mean_test_score,2))
# mean_test_score_aux = mean_test_score_[0:maxpos]

# Select C based on the one standard error rule

# # When we have an index position for automatic selection of C
# # pos = mean_test_score_aux.index(np.round(min_test_score,2)) 
# # if pos > maxpos:
# #     C = Cs[maxpos]
# # else:
# #     C = Cs[pos]
      
# We already selected C we can run from this line
random_search = modeling(x_train, y_train , C)
 
clf = random_search.best_estimator_
    
#------------------------------------------------------------------------
# Performance
#------------------------------------------------------------------------

probs = clf.predict_proba(x_test)
y_pred = np.argmax(probs,axis=1)
print('\n\n\n')
print(f"y_pred: \n\n{y_pred}")
print('\n\n\n')
# df_72['y_pred'] = y_pred_72

# df_72.to_excel(path+'bt_72_168_pred.xlsx')

print('\n\n\n')
print(f"y_test: \n\n{y_test}")
print('\n\n\n')

boot_all, boot_label = perf(y_test, y_pred, probs, labels) 

#------------------------------------------------------------------------
# Save model results
#------------------------------------------------------------------------

model = pd.DataFrame(y_pred, columns = ['y_pred'])

model['y_test'] = y_test

model = pd.concat([model,pd.DataFrame(probs)],axis=1)

if outcome == 'GOS':   
    model.to_csv(os.path.join(path_,'modelGOS.csv'), sep=',', index = False) 
elif outcome == 'MRS':    
    if n_patterns == 7:
        model.to_csv(os.path.join(path_,'modelMRS.csv'), sep=',', index = False) 
    if n_patterns == 4:
        model.to_csv(os.path.join(path_,'modelMRS4.csv'), sep=',', index = False)  
    if n_patterns == 3:
        model.to_csv(os.path.join(path_,'modelMRS3.csv'), sep=',', index = False)  
    if n_patterns == 2:
        model.to_csv(os.path.join(path_,'modelMRS2.csv'), sep=',', index = False) 
else: # outcome = CPC 
    if n_patterns == 5:
        model.to_csv(os.path.join(path_,'modelCPC5.csv'), sep=',', index = False)  
    if n_patterns == 2:
        model.to_csv(os.path.join(path_,'modelCPC2.csv'), sep=',', index = False) 

  
#------------------------------------------------------------------------
# Feature importance estimates - plot
#------------------------------------------------------------------------

features = x_train.columns

class_labels = list(range(len(np.unique(y_train))))

# plt_importance(clf, class_labels, probs, features, N=15)

plt_importance2(clf, class_labels, probs, features, N=15)

#------------------------------------------------------------------------
# Number of uni, bi and tri-grams in the set of features
#------------------------------------------------------------------------

# from nltk.tokenize import word_tokenize 

# top_features['n'] = top_features.Features.astype(str).apply(lambda x: len(word_tokenize(x)))

# print(len(top_features[top_features.n == 1])) 
# print(len(top_features[top_features.n == 2])) 
# print(len(top_features[top_features.n == 3])) 

# # x_train

# xt = pd.DataFrame(x_train.columns, columns=['Features'])

# xt['n'] = xt.Features.astype(str).apply(lambda x: len(word_tokenize(x)))

# print(len(xt[xt.n == 1])) 
# print(len(xt[xt.n == 2])) 
# print(len(xt[xt.n == 3])) 

# #------------------------------------------------------------------------
# # Performance by Race
# #------------------------------------------------------------------------

# if outcome == 'GOS':    
#     e = pd.read_csv(os.path.join(path,"X_gos.csv"))
# else:    
#     e = pd.read_csv(os.path.join(path,"X_mrs.csv"))

# #------------------------------------------------------
# # Create train and test sets
# #------------------------------------------------------

# df_train, df_test = train_test_split(e, test_size=0.3, random_state=42, stratify=e[outcome])

# indices_test = list(df_test.index)
    
# #------------------------------------------------------------------------
# # Check predictions by race
# #------------------------------------------------------------------------

# e_ = e.iloc[indices_test,:]

# e_ = pd.concat([e_.reset_index(),model], axis=1)

# #------------------------------------------------------------------------
# # Performance
# #------------------------------------------------------------------------

# def result(race,e_,labels,outcome,n_patterns):
    
#     df = e_[e_.Race == race]
    
#     if outcome == 'GOS': 
#         probs = np.array(df.iloc[:,-4:]) # gos,mrs4    
#     else: # (outcome == 'MRS')
#         if n_patterns == 7:
#             probs = np.array(df.iloc[:,-7:]) # mrs
#         if n_patterns == 4:
#             probs = np.array(df.iloc[:,-4:]) # gos,mrs4
#         if n_patterns == 3:
#             probs = np.array(df.iloc[:,-3:]) # mrs3
  
#     y_pred = df.y_pred
#     y_test = df.y_test
#     boot_all = perf(y_test, y_pred, probs, labels)
    
#     return boot_all

# #races = ['white','black or african american','asian','hispanic or latino'] # precision may be ill-defined due to class representativity

# #ba = []
# #for race in races:
# #    boot_all = result(race,e_,labels,outcome,n_patterns)
# #    ba.append(boot_all)

# # ba list:
# # each tuple for each race
#     # each race has 2 tuples: 
#         #0 bootstrapping for macro performance
#         #1 bootstrapping for N labels performance

# #############################################################################