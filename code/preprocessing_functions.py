import pandas as pd
import re



#############################
# Preprocessing functions
#------------------------


def ds_prep(df, col):
    
    #------------------------------------------------------
    # Functions
    #------------------------------------------------------
    
    def extract_all(n, d):
        n = pd.Series(n).str.extractall('(' + d + ' ([\w]).{1,200})', flags=re.IGNORECASE).iloc[:,0].str.cat(sep=', ')          
        return n
       
    def all_fields(df, field, cols):
        df[field] = df[cols].astype('str').apply(lambda x: ' '.join(x), axis=1)
        df[field] = df[field].str.strip() 
        df[field] = df[field].replace(' nan ', '')
        return df
      
    def unique_list(words):
        """Remove duplicated words from list of tokenized words"""
        ulist = []
        [ulist.append(x) for x in words if x not in ulist]
        return ulist
      
    #------------------------------------------------------
    # Discharge outcome
    #------------------------------------------------------
        
    def extract_all_ds(n, d):
        if d == 'discharged':
            n = pd.Series(n).str.extractall('(discharged ([\w]).{1,50})')  # n pd.Series(c[col].iloc[0]).str
        elif d == 'dispo':
            n = pd.Series(n).str.extractall('(dispo ([\w]).{1,50})')  # n pd.Series(c[col].iloc[0]).str 
        elif d == 'snf1':
            n = pd.Series(n).str.extractall('(snf ([\w]).{1,1})')  # n pd.Series(c[col].iloc[0]).str 
        elif d == 'snf2':
            n = pd.Series(n).str.extractall('(skilled nursing ([\w]).{1,1})')  # n pd.Series(c[col].iloc[0]).str
        elif d == 'rehab':
            n = pd.Series(n).str.extractall('(rehab([\w]).{1,1})')  # n pd.Series(c[col].iloc[0]).str 
        elif d == 'deceased':
            n = pd.Series(n).str.extractall('(physician deceased)')  # n pd.Series(c[col].iloc[0]).str           
        else:
            n = pd.Series(n).str.extractall('(discharge ([\w]).{1,50})')
        n = n.iloc[:,0].str.cat(sep=', ') #pd.Series(n.iloc[:,0]).str          
        return n

    df['deceased'] =  pd.Series(df[col]).apply(lambda x: extract_all_ds(x,'deceased'))

    df['DD'] = ''
    df['DD2'] = ''
    df['DD3'] = ''
    df['snf1'] = ''
    df['snf2'] = ''
    df['rehab'] = ''
    
    df['DD'][df['deceased'].astype(str) == ''] =  pd.Series(df[col][df['deceased'].astype(str) == '']).apply(lambda x: extract_all_ds(x,'discharged'))
    df['DD2'][df['deceased'].astype(str) == ''] =  pd.Series(df[col][df['deceased'].astype(str) == ''] ).apply(lambda x: extract_all_ds(x,'discharge'))
    df['DD3'][df['deceased'].astype(str) == ''] =  pd.Series(df[col][df['deceased'].astype(str) == ''] ).apply(lambda x: extract_all_ds(x,'dispo'))
    df['snf1'][df['deceased'].astype(str) == ''] =  pd.Series(df[col][df['deceased'].astype(str) == ''] ).apply(lambda x: extract_all_ds(x,'snf1'))
    df['snf2'][df['deceased'].astype(str) == ''] =  pd.Series(df[col][df['deceased'].astype(str) == ''] ).apply(lambda x: extract_all_ds(x,'snf2'))
    df['rehab'][df['deceased'].astype(str) == ''] =  pd.Series(df[col][df['deceased'].astype(str) == ''] ).apply(lambda x: extract_all_ds(x,'rehab'))
    
    cols = ['DD','DD2','deceased'] 
   
    df = all_fields(df,'Discharge', cols)
          
    df['Discharge'] = df['Discharge'].apply(lambda x: " ".join(unique_list(x.split())))
  
    cols = ['Discharge','DD3','snf1','snf2']
    
    df = all_fields(df,'Discharge', cols)
    
    df = df.drop(columns = ['DD','DD2','DD3','snf1','snf2'])
    
    #------------------------------------------------------
    # Principal Diagnosis and chief complaint
    #------------------------------------------------------
    
    df['diagnosis'] =  pd.Series(df[col]).apply(lambda x: extract_all(x,'diagnosis'))
        
    df['principalproblem'] =  pd.Series(df[col]).apply(lambda x: extract_all(x,'problem'))
    
    df['diagnoses'] =  pd.Series(df[col]).apply(lambda x: extract_all(x,'diagnoses'))
    
    # df['reasonadmission'] =  pd.Series(df[col]).apply(lambda x: extract_all(x,'reason for admission'))
    
    df['ChiefComplaint'] = df[col].apply(lambda x: extract_all(x,'chief complaint'))
    
    cols = ['diagnosis','principalproblem','diagnoses','ChiefComplaint']
    
    df = all_fields(df,'Diagnosis', cols)
          
    df['Diagnosis'] = df['Diagnosis'].apply(lambda x: " ".join(unique_list(x.split())))
                                
    df['Diagnosis'] = df['Diagnosis'].apply(lambda x: x.replace('no hospital surgical or procedures surgeries this admission','no surgeries'))
    
    df['Diagnosis'] = df['Diagnosis'].apply(lambda x: x.replace('diagnoses active problems',''))
   
    df['Diagnosis'] = df['Diagnosis'].apply(lambda x: x.replace('diagnoses reason for admission',''))
    
    df['Diagnosis'] = df['Diagnosis'].apply(lambda x: x.replace('reason for admission',''))
   
    df['Diagnosis'] = df['Diagnosis'].apply(lambda x: x.replace('diagnosis',''))
    
    df['Diagnosis'] = df['Diagnosis'].apply(lambda x: x.replace('problem',''))
    
    df['Diagnosis'] = df['Diagnosis'].apply(lambda x: x.replace('principal',''))
    
    df = df.drop(columns = cols)
    
    #------------------------------------------------------
    # Code Status
    #------------------------------------------------------
    
    df['CodeStatusI'] = df[col].apply(lambda x: extract_all(x,'code status'))
    
    df['CodeStatusI'] = df['CodeStatusI'].apply(lambda x: x.replace('code status',''))
     
    df['CodeStatusI'] = df['CodeStatusI'].apply(lambda x: x.replace('code status at discharge',''))

    df['CodeStatusII'] = df[col].apply(lambda x: extract_all(x,'code'))

    cols = ['CodeStatusI','CodeStatusII']
    
    df = all_fields(df,'CodeStatus', cols)
          
    df['CodeStatus'] = df['CodeStatus'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)
    
    #------------------------------------------------------
    # Surgeries
    #------------------------------------------------------
    
    df['SurgeriesI'] = df[col].apply(lambda x: extract_all(x,'surgeries'))
                                    
    df['SurgeriesI'] = df['SurgeriesI'].apply(lambda x: x.replace('surgeries',''))

    df['SurgeriesII'] = df[col].apply(lambda x: extract_all(x,'surgery'))

    cols = ['SurgeriesI','SurgeriesII']
    
    df = all_fields(df,'Surgeries', cols)
          
    df['Surgeries'] = df['Surgeries'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)
    
    # df['Surgeries'] = df['Surgeries'].apply(lambda x: x.replace('none none procedures this admission none non or procedures','no surgeries'))
    
    # df['Surgeries'] = df['Surgeries'].apply(lambda x: x.replace('none procedures this admission none non or procedures','no surgeries'))
    
    #------------------------------------------------------
    # Tests
    #------------------------------------------------------
    
    df['Tests'] = df[col].apply(lambda x: extract_all(x,'tests'))
    
    #------------------------------------------------------
    # Allergies
    #------------------------------------------------------
    
    df['AllergiesI'] =  df[col].apply(lambda x: extract_all(x,'allergies'))
    
    df['AllergiesII'] =  df[col].apply(lambda x: extract_all(x,'allergic'))
    
    df['AllergiesIII'] =  df[col].apply(lambda x: extract_all(x,'allergen')) # added this

    cols = ['AllergiesI','AllergiesII', 'AllergiesIII']
    
    df = all_fields(df,'Allergies', cols)
          
    df['Allergies'] = df['Allergies'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)
    
    #------------------------------------------------------
    # Diet and nutrition
    #------------------------------------------------------
    
    df['DietI'] =  df[col].apply(lambda x: extract_all(x,'diet'))
       
    df['DietII'] =  df[col].apply(lambda x: extract_all(x,'nutrition'))
                  
    cols = ['DietI','DietII']
    
    df = all_fields(df,'Diet', cols)
          
    df['Diet'] = df['Diet'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)
    
    #------------------------------------------------------
    # Medical history 
    #------------------------------------------------------
    
    df['MedicalHxI'] =  df[col].apply(lambda x: extract_all(x,'history'))

    df['MedicalHxII'] =  df[col].apply(lambda x: extract_all(x,'HPI'))
                  
    cols = ['MedicalHxI','MedicalHxII']
    
    df = all_fields(df,'MedicalHx', cols)
          
    df['MedicalHx'] = df['MedicalHx'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)

    #------------------------------------------------------
    # Findings
    #------------------------------------------------------
    
    df['FindingsI'] =  df[col].apply(lambda x: extract_all(x,'Findings'))

    df['FindingsII'] =  df[col].apply(lambda x: extract_all(x,'Impression'))

    df['FindingsIII'] =  df[col].apply(lambda x: extract_all(x,'assessment'))

    cols = ['FindingsI','FindingsII', 'FindingsIII']
    
    df = all_fields(df,'Findings', cols)
          
    df['Findings'] = df['Findings'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)
      
    #------------------------------------------------------
    # Discharge Plan
    #------------------------------------------------------
    
    # df['Plan'] =  df[col].apply(lambda x: extract_all(x,'discharge plan'))
    
    # df['Plan'] = df['Plan'].apply(lambda x: x.replace('discharge plan',''))
    
    #------------------------------------------------------
    # Follow-up Care
    #------------------------------------------------------
    
    df['FollowUpI'] = df[col].apply(lambda x: extract_all(x,'follow up'))
    df['FollowUpI'] = df['FollowUpI'].apply(lambda x: x.replace('follow up',''))

    df['FollowUpII'] = df[col].apply(lambda x: extract_all(x,'Recommendation'))


    cols = ['FollowUpI', 'FollowUpII'] #,'ActivityII']
    
    df = all_fields(df,'FollowUp', cols)
          
    df['FollowUp'] = df['FollowUp'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)
    
    #------------------------------------------------------
    # Discharge instructions
    #------------------------------------------------------
    
    # df['Instructions'] = df[col].apply(lambda x: extract_all(x,'instructions'))
    
    # df['Instructions'] = df['Instructions'].apply(lambda x: x.replace('instructions',''))
    
    #------------------------------------------------------
    # Activity
    #------------------------------------------------------
    
    df['ActivityI'] = df[col].apply(lambda x: extract_all(x,'physical activity'))
    
    # df['ActivityII'] = df[col].apply(lambda x: extract_all(x,'activities'))
    
    cols = ['ActivityI'] #,'ActivityII']
    
    df = all_fields(df,'Activity', cols)
          
    df['Activity'] = df['Activity'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)
    
    #------------------------------------------------------
    # Physical Exam 
    #------------------------------------------------------
    
    df['PhysicalExamI'] = df[col].apply(lambda x: extract_all(x, 'Physical Examination'))# 'discharge exam'))
    
    df['PhysicalExamII'] = df[col].apply(lambda x: extract_all(x,'physical exam'))
    
    cols = ['PhysicalExamI','PhysicalExamII']
    
    df = all_fields(df,'PhysicalExam', cols)
          
    df['PhysicalExam'] = df['PhysicalExam'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)

    #------------------------------------------------------
    # Neuro Exam 
    #------------------------------------------------------
    
    df['NeuroI'] = df[col].apply(lambda x: extract_all(x, 'Neuro'))# 'discharge exam'))
    
    df['NeuroII'] = df[col].apply(lambda x: extract_all(x,'Neurologic'))
    
    cols = ['NeuroI','NeuroII']
    
    df = all_fields(df,'Neuro', cols)
          
    df['Neuro'] = df['Neuro'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)
    
    #------------------------------------------------------
    # Physical Therapy
    #------------------------------------------------------
    
    df['PhysicalTherapyI'] =  df[col].apply(lambda x: extract_all(x,'physical therapy'))

    df['PhysicalTherapyII'] = df[col].apply(lambda x: extract_all(x, 'PT'))# 'discharge exam'))
    
    cols = ['PhysicalTherapyI','PhysicalTherapyII']
    
    df = all_fields(df,'PhysicalTherapy', cols)
          
    df['PhysicalTherapy'] = df['PhysicalTherapy'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)
    
    #------------------------------------------------------
    # Occupational Therapy
    #------------------------------------------------------
    
    df['OccupTherapyI'] =  df[col].apply(lambda x: extract_all(x,'occupational therapy'))

    df['OccupTherapyII'] = df[col].apply(lambda x: extract_all(x, 'OT'))# 'discharge exam'))
    
    cols = ['OccupTherapyI','OccupTherapyII']
    
    df = all_fields(df,'OccupTherapy', cols)
          
    df['OccupTherapy'] = df['OccupTherapy'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)

    #------------------------------------------------------
    # Progress review
    #------------------------------------------------------
    
    df['ProgressI'] = df[col].apply(lambda x: extract_all(x, 'progress'))# 'discharge exam'))
    
    df['ProgressII'] = df[col].apply(lambda x: extract_all(x,'systems review'))

    df['ProgressIII'] = df[col].apply(lambda x: extract_all(x,'review of systems'))
    
    cols = ['ProgressI','ProgressII', 'ProgressIII']
    
    df = all_fields(df,'Progress', cols)
          
    df['Progress'] = df['Progress'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)
      
    #------------------------------------------------------
    # Treatments
    #------------------------------------------------------
    
    df['Treatments'] = df[col].apply(lambda x: extract_all(x,'treatment'))
    
    #------------------------------------------------------
    # Laboratory results 
    #------------------------------------------------------
    
    df['LabsI'] = df[col].apply(lambda x: extract_all(x,'lab'))

    df['LabsII'] = df[col].apply(lambda x: extract_all(x, 'laboratory'))# 'discharge exam'))
    
    cols = ['LabsI','LabsII']
    
    df = all_fields(df,'Labs', cols)
          
    df['Labs'] = df['Labs'].apply(lambda x: " ".join(unique_list(x.split())))
    
    df = df.drop(columns = cols)
    
    #------------------------------------------------------
    # Hospital Course
    #------------------------------------------------------
  
    def extract_all(n, d):
        n = pd.Series(n).str.extractall('(' + d + ' ([\w]).{1,1500})', flags=re.IGNORECASE).iloc[:,0].str.cat(sep=', ')          
        return n
    
    df['HospitalCourse'] = df[col].apply(lambda x: extract_all(x,'course'))

    df['HospitalCourse'] = df['HospitalCourse'].apply(lambda x: x.replace('course',''))
    
    return df


def join_fields(df, field, flag):
    
    #------------------------------------------------------------------------
    # Join all fields
    #------------------------------------------------------------------------
        
    if flag == 'DD':
        cols = ['Discharge', 'CodeStatus']
        
    elif  flag == 'noDD':
        cols = ['Diagnosis', 'Tests', 'CodeStatus',
        'Allergies', 'Diet', 'Surgeries', 'MedicalHx', 'Findings',
        'FollowUp', 'Activity', 'PhysicalExam', 'Neuro',
        'PhysicalTherapy', 'OccupTherapy', 'Progress', 'Treatments',
        'Labs' , 'HospitalCourse'] 
    
    else: # all        
        cols = ['Discharge', 'CodeStatus', 'Diagnosis', 'Tests',
        'Allergies', 'Diet', 'Surgeries', 'MedicalHx',
        'FollowUp', 'Activity', 'PhysicalExam',
        'PhysicalTherapy', 'OccupTerapService', 'Treatments',
        'Labs' , 'HospitalCourse'] 
    
    #cols.append('Notes_PT')
    #cols.append('Notes_OT')
        
    for col in cols:
        df[col][~(df[col].notna())] = col + '_missing'

        
    df[field] = df[cols].astype('str').apply(lambda x: ' '.join(x), axis=1)
    df[field] = df[field].apply(lambda x: " ".join(x.split())) # removes duplicated spaces
          
    return df


def lemma(df, field, feature):
    
    #------------------------------------------------------------------------
    # Create lemma
    #------------------------------------------------------------------------
    
    import numpy as np
    import nltk
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('punkt')
    from nltk.stem import WordNetLemmatizer

    def stem_lemma(column):
        df[column] = df[column].str.join(" ") # joining
        df[column] = df[column].str.strip()
        return df[column]
    
    def lemmatize_verbs(words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas
    
    df[feature] = df[field].str.split(" ") # splitting string (nltk.word_tokenize)
       
    df[feature] = df[feature].apply(lambda x: lemmatize_verbs(x))
     
    df[feature] = stem_lemma(feature) 
    
    
    #------------------------------------------------------------------------
    # Correct lemmatization and abbreviation expansion
    #------------------------------------------------------------------------
    
    remove = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
              'youre', 'youve', 'youll', 'youd', 'your', 'yours', 'yourself', 
              'yourselves', 'he', 'him', 'his', 'himself', 'she', 'shes', 
              'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 
              'their', 'theirs',  'themselves', 'the', 'with', 'to', 'be', 
              'from', 'which', 'dob', 'date', 'summary', 'please', 'dear', 
              'fa', 'facesheet', 'wi', 'llst', 'items', 'st', 'same', 'phone', 
              'sign', 'about', 'should', 'as', 'or', 'an', 'for', 'of', 'ac', 
              'in', 'by', 'at', 'fo', 'me', 'nan', 'hea', 'pro', 'and', 'up', 
              'full', 'code', 'have', 'has', 'is', 'part', 'do', 'will', 
              'there', 'faci', 'this', 'that', 'what', 'nc', 'comment', 
              'other', 'throughout', 'md', 'mdd', 'qd', 'per', 'sig', 'bid', 
              'when', 'use', 'while', 'apt', 'resu', 'con', 'dis', 'go', 'doct', 
              'mch', 'wnl', 'ml', 'mg', 'diff', 'tid', 'id', 'hs', 'medic', 
              'contact', 'but', 'hid', 'post', 'nt', 'first', 'if', 'then', 
              'who', 'whom',  'these', 'those', 'am', 'are', 'was', 'were', 
              'been', 'being', 'had', 'having', 'does', 'did', 'doing', 
              'because', 'until', 'against', 'between', 'into', 'through', 
              'during', 'before', 'after', 'above', 'below', 'down', 'out', 
              'on', 'off', 'over', 'under', 'again', 'further', 'once', 'here',
              'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
              'few', 'more', 'most', 'some', 'such', 'nor', 'only', 'own',
              'so', 'than', 'too', 'very', 'can',  'just', 'don', 'now']

     
    from nltk.tokenize import word_tokenize 
    
    def word(x, remove):
        x = word_tokenize(x)
        filtered_sentence = []
        for w in x: 
            if (w not in remove) & (len(w)>1): # remove single letters
                filtered_sentence.append(w) 
        filtered_sentence = ' '.join(filtered_sentence)        
        return filtered_sentence
    
    
    df[feature] = df[feature].astype(str).apply(lambda x: word(x, remove))  
    
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' covi ', ' covid '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' caregive ', ' caregiver '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' tota ', ' total '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' disch ', ' discharge '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' diagno ', ' diagnosis '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' demonst ', ' demonstrate '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' servi ', ' services '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' requir ', ' require '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' preferen ', ' preference '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' preferenc ', ' preference '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' orient ', ' orientation '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' abd ', ' abdominal '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' abn ', ' abdominal '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' abdo ',' abdominal '))
    df[feature] = df[feature].astype('str').apply(lambda x: x.replace(' afib ',' atrial fibrillation '))
   
    return df