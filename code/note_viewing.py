import pandas as pd

# df = pd.read_csv('../data/harmonized.csv')

# nrow = len(df)
# print(f'\n Total number of notes: {nrow} \n')

# mimic = df[df['site']=='Mimic iii']
# zsfg = df[df['site']=='ZSFG']
# parn = df[df['site']=='Parnassus']

# print(len(mimic))
# print(len(zsfg))
# print(len(parn))

# note_types = pd.unique(df['note_type'])
# print(df['note_type'].value_counts())

# pts = len(pd.unique(df['ptid']))
# print(f'\n Number of patients: {pts} \n')

# print(pd.unique(df['cpc_at_discharge']))
# print(df['cpc_at_discharge'].value_counts())

# df = pd.read_csv('../data/ZSFG_ids_notes_mapped_01-04-2024.csv')

# look_df = df[['ptid','cpc_at_discharge']]
# ids = look_df.sort_values(by='ptid') # 2600 notes with 368 patients

# rows_with_nan = look_df[look_df['cpc_at_discharge'].isna()] # 683 rows with NaN cpc
# print(len(pd.unique(rows_with_nan['ptid'])))

# df_bo = pd.read_csv('../data/ZSFG_all_notes_table_01-04-2024.csv')
# df_me = pd.read_csv('../data/ZSFG_all_notes_mapped_01-04-2024.csv')
# print(f'Rows in Bo table: {df_bo.shape}')
# print(f'Rows in Sophie table: {df_me.shape}')
# print(f'Columns in Sophie table: {df_me.columns}')


# miss = pd.read_csv('../../harmonizing/ZSFG_notes_missing_cpc.csv')
# pts = len(pd.unique(miss['ptid']))
# print(f'Number of patients in ZSFG_notes_missing_cpc: {pts}')

# map = pd.read_csv('../data/CardiacArrestUCSFRet_DATA_2023-09-27_0943_deidentified.csv')
# map = map[['ptid', 'cpc_at_discharge']]
# # print(f'Number of rows in CardiacArrestUCSF: {map.shape}')
# missing_cpc_in_CA = map[map['cpc_at_discharge'].isna()]
# pts = len(pd.unique(missing_cpc_in_CA['ptid']))
# print(f'Number of patients in CardiacArrestUCSF with missing CPC: {pts}')

df = pd.read_csv("../data/harmonized.csv") #,encoding="ISO-8859-1")
# df = df.rename(columns={'ï»¿SUBJECT_ID': 'SUBJECT_ID'}) # fix weird extra characters on column name
df = df.dropna(subset=["cpc_at_discharge"])
# df = df[df['note_type'].fillna('').str.contains('ischarge')]
df = df[~df['note_type'].fillna('').str.contains('ischarge')]
len_bf = len(df)

df = df.drop_duplicates(subset=['deid_note'])
len_after = len(df)

print(f"\nDf before dropping: {len_bf} rows.\nDf after dropping: {len_after}\n")
print(df['cpc_at_discharge'].value_counts())



# terms_dict_5_group = {'abdominal': -0.12703974878114896, 'active': 0.296189725497719, 'acute': 0.4270435128959583, 'admit': 0.3018200054172699, 'appropriate': 0.13606288937680908, 'assessment': 0.3072751941835176, 'assistance': -0.6552299360733999, 'attempt': 0.09977686950947941, 'awake': 0.6160793367878497, 'caregiver': 0.511357471225205, 'chair': -0.5758354651453146, 'consult': 0.059272290572329515, 'cont': 0.16637365879756497, 'cough': -0.08110375664416863, 'daily': -0.0950574580188167, 'decrease': 0.0858188095399888, 'dependent': -0.2474865885887821, 'device': -0.134522549839124, 'fall': -0.0840614358064465, 'focus': 0.14803214469365136, 'follow': -0.3010279838245354, 'general': 0.40808128412880773, 'hand': 0.15481261769584995, 'home': 0.40517289182870814, 'hospital': -0.0026562567728255428, 'intubate': -0.048203090274494986, 'lab': 0.0495082957625859, 'leave': 0.04465193711441156, 'line': 0.04573011010747038, 'maintain': -0.14889699564859848, 'maximal': -0.4809031150356884, 'mod': -0.06366528707329519, 'neuro': -0.6199525389858487, 'non': -0.7233452374753225, 'none': 0.07545098006809225, 'normal': 0.0914212104922027, 'note': -0.040467131759522455, 'placement': -0.06602853960680553, 'po': 0.5375161563008829, 'present': 0.03441655627934741, 'pt': -0.3023447825707482, 're': -0.07341563089347117, 'reach': -0.22698917352063636, 'recommend': 0.062434819078568474, 'recommendations': 0.09926618609964764, 'reinforcement': -0.5773438667608455, 'remain': 0.05456796472471844, 'require': -0.03168775731479021, 'rest': -0.08871174820361045, 'schedule': 0.4598635605430517, 'see': 0.25583008736055146, 'service': 0.27459809609802016, 'session': 0.022598613635457156, 'shock': -0.24097255538459703, 'skilled': -0.021086839830845226, 'skin': -0.26885512169929054, 'stable': 0.034563896224932486, 'therapy': 0.3439456858029539, 'trach': -0.8111158686706537, 'ue': 0.018660570857159834, 'vent': -0.290723391381109, 'visit': 0.06058455318152778, 'weight': -0.06472436220386277, 'well': -0.03425662463062915, 'cardiac arrest': 0.17587270626996793, 'discharge recommendation': 0.19209436456387988, 'follow command': -0.40989213787800566, 'physical therapy': 0.14902797773290377, 'progress note': -0.09331716137164967}