import pandas as pd
from datetime import datetime

big_df = pd.read_excel("../data/harmonized.xlsx")

sites = ['ZSFG', 'Parnassus']
df = big_df[big_df['site'].isin(sites)]
len_bf = len(df)

df = df.dropna(subset=['note_date'])
pts_bf = len(pd.unique(df['ptid']))

def conv_to_dt(dat_str, site):
    # format_dat = "%Y-%m-%d %H:%M:%S.%f"
    # print(dat_str)
    if isinstance(dat_str, datetime):
        return dat_str
    if site == 'ZSFG':
        format_dat = "%Y-%m-%d %H:%M:%S.%f"
    elif site == 'Parnassus':
        format_dat = "%Y-%m-%d %H:%M:%S"
    
    dat = datetime.strptime(dat_str, format_dat)
    return dat

# print(df[df['note_date'].isna()])

# print(f'Original value: {date_string}\nConverted to datetime: {dat}')
# df[df['site']=='ZSFG']['note_date'] = df[df['site']=='ZSFG']['note_date'].apply(conv_to_dt, args=('ZSFG'))
# df[df['site']=='Parnassus']['note_date'] = df[df['site']=='Parnassus']['note_date'].apply(conv_to_dt, args=('Parnassus'))

dats = []
for index, row in df.iterrows():
    site = row['site']
    curr_dat = row['note_date']
    new_dat = conv_to_dt(curr_dat, site)
    dats.append(new_dat)

df['note_date'] = dats

df['first_time'] = df.groupby('ptid')['note_date'].transform('min')
df['timedelta_from_first'] = df['note_date'] - df['first_time']

df_within_168_hours = df[df['timedelta_from_first'] <= pd.Timedelta(days=7)]
df_between_72_and_168_hours = df_within_168_hours[df_within_168_hours['timedelta_from_first'] > pd.Timedelta(days=3)]
df_between_72_and_168_hours.drop(columns=['first_time', 'timedelta_from_first'], inplace=True)

print(df_between_72_and_168_hours)

df_between_72_and_168_hours.to_excel("../data/bt_72_168.xlsx")

pts_after = len(pd.unique(df_between_72_and_168_hours['ptid']))
print(f'Number of patients before filtering: {pts_bf}')
print(f'Number of patients after filtering: {pts_after}')


# # Calculate the time differences within each subject_id group
# df['time_diff'] = df.groupby('ptid')['note_date'].diff()

# # For the first row of each group, 'time_diff' will be NaN, so we replace it with 0
# df['time_diff'] = df['time_diff'].fillna(pd.Timedelta(seconds=0))

# # Calculate the cumulative sum of time differences within each subject_id group
# df['cumulative_time'] = df.groupby('ptid')['time_diff'].cumsum()

# # print(df[['time_diff','cumulative_time']])

# # Filter out rows where the cumulative time is less than or equal to 72 hours
# first_72_hours_df = df[df['cumulative_time'] <= pd.Timedelta(hours=168)]

# # Drop the intermediate columns we added
# first_72_hours_df.drop(columns=['time_diff', 'cumulative_time'], inplace=True)
# len_after = len(first_72_hours_df)

# pts_after = len(pd.unique(first_72_hours_df['ptid']))
# print(f'Number of patients before filtering: {pts_bf}')
# print(f'Number of patients after filtering: {pts_after}')
# print(f'Number of rows before filtering: {len_bf}')
# print(f'Number of rows after filtering: {len_after}')

# first_72_hours_df.to_excel("../data/first_168.xlsx")
