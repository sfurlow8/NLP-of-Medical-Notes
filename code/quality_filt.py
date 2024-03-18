import pandas as pd

def filter_by_word(df, col, target_word):
    """
    Parameters:
    - df (pd.DataFrame): df to be filtered.
    - col (str): name of note column.
    - target_word (str): word you do not want to keep.

    Returns:
    - pd.DataFrame: new df containing only the rows without the target word.
    """
    filtered_df = df[-df[col].str.contains(target_word, case=False)]

    return filtered_df

def filter_first_n(df, col, target_word, n=10000):
    """
    Parameters:
    - df (pd.DataFrame): df to be filtered.
    - col (str): name of note column.
    - target_word (str): word you do not want to keep.
    - n (int): # of chars from beginning of the string to check.

    Returns:
    - pd.DataFrame: new df containing only the rows without the target word in the first n chars.
    """
    filtered_df = df[df[col].str[:n].str.contains(target_word, case=False)]

    return filtered_df


OT_notes = pd.read_csv("../../harmonizing/ca_eeg_occupational_therapy.csv")

pd.set_option('display.max_columns', None)  



temp_df = pd.DataFrame(OT_notes['note_text'])

for src in ['physical', 'occupational']:
    notes = pd.read_csv("../../harmonizing/ca_eeg_"+src+"_therapy.csv")

    temp_df = notes[pd.notna(notes['note_text'])] # remove NaN notes

    # filter out canceled sessions and children
    filt_df = filter_by_word(temp_df, 'note_text', 'canceled session')
    filt_df = filter_by_word(filt_df, 'note_text', 'infant')
    filt_df = filter_by_word(filt_df, 'note_text', 'INFANT')
    filt_df = filter_by_word(filt_df, 'note_text', 'Benioff')
    filt_df = filter_by_word(filt_df, 'note_text', 'BENIOFF')

    # filter out notes that only reference PT/OT
    filt_df = filter_first_n(filt_df, 'note_text', src, n=100)

    # view how many notes are remaining
    print('\nTotal # of original ' + src + ' notes: '+str(pd.DataFrame(notes['note_text']).shape[0]))
    print('Total # of ' + src + ' notes after quality filtering: '+str(filt_df['note_text'].shape[0]))

    filt_df.to_excel('../../harmonizing/ca_eeg_'+src+'_therapy_filtered.xlsx')

