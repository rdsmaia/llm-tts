import pandas as pd
import os
from utils import clean_text

## Setting paths
RELATIVE_PATH_TO_COMMON_VOICE_DATASET = 'datasets/common_voice_pt'
RELATIVE_PATH_TO_COMMON_VOICE_WAVES_DIR = RELATIVE_PATH_TO_COMMON_VOICE_DATASET + '/clips'
PATH_TO_COMMON_VOICE_DATASET = os.getcwd() + '/' + RELATIVE_PATH_TO_COMMON_VOICE_DATASET
PATH_TO_COMMON_VOICE_WAVES_DIR = os.getcwd() + '/' + RELATIVE_PATH_TO_COMMON_VOICE_WAVES_DIR
## Reading datasets
common_train = pd.read_csv(PATH_TO_COMMON_VOICE_DATASET + '/' + 'train.tsv', sep='\t')
common_test = pd.read_csv(PATH_TO_COMMON_VOICE_DATASET + '/'  + 'test.tsv', sep='\t')
# concating dataaframes
common_df = pd.concat([common_train, common_test]).reset_index() \
                                                  .drop(columns='index')
# checking shape of new dataframe
assert common_df.shape[1] == common_train.shape[1]
assert common_df.shape[1] == common_test.shape[1]
assert common_df.shape[0] == common_train.shape[0] + common_test.shape[0]

## Getting metadata
# adding audio_path
common_df['meta_audio_path'] = common_df.path.astype('str').apply(
    lambda w_filename: PATH_TO_COMMON_VOICE_WAVES_DIR + '/' + w_filename
)
# getting and cleaning transcripted audio
common_df['meta_text'] = common_df.sentence.astype('str').apply(
    lambda text: text.strip().lower()
)
# getting speecher_id
common_df['meta_speecher']  = common_df.client_id


## Creating metadata
metadata_common = common_df[['meta_audio_path', 'meta_text', 'meta_speecher']]
metadata_common.columns=['audio_path', 'text', 'speecher'] # renaming columns
metadata_common.head()
# reseting speechers ids
original_qtd_speechers = len(metadata_common.speecher.unique())
metadata_common.loc[:,'speecher'] = metadata_common.groupby(by='speecher').ngroup()
# checking we don't loose any speecher
assert original_qtd_speechers == len(metadata_common.speecher.unique())

## Saving metada
metadata_common.to_csv('common_voice_pt_metadata.tsv',sep='|', index=False)
