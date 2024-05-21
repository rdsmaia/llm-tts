import pandas as pd
import os
from utils import clean_text

## Setting paths
RELATIVE_PATH_TO_CML_TTS_DATASET = 'datasets/cml_tts_dataset_portuguese_v0.1'
PATH_TO_CML_TTS_DATASET = os.getcwd() + '/' + RELATIVE_PATH_TO_CML_TTS_DATASET

## Reading datasets
cml_train = pd.read_csv(PATH_TO_CML_TTS_DATASET + '/' + 'train.csv', sep='|')
cml_test = pd.read_csv(PATH_TO_CML_TTS_DATASET + '/'  + 'test.csv', sep='|')
# concating dataaframes
cml_df = pd.concat([cml_train, cml_test]).reset_index() \
                                         .drop(columns='index')
# checking shape of new dataframe
assert cml_df.shape[1] == cml_train.shape[1]
assert cml_df.shape[1] == cml_test.shape[1]
assert cml_df.shape[0] == cml_train.shape[0] + cml_test.shape[0]

## Getting metadata
# adding audio_path
cml_df['meta_audio_path'] = cml_df.wav_filename.astype('str').apply(
    lambda w_filename: PATH_TO_CML_TTS_DATASET + '/' + w_filename
)
# getting and cleaning transcripted audio
cml_df['meta_text'] = cml_df.transcript_wav2vec.astype('str').apply(
    lambda text: text.strip().lower()
)
# getting speecher_id
cml_df['meta_speecher']  = cml_df.client_id.astype('int')


## Creating metadata
metadata_cml = cml_df[['meta_audio_path', 'meta_text', 'meta_speecher']]
metadata_cml.columns=['audio_path', 'text', 'speecher'] # renaming columns
metadata_cml.head()
# reseting speechers ids
original_qtd_speechers = len(metadata_cml.speecher.unique())
metadata_cml.loc[:,'speecher'] = metadata_cml.groupby(by='speecher').ngroup()
# checking we don't loose any speecher
assert original_qtd_speechers == len(metadata_cml.speecher.unique())

## Saving metada
metadata_cml.to_csv('cml_tts_pt_metadata.tsv',sep='|', index=False)
