import pandas as pd

# Reading metadatas
cml_tts_metadata = pd.read_csv('./cml_tts_pt_metadata.tsv', sep='|')
common_voice_metadata = pd.read_csv('./common_voice_pt_metadata.tsv', sep='|')
### Concatenating metadatas
## we need to avoid mix speechers between datasets
org_qtd_ids_cml = len(cml_tts_metadata.speecher.unique())
org_qtd_ids_common = len(common_voice_metadata.speecher.unique())
bigger_cml_id = cml_tts_metadata.speecher.max()
common_voice_metadata.speecher = common_voice_metadata.speecher.apply(lambda id: bigger_cml_id + id + 1)
metadata = pd.concat([cml_tts_metadata, common_voice_metadata]).reset_index() \
                                                               .drop(columns='index')
# Checking if we successfully avoid mix speechers between datasets
assert len(metadata.speecher.unique()) == org_qtd_ids_cml + org_qtd_ids_common
# Saving metadata
metadata.to_csv('metadata.tsv', sep='|', index=False, header=False, encoding='utf-8')
