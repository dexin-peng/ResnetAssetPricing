import gc
import pandas as pd
import numpy as np

print('Reading source sas file in chunks...', flush=True)
FactorDetails = pd.read_excel('source_data/Factor Details.xlsx')
parts = []
rows_read = 0
# Keep only the US panel in memory; do not materialize the full global pickle.
with pd.read_sas('source_data/t9ftbki1xauaox25.sas7bdat', format='sas7bdat', chunksize=100_000) as reader:
    for df in reader:
        rows_read += len(df)
        df_us = df.loc[(df['excntry']==b'USA') & (df['obs_main']==1) & (df['common']==1) & (df['exch_main']==1) & (df['primary_sec']==1)]
        del df

        X = df_us[FactorDetails.dropna(subset='abr_jkp').abr_jkp.reset_index(drop=True).values.tolist() + ['me']]
        y = df_us.ret_exc_lead1m

        permno_time = df_us[['permno', 'eom']].rename(columns={'eom':'DATE'}).dropna()
        permno_time['permno'] = permno_time['permno'].astype(int)

        y_valid_index = y.dropna().index
        permno_time_valid_index = permno_time.index
        valid_index = np.intersect1d(permno_time_valid_index, y_valid_index)

        X_valid = X.loc[valid_index]
        y_valid = y.loc[valid_index]
        permno_time_valid = permno_time.loc[valid_index]

        X_ext = pd.concat([X_valid, permno_time_valid], axis=1)
        panel_part = pd.concat([X_ext, y_valid], axis=1).reset_index(drop=True)
        if not panel_part.empty:
            parts.append(panel_part)
        del df_us, X, y, permno_time, X_valid, y_valid, permno_time_valid, X_ext, panel_part
        gc.collect()
        print(f'Read {rows_read:,}/{reader.row_count:,} rows', flush=True)

if not parts:
    raise ValueError('No eligible US observations found in the SAS file.')
datashare_with_return_me = pd.concat(parts, ignore_index=True)
del parts
gc.collect()
datashare_with_return_me.to_pickle('source_data/datashare_with_return.pkl')
