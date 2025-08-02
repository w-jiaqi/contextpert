#TODO test code and refactor into notebook

#-------------------------------------------------------------
# READ IN EXPRESSION DATA AS PANDAS AND FILTER TO LANDMARK
#-------------------------------------------------------------

import h5py
import pandas as pd
import sys

chunk_size = 10  

gctx_file_path = '/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx'
text_file_path = '/GSE92742_Broad_LINCS_gene_info_delta_landmark.txt'  

# Read the text file into a pandas DataFrame
gene_df = pd.read_csv(text_file_path, sep='\t')

# Extract the pr_gene_id values
pr_gene_ids = gene_df['pr_gene_id'].astype(str).tolist()
print(pr_gene_ids)
data_chunks = []

with h5py.File(gctx_file_path, 'r') as f:
    # Access the matrix dataset
    data = f['0/DATA/0/matrix']
    
    # Get the column and row headers
    row_headers = f['0/META/COL/id'][:].astype(str)
    col_headers = f['0/META/ROW/id'][:].astype(str)

    filtered_col_headers = [header for header in col_headers if header in pr_gene_ids]
    print(len(filtered_col_headers))
    
    # Iterate over the data in chunks
    for i in range(0, data.shape[0], chunk_size):
        data_chunk = data[i:i + chunk_size, :977]
        
        df_chunk = pd.DataFrame(data_chunk, columns=filtered_col_headers, index=row_headers[i:i + chunk_size])
        
        data_chunks.append(df_chunk)
        print(df_chunk.shape)

df = pd.concat(data_chunks)

#-------------------------------------------------------------------------
# ADD PERTURBATION INFO (DOSE, UNIT, QUALITY CONTROLS, PERT_TYPE, etc)
#-------------------------------------------------------------------------
sig_info = pd.read_csv('/GSE92742_Broad_LINCS_sig_info.txt', delimiter='\t')
sig_metrics = pd.read_csv('/GSE92742_Broad_LINCS_sig_metrics.txt', delimiter='\t')
info_exploded = info.assign(inst_id=sig_info['distil_id'].str.split('|')).explode('inst_id')

merged_df = pd.merge(df, info_exploded[['inst_id', 'sig_id']], on='inst_id', how='left')
merged_df = merged_df.drop_duplicates(subset='inst_id', keep='first')
merged_df = pd.merge(merged_df, sig_metrics[['distil_cc_q75', 'pct_self_rank_q25', 'sig_id']], on='sig_id', how='left')

merged_df.to_csv('full_lincs.csv')
