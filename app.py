import gradio as gr
import os
import shutil
import zipfile
from PIL import Image
import scanpy as sc
import numpy as np

def process_fn(fileobj):
    # Unzip file
    print("fileobj name:", fileobj.name)
    UPLOAD_FOLDER = './data'
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    shutil.copy(fileobj, UPLOAD_FOLDER)
    
    with zipfile.ZipFile(fileobj, 'r') as zip_ref:
        zip_ref.extractall('./data')

    # Analyze
    adata = sc.read_10x_mtx(
        './data', 
        var_names='gene_symbols', # use gene symbols for the variable names (variables-axis index)
        cache=True)
    
    sc.pp.filter_cells(adata, min_genes=200) # get rid of cells with fewer than 200 genes
    sc.pp.filter_genes(adata, min_cells=3) # get rid of genes that are found in fewer than 3 cells

    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    # print(adata.obs)

    upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, .98)
    lower_lim = np.quantile(adata.obs.n_genes_by_counts.values, .02)
    adata[adata.obs.index == 'AAACCCAAGCCTGTGC-1']
    adata = adata[(adata.obs.n_genes_by_counts < upper_lim) & (adata.obs.n_genes_by_counts > lower_lim)]
    adata = adata[adata.obs.pct_counts_mt < 20]
    sc.pp.normalize_total(adata, target_sum=1e4) #normalize every cell to 10,000 UMI
    sc.pp.log1p(adata) # change to log counts

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5) # these are default values
    adata.raw = adata # save raw data before processing values and further filtering
    adata = adata[:, adata.var.highly_variable] # filter highly variable
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt']) #Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed
    sc.pp.scale(adata, max_value=10) # scale each gene to unit variance
    sc.tl.pca(adata, svd_solver='arpack')
    # sc.pl.pca_variance_ratio(adata, log=True)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution = 0.25)
    sc.pl.umap(adata, color=['leiden'], save='_1.png')
    img = Image.open('figures/umap_1.png')

    return img

title = "Single Cell RNA UMAP Generator"
desc = "Upload a zip file and generate UMAP figure"

with gr.Blocks() as demo:
    result = gr.Image(label="Result", show_label=False)

    gr.Interface(
        title=title,
        description=desc,
        fn=process_fn,
        inputs=[
            "file",
        ],
        outputs=[result]
    )


demo.queue().launch()
