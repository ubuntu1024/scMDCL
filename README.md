# scMDCL
The source code and input data of scMDCL

## Requirement
- Pytorch --- 1.12.1
- Python --- 3.8.16
- Numpy --- 1.24.3
- Scipy --- 1.10.1
- Sklearn --- 1.2.2
- Munkres --- 1.1.4
- tqdm ---4.65.0
- Matplotlib ---3.7.1

## Usage
#### Clone this repo.
```
git clone https://github.com/ubuntu1024/scMDCL.git
```

#### Code structure
- ```data_loader.py```: loads the dataset and construct the cell graph
- ```opt.py```: defines parameters
- ```utils.py```: defines the utility functions
- ```encoder.py```: defines the AE, GAE and q_distribution
- ```ops_loss.py```: defines the Contrastive Learning loss
- ```scMDCL.py```: defines the architecture of the whole network
- ```main.py```: run the model
- ```run.py```: conduct parameter analysis and ablation studies 
- ```tsne_plot.ipynb```: t-SNE visualization

#### Example command
Take the dataset "PBMC-10k" as an example
```
python main.py --name PBMC-10k
```

## Data availability
|  Dataset              | Protocol   | Source |
| --------------------------- | ----------------------- | ----------------------- |
| ***PBMC-10k***             | ***10x Multiome***      | ***https://support.10xgenomics.com/single-cell-multiome-atac-gex/datasets/1.0.0/pbmc_granulocyte_sorted_10k*** |
| ***Ma-2020***             | ***SHARE-seq*** | ***https://scglue.readthedocs.io/en/latest/data.html***        |
| ***PBMC-3K***          | ***10x Multiome***      | ***https://www.10xgenomics.com/resources/datasets/pbmc-from-a-healthy-donor-no-cell-sorting-3-k-1-standard-2-0-0***     |
| ***PBMC-3K***          | ***10x Multiome***      | ***https://www.10xgenomics.com/resources/datasets/pbmc-from-a-healthy-donor-no-cell-sorting-3-k-1-standard-2-0-0***     |

## Comparison methods availability
|  Method              | Source |
| --------------------------- | ----------------------- |
| ***scziDesk***             | ***https://github.com/xuebaliang/scziDesk*** |
| ***scGAE***          | ***https://github.com/ZixiangLuo1161/scGAE***     |
| ***scTAG***              | ***https://github.com/Philyzh8/scTAG*** |
| ***DCCA***             | ***https://github.com/cmzuo11/DCCA***        |
| ***DSIR***             | ***https://github.com/Polytech-bioinf/Deep-structure-integrative-representation***        |
| ***DEMOC***             | ***https://github.com/LongLVv/DEMOC_code***        |
| ***scMCs***             | ***http://www.sdu-idea.cn/codes.php?name=ScMCs***        |

#### Quick start
```
python main.py --name dataset
```

