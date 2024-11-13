<img src="Model image/The architecture of scMDCL.png" width="700" height="500" />
# scMDCL
The source code and input data of scMDCL

## Requirement
- Pytorch --- 2.0.1
- Python --- 3.11.4
- Numpy --- 1.24.4
- Scipy --- 1.13.0
- Sklearn --- 1.3.21
- Munkres --- 1.1.4
- tqdm ---4.66.1
- Matplotlib ---3.7.2

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
| ***Chen-2019***          | ***SHARE-seq***      | ***https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126074***     |
| ***SLN206***          | ***CITE-seq***      | ***https://github.com/YosefLab/totalVI_reproducibility/tree/master/data***     |

## Comparison methods availability
|  Method              | Source |
| --------------------------- | ----------------------- |
| ***scziDesk***             | ***https://github.com/xuebaliang/scziDesk*** |
| ***scGAE***          | ***https://github.com/ZixiangLuo1161/scGAE***     |
| ***DCCA***             | ***https://github.com/cmzuo11/DCCA***        |
| ***DEMOC***             | ***https://github.com/LongLVv/DEMOC_code***        |
| ***scMIC***             | ***https://github.com/Zyl-SZU/scMIC***        |
| ***scMFC***             | ***https://github.com/DayuHuu/scMFC***        |
| ***scEMC***             | ***https://github.com/DayuHuu/scEMC***        |

#### Quick start
```
python main.py --name dataset
```

