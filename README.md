# Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data
Source code for "Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data", published in NeurIPS 2021.

This project also provides pre-processed biased training samples for three small-size (Cora/Citeseer/PubMed) and two medium-size graph (Ogb-arxiv/Reddit) benchmarks. Besides, scalable biased sampler is a tool to create biased training sample for any large-size graphs.

## Installation

## Processed data
For three smaller GNN benchmarks, we provide 100 different biased training seeds under ``data/localized_seeds_{dataset}.p``. We also provide the tool to load our indexed network and you can test the performance of your GNN using our biased seeds.

## Quick Start
To reproduce the results of different GNN methods under biased training,

``
python main_gnn.py --n-epochs=200 --dataset=citeseer --gnn-arch=$METHOD --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --gpu=$GPU_ID
``

Option of ``$Method`` can be one of *graphsage, gat, ppnp, sgc*.


The mean accuracy (Mirco-F1) and standard deviation are summarized (with part of baselines) in the following tables:

| Dataset | Cora | Citeseer | Pubmed |
| --------  |----------|----------|----------| 
| GCN | 68.3 (4.3) |  62.1 (1.4) | 59.4 (4.5) |
| APPNP | 71.2 (5.0) | 63.7 (1.4) | 65.8 (4.7) |
| SR-GNN | 74.0 (3.3) | 67.2 (1.5) | 70.5 (3.0) |

The results in the table are sightly different from the paper, since we use a different random seed to generate biased training seeds.
More comprehensive results can be found in Table 2 of the paper.

## Larger Graph Experiments
