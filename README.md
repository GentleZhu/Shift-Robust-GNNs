# Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data
Source code for "Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data", published in NeurIPS 2021.

This project also provides pre-processed biased training samples for three small-size (Cora/Citeseer/PubMed) and two medium-size graph (Ogb-arxiv/Reddit) benchmarks. Besides, scalable biased sampler is a tool to create biased training sample for any large-size graphs.

## Installation
``
pip3 install -r requirements.txt
``
## Quick Start
 - [SR-GNN on Small Graph](#small-graph-benchmark)

 - [SR-GNN on Large Graph](#larger-graph-experiments)

 - [Use SR-GNN in your model](#apply-srgnn)


## Processed data
For three smaller GNN benchmarks, we provide 100 different biased training seeds under ``data/localized_seeds_{dataset}.p``. We also provide the tool to load our indexed network and you can test the performance of your GNN using our biased seeds. For two medium size graphs, we provide 10 different biased training seeds as ``data/{dataset}_{label_ratio}``.

We use pre-computed PPR vector in our model, which can be found at:

``intermediate``: https://drive.google.com/file/d/1zlJp9KEqxiApWX3IxC8nRx6tlkiqR-5V/view?usp=sharing

Please unzip the file into ``intermediate/``.

## Small Graph Benchmark
To reproduce the results of different GNN methods under biased training,

``
python main_gnn.py --n-epochs=200 --dataset=$DATASET --gnn-arch=$METHOD --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --gpu=$GPU_ID --SR=$SR
``

Option of ``$Method`` can be one of *graphsage, gat, ppnp, sgc*, ``$DATASET`` can be one of *cora/citeseer/pubmed*, Shift-Robust is applied when ``$SR`` is True.


The mean accuracy (Mirco-F1) and standard deviation are summarized (with part of baselines) in the following table:

| Dataset | Cora | Citeseer | Pubmed |
| --------  |----------|----------|----------| 
| GCN | 68.3 (4.3) |  62.1 (1.4) | 59.4 (4.5) |
| APPNP | 71.2 (5.0) | 63.7 (1.4) | 65.8 (4.7) |
| SR-APPNP | 74.0 (3.3) | 67.2 (1.5) | 70.5 (3.0) |

The results in the table are sightly different from the paper, since we use a different random seed to generate biased training seeds.
More comprehensive results can be found in Table 2 of the paper.

## Larger Graph Experiments
We use GCN as base model in larger graph experiments as an example of applying SR-GNN on deep GNNs. The mean accuracy (Mirco-F1) and standard deviation are summarized (with part of baselines) in the following table:
Running the follow command will first download the ogbn-arxiv and reddit dataset.

``
python main_gnn_large.py --n-epochs=100 --dataset=$DATASET --gnn-arch=gcn --n-repeats=10 --n-hidden=256 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --gpu=$GPU_ID --SR=$SR
``


| Dataset | Ogbn-Arxiv | Reddit |
| --------  |----------|----------|
| GCN | 59.3 (1.2) |  89.6 (0.9) |
| SR-GCN | 61.6 (0.6) | 91.3 (0.5) |

## Apply SRGNN
There are two hyper-parameter in our shift-robust function. One is λ for the regularization and B as the minimal weight of instance re-weighting.
