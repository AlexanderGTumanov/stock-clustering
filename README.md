# Stock Clustering

A core principle of portfolio construction is diversification — spreading investments across different industries to reduce risk. In theory, this assumes that stocks within the same industry behave similarly, and that industries themselves are meaningfully distinct. In practice, however, stock behavior can deviate from industry norms, especially during periods of market instability. During such times, correlations across industries may increase, and traditional classifications may fail to capture meaningful behavioral groupings.

The goal of this project is to identify stock groupings more robustly by using unsupervised neural networks to cluster stocks based solely on their market behavior, without relying on industry labels. This is achieved by integrating the KMeans algorithm into the model’s training loop. The objective is to train the model on raw return patterns and uncover structure that may or may not align with standard sector classifications.

During stable periods, we expect the model’s clusters to correspond closely to industry groups — assuming those industries have distinct behavior. However, the more interesting insight comes from observing how these clusters evolve during turbulent times, such as the 2020 COVID-19 market crash. In these periods, we expect to see nontrivial cluster reassignments that indicate cross-industry behavioral convergence.

We analyze the time evolution of these learned clusters, interpret what the model has captured in terms of stock behavior, and investigate specific examples where the model’s output diverges from traditional industry boundaries. The goal is to understand when and why such divergences occur, and what they reveal about market structure beyond static classifications.

The project is organized into three main directories: the `/notebooks` folder, which contains the `stock_clustering.ipynb` notebook; the `/src` folder, which holds the source code file `model.py` with all the functions used in the notebook; and the `/data` folder, which contains the `default_tickers.py` file with the default selection of stock tickers pre-selected for this project, along with their industry labels. The time series data is retrieved from *yfinance*.

---

## What It Does

- Downloads historical stock prices from Yahoo Finance for a given selection of tickers from different industries.
- Builds and trains a PyTorch neural network (NN) that clusters these stocks together based on the behavior of their normalized log returns during a given period.
- Analyzes the time evolution of these clusters to identify stocks that deviate from the typical behavior of their industry.

---

## How to Use

1. Clone this repository:
   ```bash
   git clone <https://github.com/AlexanderGTumanov/stock-clustering>
   cd <stock-clustering>

---

## Contents of the Notebook

Notebook `/notebooks/stock_clustering.ipynb` is divided into two sections. In the first, we perform static analysis over a period of market stability, which allows the model to infer industry labels based solely on stock return behavior. In the second, we extend the analysis to a dynamic setting: using a rolling window approach, we train the model at each position of the window and align cluster assignments across time steps. This setup spans a long time period that includes the 2020 COVID crash, which allows us to study how the clustering of stocks evolves during periods of market stress.

---


## Contents of the `/data` folder

This folder contains the `default_tickers.py` module. In it, there is a balanced list of tickers from selected industries: *tech*, *energy*, *finance*, *healthcare*, *utilities*, *materials*, and *real estate*.

---

## Contents of the `/src` folder

The **model.py** file contains all the functions and tools used to perform analysis in the notebook. What follows is a brief description of them.

- **build_dataset(tickers, start, end, industries = None, shuffle = False, normalize = True, verbose = True)**:
   This function retrieves a collection of time series from **yfinance** between the dates given by **start** and **end** that correspond to tickers contained in **tickers**. It then converts them to log returns. **tickers** must be organized in the form of a dictionary: **{'industry label': [tickers]}**. The industry labels are not used during training, but are recorded for comparisons with the model's predictions. **industries** is an optional variable that can be given as a list of labels from the dictionary. Only stocks with these labels will be considered. If not provided, all the stocks from **tickers** will be retrieved. The dataset constructed will be balanced: i.e., the number of stocks per label will be the same. If some of the stocks fail to be retrieved, the function will discard stocks from other industries until the balance is restored. If **shuffle** is **True**, this will happen randomly. If **normalize** is **True** (recommended), all the log return series retrieved will be normalized. If **verbose** is **True**, then the function will print the size of the dataset once it's constructed.
  
  This function returns **X**, **y**, **t**, **industry_keys**, and **index**.<br>
  &nbsp;&nbsp;&nbsp;**X** is a torch tensor that contains all the log return series.<br>
  &nbsp;&nbsp;&nbsp;**y** and **t** are not necessary for the model to work, but become useful post-training. The former contains all the industry labels of the stocks in **X**, encoded as integers, while the latter records their tickers.<br>
  &nbsp;&nbsp;&nbsp;**industry_keys** is used to decode the values in **y** back into industry names.<br>
  &nbsp;&nbsp;&nbsp;**index** contains the index of the extracted time series.

- **prepare_dataloaders(X, batch_size = 16, valid_split = 0.2, seed = 42)**:
  Packages **X** into training and validation data loaders. **batch_size** controls the batch size, while **valid_split** specifies the proportion of data reserved for validation. The function returns *DataLoader()* objects **train_loader**, **valid_loader**.

- **DECModel(nn.Module)**, **Encoder(nn.Module)**, **ClusteringLayer(nn.Module)**:
  Deep Embedding Clustering (DEC) neural network. Consists of two layers: **Encoder** and **ClusteringLayer**. **Encoder** is a sequential layer that reduces the dimensionality of the input data by transferring it into a low-dimensional latent space. When fully trained, this space retains only the degrees of freedom that are essential for the clustering task. **ClusteringLayer** uses a Student's t-distribution kernel to compute the soft assignment of each latent point to a cluster. **DECModel** combines the two and returns **q**, which is the soft cluster assignment for each point, and **z**, the latent representations of the input data after passing through the encoder.

- **target_distribution(q)**:
  Computes the sharpened version of the soft cluster assignments **q**, used as the target distribution during DEC training to improve cluster purity. It amplifies confident assignments and suppresses uncertain ones.

- **train_model(train_loader, valid_loader, max_epochs = None, min_epochs = 100, patience = 50, lr = 1e-3, n_clusters = None, centers = None)**:
  Creates and trains the model on the data from **train_loader** and **valid_loader**. By default, the number of epochs is unrestricted. Instead, the model is coded to stop once the validation loss curve flattens or starts growing. **max_epochs** can be provided to hard-limit the number of epochs, while **min_epochs** is necessary because of the characteristic profile of the DEC loss curves: they tend to go up and spike early on, before starting to slowly go down. The **patience** parameter prevents the model's training from stopping due to random loss spikes. **lr** controls the learning rate, and **n_clusters** determines the number of clusters the stocks will be divided into. The **centers** parameter can be used to initialize cluster centers pre-training. If not provided, the centers will be initialized through KMeans. This parameter is very useful when determining the time evolution of cluster arrangements: cluster output at a given time window's position can be used as a seed for the next one. This greatly speeds up the model, because due to the temporal continuity, the new model's starting point lands very close to the vacuum configuration that it seeks. The function outputs the trained model **model**, along with a pandas dataframe **history** containing loss and validation histories throughout the epochs.
