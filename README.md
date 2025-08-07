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

The `model.py` contains all the functions and tools used to perform analysis in the notebook. What follows is a brief description of them. 

- **build_dataset(tickers, start, end, industries = None, shuffle = False, normalize = True, verbose = True)**:
  This function retrieves a collection of time series from *yfinance* between the dates given by **start** and **end** that correspond to tickers contained in **tickers**. It then converts them to log returns. **tickers** must be organized in the form of a dictionary: `{'industry label': [tickers]}`. The industry labels are not used during training, but are recorded for comparisons with model's predictions. **industries** is an optional variable which can be given as a list of labels from the dictionary. Only Stocks with these labels will be considered. If not provided, all the stocks from **tickers** will be retrieved. The dataset constructed will be balanced: i.e. the numer of stocks per label will be the same. If some of the stocks fail to be retrieved, the function will discard stocks from other industrues until the balance is restored. If **shuffle** is `True`, this will happen randomly. If **normalize** is `True` (recommended) al the log returns series retrieved will be normalized. If **verbose** is `True` then the function will print the size of the dataset once it's constructed.
  Tis function returns **X**, **y**, **t**, **industry_keys**, **index**. **X** is a torch tensor that contains all the log returns series; **y** and **t** are not necessary for the model to work, but become useful post-traing. The former contains all ithe industry lables of the stocks in **X**, encoded as integers, while the latter records their tickers. **industry_keys** is used to decode the values in **y** back into industry names. Lastly, **index** caontains the index of the extracted time series.
- **log_returns(series: pd.Series)**:
  Computes the logarithmic returns of **series**.
- **rolling_mean(series: pd.Series, window: int = None)**:
  Smooths **series** using a rolling mean. The window length is controlled by **window**; if not provided, it defaults to 1% of the total dataset length.
