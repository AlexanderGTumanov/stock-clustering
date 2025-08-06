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
