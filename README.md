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

Notebook `/notebooks/stock_clustering.ipynb`.

---


## Contents of the `/data` folder

This folder contains the `default_tickers.py` module. In it, there is a list of tickers from selected industries: *tech*, *energy*, *finance*, *healthcare*, *utilities*, *materials*, and *real estate*.

---

## Contents of the `/src` folder

The `/src` folder contains the `model.py` module.

### `analysis.py`

This file contains tools for general time series analysis:

- **retrieve_stock(ticker: str, start, end)**:
  Retrieves the adjusted close price of the stock identified by **ticker** between the dates **start** and **end** using *yfinance*, and returns the data as a *pandas* Series.
- **log_returns(series: pd.Series)**:
  Computes the logarithmic returns of **series**.
- **rolling_mean(series: pd.Series, window: int = None)**:
  Smooths **series** using a rolling mean. The window length is controlled by **window**; if not provided, it defaults to 1% of the total dataset length.
- **stationarity_test(series: pd.Series)**:
  Performs stationarity tests on a time series using the *Augmented Dickey-Fuller (ADF)* and *Kwiatkowski-Phillips-Schmidt-Shin (KPSS)* tests. Prints the test statistic, p-value, and critical values for both tests, and provides a basic assessment of stationarity based on a 5% significance level.
- **plot_series(series: pd.Series)**:
  Plots **series**.
- **plot_acf_pacf(series: pd.Series, lags: int = 40)**:
  Plots the *autocorrelation function* and *partial autocorrelation function* of **series** up to the number of lags specified by **lags**.
- **compare_gaussian_nll(series: pd.Series, mean1: pd.Series, std1: pd.Series, mean2: pd.Series, std2: pd.Series, label1: str = "NN", label2: str = "ARCH")**:
  The main tool for model comparisons. Compares the Gaussian NLL for two forecasts over their overlapping portions. **series** is the observed time series against which both forecasts are evaluated; **mean1** and **std1** are the mean and standard deviation forecasts of the first model, while **mean2** and **std2** belong to the second model. By default, the function assumes the first forecast comes from the NN and the second from the ARCH model and labels them accordingly. Labels can be customized via **label1** and **label2**.

### `model_arch.py`

This file provides tools for fitting and analyzing ARMA/ARCH models:

- **fit_ARMA(series: pd.Series, p: int, q: int)**:
  Fits **series** using an ARMA(p, q) model.
- **fit_ARCH(series: pd.Series, lags: int = 0, p: int = 1, q: int = 1)**:
  Fits **series** using an AR(lags)-GARCH(p, q) model with normally distributed residuals.
- **print_diagnostics(model)**:
  Prints the *Log-Likelihood*, *Akaike Information Criterion (AIC)*, and *Bayesian Information Criterion (BIC)* for **model**.
- **forecast(model, horizon: int)**:
  Forecasts the mean and standard deviation of the series **horizon** steps beyond the model’s fit range, returning them as a tuple of pandas Series.
- **plot_series(series: pd.Series, model = None, k: int = 2, horizon: int = 0)**:
  Plots **series** along with the model’s mean prediction and a ±**k** standard deviation confidence interval. If **horizon** is provided, the function also forecasts **horizon** steps beyond the fit range. The fitted and unfitted portions of the series are plotted in different colors.

### `model_pytorch.py`

This file provides tools for fitting and analyzing neural networks for time series forecasting. The model takes a segment of a time series of length **window** and predicts its continuation for **horizon** steps ahead. It is trained on the historical data of the stock in question prior to the event being modeled.

- **ForecastingDataset(Dataset) — __init__(self, series: np.ndarray, window: int, horizon: int, scaler: StandardScaler)**:  
  A PyTorch dataset class that transforms **series** into training samples for the model. **scaler** is used to normalize the data, with `StandardScaler()` being the default choice.
- **ReturnForecaster(nn.Module) — __init__(self, window: int, horizon: int, hidden_sizes = (64, 64), dropout_rate = 0.2)**:  
  The neural network model. The number and size of hidden layers can be adjusted via **hidden_sizes**. **dropout_rate** controls overfitting. The model outputs **mean** and **logevar**, representing the predicted mean and logarithmic variance over the forecast horizon. Logarithmic variance is used instead of normal variance to avoid unnecessary exponentiation, which can degrade performance.
- **gaussian_nll_loss(mean, logvar, target)**:
  Computes the Gaussian negative log-likelihood of the fit, defined as
  
  $$
  L_\text{nll} = E\left[\frac{1}{2}\left(\text{log}\ 2\pi + \text{\bf logvar}\right) + \frac{1}{2}\left(\text{\bf target} - \text{\bf mean}\right)^2\ e^{-\text{\bf logvar}}\right].
  $$

- **prepare_dataloaders(series: pd.Series, window: int, horizon: int, batch_size: int = 64, valid_split: float = 0.2)**:
  Builds the input data for the model based on **series**, **window**, and **horizon**, splits it into training and validation sets, and packages them into dataloaders. **batch_size** controls the batch size, while **valid_split** specifies the proportion of data reserved for validation. The function returns *DataLoader()* objects **train_loader**, **valid_loader**, and the **scaler** used to normalize the data.
- **train_model(train_loader, valid_loader, epochs = 50, lr = 1e-3)**:
  Trains the model on data provided by **train_loader** and **valid_loader**. **epochs** specifies the number of training iterations, and **lr** is the learning rate. The function returns the trained model and a dataframe **pd.DataFrame({"train": train_loss, "valid": valid_loss})** that contains train and validation losses across epochs.
- **plot_loss_history(history: pd.DataFrame)**:
  Plots the training and validation loss curves over the course of training.
- **forecast(model: torch.nn.Module, series: pd.Series, scaler: StandardScaler)**:
  Uses **model** to generate forecasts based on **series**. The predictions are then denormalized using the **scaler** applied during training. The length of **series** must match the **model**’s window. The function converts the model’s **logvar** output to standard deviation and returns **mean** and **std** time series.
- **plot_series(series: pd.Series, model: torch.nn.Module, fit_series: pd.Series, scaler: StandardScaler, end_of_training = None, k: int = 2)**:
  Generates forecasts from fit_series **fit_series** and plots orignial **series** along with the model’s mean prediction and a ±**k** standard deviation confidence interval. The fitted and unfitted portions of the series are plotted in different colors. **scaler** is used to undo data normalization. The optional **end_of_training** parameter marks the boundary of the training dataset. The length of **fit_series** must match **model**’s window.
