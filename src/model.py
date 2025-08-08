import logging, yfinance as yf
import pandas as pd
import numpy as np
import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from collections import Counter, defaultdict
from matplotlib.lines import Line2D

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

def build_dataset(tickers, start, end, industries = None, shuffle = False, normalize = True, verbose = True):
    if industries is None:
        industries = list(tickers.keys())
    industry_keys = {i: ind for i, ind in enumerate(industries)}
    industry_indices = {ind: i for i, ind in industry_keys.items()}
    index = pd.DatetimeIndex([])
    ticker_map = {}
    data = {}
    for industry in industries:
        series_list = []
        ticker_list = []
        for ticker in tickers[industry]:
            try:
                df = yf.download(ticker, start = start, end = end, progress = False)
                if df.empty or 'Close' not in df.columns:
                    continue
                prices = df['Close'].squeeze()
                log_returns = np.log(prices).diff().dropna()
                if normalize:
                    mean = log_returns.mean()
                    std = log_returns.std()
                    if std == 0 or np.isnan(std):
                        continue
                    log_returns = (log_returns - mean) / std
                if len(log_returns) == 0 or np.isnan(log_returns).any():
                    continue
                series_list.append(log_returns)
                ticker_list.append(ticker)
                if len(index) < len(log_returns):
                    index = log_returns.index
            except Exception:
                continue
        data[industry] = series_list
        ticker_map[industry] = ticker_list
    for industry in industries:
        full_series = []
        full_tickers = []
        for s, t in zip(data[industry], ticker_map[industry]):
            if len(s) == len(index):
                full_series.append(s)
                full_tickers.append(t)
        data[industry] = full_series
        ticker_map[industry] = full_tickers
    min_count = min(len(data[industry]) for industry in industries)
    if min_count == 0:
        raise ValueError("One or more industries have no complete time series of the required length.")
    X = []
    y = []
    t = []
    for industry in industries:
        series_list = data[industry]
        ticker_list = ticker_map[industry]
        paired = list(zip(series_list, ticker_list))
        if shuffle:
            paired = random.sample(paired, min_count)
        else:
            paired = paired[:min_count]
        for series, ticker in paired:
            X.append(series.values)
            y.append(industry_indices[industry])
            t.append(ticker)
    if verbose:
        print(f"\nDataset constructed: {len(industries)} industries × {min_count} complete series each = {len(X)} total samples. Sample length is {len(index)}.")
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype = np.int64)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    perm = torch.randperm(len(X))
    X, y, t = X[perm], y[perm], [t[i] for i in perm.tolist()]
    return X, y, t, industry_keys, index

def prepare_dataloaders(X, batch_size = 16, valid_split = 0.2, seed = 42):
    total_samples = len(X)
    valid_len = int(total_samples * valid_split)
    train_len = total_samples - valid_len
    generator = torch.Generator().manual_seed(seed)
    X_train, X_valid = random_split(X, [train_len, valid_len], generator = generator)
    train_loader = DataLoader(X_train, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(X_valid, batch_size = batch_size, shuffle = False)
    return train_loader, valid_loader

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim = 10):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, latent_dim, dof = 1.0):
        super(ClusteringLayer, self).__init__()
        self.clusters = nn.Parameter(torch.randn(n_clusters, latent_dim))
        self.dof = dof

    def forward(self, z):
        dist = torch.sum((z.unsqueeze(1) - self.clusters) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.dof)
        q = q ** ((self.dof + 1.0) / 2.0)
        q = q / torch.sum(q, dim = 1, keepdim = True)
        return q
    
class DECModel(nn.Module):
    def __init__(self, input_dim, latent_dim, n_clusters, dof = 1.0):
        super(DECModel, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.clustering = ClusteringLayer(n_clusters, latent_dim, dof)
    
    def forward(self, x):
        z = self.encoder(x)
        q = self.clustering(z)
        return q, z
    
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_model(train_loader, valid_loader, max_epochs = None, min_epochs = 100, patience = 50, lr = 1e-3, n_clusters = None, centers = None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if n_clusters is None:
        if centers is None:
            raise ValueError("Either 'n_clusters' must be specified or 'centers' must be provided.")
        else:
            n_clusters = centers.shape[0]
    else:
        if centers is not None:
            if centers.shape[0] != n_clusters:
                print(f"Warning: number of cluster centers in the provided seed ({centers.shape[0]}) differs from 'n_clusters' ({n_clusters}). Using {centers.shape[0]} centers from the seed.")
            n_clusters = centers.shape[0]
    input_dim = next(iter(train_loader)).shape[1]
    model = DECModel(input_dim = input_dim, latent_dim = 10, n_clusters = n_clusters, dof = 1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    if centers is None:
        zs = []
        with torch.no_grad():
            for x_batch in train_loader:
                x_batch = x_batch.to(device)
                z = model.encoder(x_batch)
                zs.append(z.cpu())
        zs = torch.cat(zs, dim = 0)
        kmeans = KMeans(n_clusters = n_clusters, n_init = 20, random_state = 0)
        centers = kmeans.fit(zs.numpy()).cluster_centers_
        centers = torch.from_numpy(centers).float().to(device)
    model.clustering.clusters.data.copy_(centers.to(device))
    train_loss_history = []
    valid_loss_history = []
    best_valid_loss = float('inf')
    epochs_since_improvement = 0
    epoch = 0
    while True:
        model.train()
        train_loss = 0
        for x_batch in train_loader:
            x_batch = x_batch.to(device)
            q, _ = model(x_batch)
            p = target_distribution(q).detach()
            loss = F.kl_div(q.log(), p, reduction = 'batchmean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)
        train_loss /= len(train_loader.dataset)
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for x_batch in valid_loader:
                x_batch = x_batch.to(device)
                q, _ = model(x_batch)
                p = target_distribution(q)
                loss = F.kl_div(q.log(), p, reduction = 'batchmean')
                valid_loss += loss.item() * len(x_batch)
        valid_loss /= len(valid_loader.dataset)
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        epoch += 1
        if max_epochs and epoch >= max_epochs:
            break
        if epoch > min_epochs:
            if valid_loss < best_valid_loss - 1e-4:
                best_valid_loss = valid_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    break
    history = pd.DataFrame({"train": train_loss_history, "valid": valid_loss_history})
    return model, history

def predict_clusters(model, X):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.eval()
    with torch.no_grad():
        X_device = X.to(device)
        z = model.encoder(X_device).cpu().numpy()
        q, _ = model(X_device)
        clusters = torch.argmax(q, dim = 1).cpu().numpy()
    return z, clusters

def tickers_by_cluster(clusters, y, t, industry_keys):
    industry_cluster_map = defaultdict(list)
    for label, cluster, ticker in zip(y, clusters, t):
        industry = industry_keys[int(label)]
        industry_cluster_map[(industry, cluster)].append(ticker)
    industry_list = sorted(set(industry_keys.values()))
    cluster_list = sorted(set(clusters))
    df = pd.DataFrame(index = industry_list, columns = [f"cluster {cluster + 1}" for cluster in cluster_list])
    for industry in industry_list:
        for cluster in cluster_list:
            df.at[industry, f"cluster {cluster + 1}"] = industry_cluster_map.get((industry, cluster), [])
    return df

def mixing_table(clusters, y, industry_keys, verbose = False):
    industries = [industry_keys[int(label)] for label in y]
    counts = Counter(zip(industries, clusters))
    norm = len(y) // len(set(industries))
    industry_list = sorted(set(industry_keys.values()))
    cluster_list = sorted(set(clusters))
    proportions = {
        int(cluster): {
            industry: round(counts.get((industry, cluster), 0) / norm, 2)
            for industry in industry_list
        }
        for cluster in cluster_list
    }
    if verbose:
        columns = [f"cluster {cluster + 1}" for cluster in cluster_list]
        data = {
            industry: [proportions[cluster][industry] for cluster in cluster_list]
            for industry in industry_list
        }
        df = pd.DataFrame.from_dict(data, orient = 'index', columns = columns)
        print("\nProportion of each industry assigned to each cluster:\n")
        print(df.to_string())
    return proportions

def relabel_clusters(curr, prev, n_clusters, locked_assignments = None, current_assignments = None):
    overlap = np.zeros((n_clusters, n_clusters), dtype=int)
    for p, c in zip(prev, curr):
        overlap[p, c] += 1
    if locked_assignments and current_assignments:
        best_perm = None
        best_score = -1
        for perm in itertools.permutations(range(n_clusters)):
            ok = True
            for ind, locked_cluster in locked_assignments.items():
                j = current_assignments.get(ind)
                if j is not None and perm[j] != locked_cluster:
                    ok = False
                    break
            if not ok:
                continue
            score = sum(overlap[perm[j], j] for j in range(n_clusters))
            if score > best_score:
                best_score = score
                best_perm = perm
        if best_perm is not None:
            mapping = {j: best_perm[j] for j in range(n_clusters)}
        else:
            row_ind, col_ind = linear_sum_assignment(-overlap)
            mapping = dict(zip(col_ind, row_ind))
    else:
        row_ind, col_ind = linear_sum_assignment(-overlap)
        mapping = dict(zip(col_ind, row_ind))
    return np.array([mapping[label] for label in curr])

def cluster_evolution(tickers, start, end, n_clusters, industries = None, window = 60, step = 15, shuffle = False, lock_threshold = None):
    X, y, _, industry_keys, index = build_dataset(tickers, start, end, industries = industries, shuffle = shuffle, verbose = False)
    if len(X[0]) < window:
        raise ValueError("Window is larger than the series length.")
    n_steps = (len(X[0]) - window) // step + 1
    industry_list = sorted(set(industry_keys.values()))
    evolution = defaultdict(lambda: defaultdict(list))
    locked_assignments = {}
    prev_centers = None
    prev_clusters = None
    for k in tqdm(range(n_steps), desc = "Computing cluster-industry proportions"):
        left, right = step * k, step * k + window
        X_window = X[:, left:right]
        train_loader, valid_loader = prepare_dataloaders(X_window)
        if prev_centers is None:
            dec, _ = train_model(train_loader, valid_loader, n_clusters = n_clusters, min_epochs = 100, patience = 50, lr = 1e-3)
        else:
            dec, _ = train_model(train_loader, valid_loader, n_clusters = n_clusters, centers = prev_centers, min_epochs = 20, patience = 10, lr = 5e-4)
        _, clusters = predict_clusters(dec, X_window)
        mixing_pre = mixing_table(clusters, y, industry_keys, verbose = False)
        if prev_clusters is not None:
            if locked_assignments:
                current_assignments = {ind: max(range(n_clusters), key = lambda c: mixing_pre.get(c, {}).get(ind, 0.0)) for ind in locked_assignments}
                clusters = relabel_clusters(curr = clusters, prev = prev_clusters, n_clusters = n_clusters, locked_assignments = locked_assignments, current_assignments = current_assignments)
            else:
                clusters = relabel_clusters(curr = clusters, prev = prev_clusters, n_clusters = n_clusters)
        prev_centers = dec.clustering.clusters.detach().cpu().clone()
        prev_clusters = clusters.copy()
        mixing = mixing_table(clusters, y, industry_keys, verbose = False)
        for c in range(n_clusters):
            for ind in industry_list:
                evolution[c][ind].append(mixing.get(c, {}).get(ind, 0.0))
        if lock_threshold is not None:
            for ind in industry_list:
                if ind not in locked_assignments:
                    best_cluster = max(range(n_clusters), key = lambda c: mixing_pre.get(c, {}).get(ind, 0.0))
                    prop = mixing_pre.get(best_cluster, {}).get(ind, 0.0)
                    if prop >= lock_threshold:
                        locked_assignments[ind] = best_cluster
    multi_cols = pd.MultiIndex.from_product([range(n_clusters), industry_list], names = ["Cluster", "Industry"])
    data = []
    for t in range(n_steps):
        row = []
        for c in range(n_clusters):
            for ind in industry_list:
                row.append(evolution[c][ind][t])
        data.append(row)
    dates = pd.to_datetime([index[step * k + window // 2] for k in range(n_steps)])
    df = pd.DataFrame(data, columns = multi_cols, index = dates)
    return df

def plot_log_returns(ticker, start, end, normalize = False):
    if isinstance(ticker, str):
        ticker = [ticker]
    plt.figure(figsize = (10, 5))
    found = False
    for t in ticker:
        df = yf.download(t, start = start, end = end, progress = False)
        if df.empty or 'Close' not in df.columns:
            continue
        prices = df['Close'].dropna()
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()
        log_returns = np.log(prices).diff().dropna()
        if normalize:
            std = log_returns.std()
            mean = log_returns.mean()
            if std == 0 or np.isnan(std):
                continue
            log_returns = (log_returns - mean) / std
        if log_returns.empty:
            continue
        plt.plot(log_returns.index, log_returns.values, label = t)
        found = True
    if not found:
        raise ValueError("No valid data found for any tickers.")
    plt.xlabel('Date')
    plt.ylabel('Normalized Log Returns' if normalize else 'Log Returns')
    plt.title('Normalized Log Returns' if normalize else 'Log Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_loss_history(history):
    plt.figure(figsize = (10, 5))
    plt.plot(history["train"], label = "Train Loss")
    plt.plot(history["valid"], label = "Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_tsne_clusters(clusters, y, z, industry_keys):
    tsne = TSNE(n_components = 2, random_state = 42)
    z_tsne = tsne.fit_transform(z)
    _, axes = plt.subplots(1, 2, figsize = (14, 6))
    clusters = np.array(clusters)
    y = np.array(y)
    cluster_list = sorted(set(clusters))
    cluster_to_color = {c: plt.cm.tab10(i % 10) for i, c in enumerate(cluster_list)}
    cluster_colors = [cluster_to_color[c] for c in clusters]
    axes[0].scatter(z_tsne[:, 0], z_tsne[:, 1], color = cluster_colors, s = 10)
    axes[0].set_title('t-SNE of DEC Clusters')
    cluster_legend = [
        Line2D(
            [0], [0],
            marker = 'o',
            linestyle = '',
            markerfacecolor = cluster_to_color[c],
            markeredgecolor = 'none',
            markersize = 5,
            label = f"cluster {c + 1}"
        )
        for c in cluster_list
    ]
    axes[0].legend(handles = cluster_legend)
    industry_labels = range(len(industry_keys))
    industry_to_color = {i: plt.cm.tab10(i % 10) for i in industry_labels}
    industry_colors = [industry_to_color[i] for i in y]
    axes[1].scatter(z_tsne[:, 0], z_tsne[:, 1], color = industry_colors, s = 10)
    axes[1].set_title('t-SNE Colored by Industry')
    industry_legend = [
        Line2D(
            [0], [0],
            marker = 'o',
            linestyle = '',
            markerfacecolor = industry_to_color[i],
            markeredgecolor = 'none',
            markersize = 5,
            label = industry_keys[i]
        )
        for i in industry_labels
    ]
    axes[1].legend(handles = industry_legend)
    plt.tight_layout()
    plt.show()

def plot_cluster_proportions(df, rolling_window = None):
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("DataFrame must have a MultiIndex on columns with levels [Cluster, Industry].")
    clusters = sorted(set(df.columns.get_level_values(0)))
    industries = list(df.columns.levels[1])
    for cluster in clusters:
        plt.figure(figsize = (14, 6))
        for industry in industries:
            series = df[(cluster, industry)]
            if rolling_window is not None and rolling_window > 1:
                series = series.rolling(rolling_window, min_periods = 1).median()
            plt.plot(df.index, series, label = industry)
        plt.xlabel("Date")
        plt.ylabel("Proportion")
        plt.title(f"Cluster {cluster + 1}: Industry Proportions Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()