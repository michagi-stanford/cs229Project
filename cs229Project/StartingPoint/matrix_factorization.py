import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch import nn as nn

# --- Hyper params ---
n_min_movies = 20
n_min_audience = 50
batch_size = 8192
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learn_rate = 0.2
n_epochs = 10


class AnimeDataset(Dataset):
    def __init__(self, the_interaction_matrix):
        super().__init__()
        df_org = the_interaction_matrix.reset_index()
        self.df = df_org.melt(id_vars=['user_id'], var_name='anime_id', value_name='was_watched')
        self.x_user_movie = list(zip(self.df.user_id.values, self.df.anime_id.values))
        self.y_watched = self.df.was_watched.values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.x_user_movie[idx], self.y_watched[idx]


class MF(nn.Module):
    """ Matrix factorization model simple """

    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=emb_dim)
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=emb_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        element_product = (user_emb * item_emb).sum(1)
        return self.sigmoid(element_product)


if __name__ == '__main__':

    # %% ---  ---
    df_movie = pd.read_csv('../DataSets/anime.csv', delimiter=',', nrows=None)
    df_movie.dataframeName = 'anime.csv'
    df_user = pd.read_csv('../DataSets/rating.csv', delimiter=',', nrows=None)
    df_user.dataframeName = 'rating.csv'

    # %% --- Drop user ratings ---
    df_user = df_user.drop(columns=['rating'])

    # %% --- Merge data ---
    data = df_movie.merge(df_user)

    # %% --- Count the number of movies watched by each user ---
    user_counts = data.groupby('user_id').size()

    # %% --- Filter users with at least 20 watched movies ---
    filtered_users = user_counts[n_min_movies <= user_counts].index

    # %% --- Create a new dataframe with only these users ---
    data = data[data['user_id'].isin(filtered_users)]

    # %% --- Count the number of unique users who have watched each movie ---
    movie_counts = data.groupby('anime_id')['user_id'].nunique()

    # %% --- Filter for movies watched by at least 50 unique users ---
    popular_movies = movie_counts[n_min_audience <= movie_counts].index

    # %% --- Create a new dataframe with only these movies ---
    data = data[data['anime_id'].isin(popular_movies)]

    # %% --- Count the number of movies watched by each user ---
    user_counts = data.groupby('user_id').size()

    # %% --- Filter users with at least 20 watched movies ---
    filtered_users = user_counts[n_min_movies <= user_counts].index

    # %% --- Create a new dataframe with only these users ---
    data = data[data['user_id'].isin(filtered_users)]

    # %% --- Convert columns into categorical ---
    tmp_dict = defaultdict(LabelEncoder)
    cols_cat = ['user_id', 'anime_id']
    for col in cols_cat:
        tmp_dict[col].fit(data[col].unique())
        data[col] = tmp_dict[col].transform(data[col])
    data.head(3)

    # %% --- Split data into train / validation and create batches ---
    interaction_matrix = data.pivot_table(index='user_id', columns='anime_id', aggfunc='size', fill_value=0)
    i_train, i_val = train_test_split(interaction_matrix, test_size=0.2, random_state=42)

    ds_train = AnimeDataset(i_train)
    ds_val = AnimeDataset(i_val)
    dl_train = DataLoader(ds_train, batch_size, shuffle=True, num_workers=4)
    dl_val = DataLoader(ds_val, batch_size, shuffle=True, num_workers=4)

    # %% --- Matrix Factorization model ---
    n_users = len(data.user_id.unique())
    n_items = len(data.anime_id.unique())
    mdl = MF(n_users, n_items, emb_dim=32)
    mdl.to(device)
    print(mdl)

    # %% --- Train mode ---
    opt = torch.optim.AdamW(mdl.parameters(), lr=learn_rate)
    loss_fn = nn.MSELoss()
    epoch_train_losses, epoch_val_losses = [], []

    for i in range(n_epochs):
        train_losses, val_losses = [], []
        mdl.train()
        for xb, yb in dl_train:
            xUser = xb[0].to(device, dtype=torch.long)
            xItem = xb[1].to(device, dtype=torch.long)
            yRatings = yb.to(device, dtype=torch.float)
            y_hat = mdl(xUser, xItem)
            loss = loss_fn(y_hat, yRatings)
            train_losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        mdl.eval()
        for xb, yb in dl_val:
            xUser = xb[0].to(device, dtype=torch.long)
            xItem = xb[1].to(device, dtype=torch.long)
            yRatings = yb.to(device, dtype=torch.float)
            y_hat = mdl(xUser, xItem)
            loss = loss_fn(y_hat, yRatings)
            val_losses.append(loss.item())
        # Start logging
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_losses.append(epoch_train_loss)
        epoch_val_losses.append(epoch_val_loss)
        print(f'Epoch: {i}, Train Loss: {epoch_train_loss:0.1f}, Val Loss:{epoch_val_loss:0.1f}')

    # %% --- Plot results ---
    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.plot(epoch_train_losses, label='Train Loss')
    axs.plot(epoch_val_losses, label='Validation Loss')
    axs.legend()
    axs.set_xlabel('epoch', fontsize=12)
    axs.set_ylabel('loss', fontsize=12)
    plt.show()
