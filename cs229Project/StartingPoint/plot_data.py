# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# Distribution graphs (histogram/bar graph) of column data
def plot_per_column_distribution(the_df, n_graph_shown, n_graph_per_row):
    n_unq_val_per_col = the_df.nunique()
    # For displaying purposes, pick columns that have between 1 and 50 unique values
    the_df = the_df[[col for col in the_df if 1 < n_unq_val_per_col[col] < 50]]
    n_row, n_col = the_df.shape
    column_names = the_df.columns.to_list()
    # noinspection PyUnresolvedReferences
    n_graph_row = np.ceil(n_col / n_graph_per_row).astype(int)
    fig = plt.figure(num=None, figsize=(6 * n_graph_per_row, 8 * n_graph_row), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(n_col, n_graph_shown)):
        axs = fig.add_subplot(n_graph_row, n_graph_per_row, i + 1)
        column_df = the_df.iloc[:, i]  # current column
        if not np.issubdtype(column_df.dtype, np.number):
            cmap = plt.get_cmap('tab10')
            value_counts = column_df.value_counts()
            colors = [cmap(i) for i in range(len(value_counts))]
            value_counts.plot.bar(ax=axs, color=colors)
            axs.set_xticks(np.arange(len(value_counts)), value_counts.index.values.tolist(), rotation=90)
        else:
            column_df.hist(ax=axs)
        axs.set_ylabel('counts')
        axs.set_title(f'{column_names[i]} (column {i})')
    fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()


# Correlation matrix
def plot_correlation_matrix(the_df, the_graph_width):
    file_name = the_df.dataframeName
    the_df = the_df.dropna(axis=1)  # drop columns with NaN
    # keep columns where there are more than 1 unique values
    the_df = the_df[[col for col in the_df if the_df[col].nunique() > 1]]
    if the_df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or ' +
              'constant columns ({the_df.shape[1]}) is less than 2')
        return
    corr = the_df.corr(numeric_only=True)
    plt.figure(num=None, figsize=(the_graph_width, the_graph_width), dpi=80, facecolor='w', edgecolor='k')
    correlation_mat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(correlation_mat)
    plt.title(f'Correlation Matrix for {file_name}', fontsize=15)
    plt.show()


# Scatter and density plots
def plotScatterMatrix(the_df, the_plot_size, the_text_size):
    the_df = the_df.select_dtypes(include=[np.number])  # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    the_df = the_df.dropna(axis=1)
    # keep columns where there are more than 1 unique values
    the_df = the_df[[col for col in the_df if 1 < the_df[col].nunique()]]
    column_names = the_df.columns.to_list()
    if 10 < len(column_names):  # reduce the number of columns for matrix inversion of kernel density plots
        column_names = column_names[:10]
    the_df = the_df[column_names]
    # noinspection PyTypeChecker
    ax = pd.plotting.scatter_matrix(the_df, alpha=0.75, figsize=[the_plot_size, the_plot_size], diagonal='kde')
    correlations_data = the_df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate(f'Corr. coef = {correlations_data[i, j]:.3f}', (0.8, 0.2),
                          xycoords='axes fraction', ha='center', va='center', size=the_text_size)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


if __name__ == '__main__':
    nRowsRead = 1000  # None
    df1 = pd.read_csv('../DataSets/anime.csv', delimiter=',', nrows=nRowsRead)
    df1.dataframeName = 'anime.csv'
    nRow, nCol = df1.shape
    print(f'There are {nRow} rows and {nCol} columns')

    plot_per_column_distribution(df1, 10, 5)

    plot_correlation_matrix(df1, 8)

    plotScatterMatrix(df1, 9, 10)

    # %% --- second file ---
    nRowsRead = 1000  # None
    df2 = pd.read_csv('../DataSets/rating.csv', delimiter=',', nrows=nRowsRead)
    df2.dataframeName = 'rating.csv'
    nRow, nCol = df2.shape
    print(f'There are {nRow} rows and {nCol} columns')

    plot_per_column_distribution(df2, 10, 5)

    plot_correlation_matrix(df2, 8)

    plotScatterMatrix(df2, 9, 10)

    print('wow')
