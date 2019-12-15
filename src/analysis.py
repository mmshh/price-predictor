from src.utils import read_data_to_dataframe
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def bar_plot(data):
    ax = data.groupby('neighbourhood_group')['price'].agg(['mean']).plot(kind='bar')
    ax.set_ylabel("Average Price")
    ax = data.groupby('room_type')['price'].agg(['mean']).plot(kind='bar')
    ax.set_ylabel("Average Price")

    neighb = data.groupby('neighbourhood')['price'].agg(['mean'])
    top_neighb = neighb.sort_values(ascending=False, by='mean')
    ax = top_neighb.head(10).plot(kind='bar')
    ax.set_ylabel("Average Price")
    ax.set_xlabel("Top 10 Expensive Neighborhoods")
    ax = top_neighb.tail(10).plot(kind='bar')
    ax.set_ylabel("Average Price")
    ax.set_xlabel("Top 10 Cheapest Neighborhoods")

    agg = data.groupby('calculated_host_listings_count')['price'].agg(['mean'])
    agg.reset_index(level=0, inplace=True)
    ax = agg.plot.line(x='calculated_host_listings_count', y='mean')
    ax.set_xlabel("Number of Listings by Host")
    ax.set_ylabel("Average Price")

    agg = data.groupby('minimum_nights')['price'].agg(['mean'])
    agg.reset_index(level=0, inplace=True)
    ax = agg.plot.line(x='minimum_nights', y='mean')
    ax.set_xlabel("Minimum Stay")
    ax.set_ylabel("Average Price")

    agg = data.groupby('number_of_reviews')['price'].agg(['mean'])
    agg.reset_index(level=0, inplace=True)
    ax = agg.plot.line(x='number_of_reviews', y='mean')
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Average Price")

    agg = data.groupby('reviews_per_month')['price'].agg(['mean'])
    agg.reset_index(level=0, inplace=True)
    ax = agg.plot.line(x='reviews_per_month', y='mean')
    ax.set_xlabel("Reviews per Month")
    ax.set_ylabel("Average Price")

    agg = data.groupby('availability_365')['price'].agg(['mean'])
    agg.reset_index(level=0, inplace=True)
    ax = agg.plot.line(x='availability_365', y='mean')
    ax.set_xlabel("Number of Days Available per Year")
    ax.set_ylabel("Average Price")

    plt.show()
    # for col in data.columns:


def plot_target(data):
    data.price.hist(bins=200)
    plt.xlim(xmin=0, xmax=1000)
    plt.xlabel('price')
    plt.ylabel('count')
    plt.show()


def plot_corr(data):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns, annot=True, ax=ax)
    plt.show()


if __name__ == "__main__":
    data = read_data_to_dataframe("./AB_NYC_2019.csv")
    data = data[data['price'] > 0]
    print(data.describe())
    print(len(data.index)-data.count())
    plot_corr(data)
    bar_plot(data)
    plot_target(data)
    gooz=""