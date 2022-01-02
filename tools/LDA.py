# LIBRARY IMPORT
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sns.set()


def LDA_2D(
    data: pd.DataFrame, target: pd.Series, target_names: Iterable
) -> pd.DataFrame:
    """Does the 2D linear discriminant analysis of a dataset provided in the dataframe.

    Data is expressed according to it's two most appropriate axes to explain the variance,
    and this function generates the Pareto Diagram,
    """

    lda = LinearDiscriminantAnalysis(n_components=2)
    transformed_data = lda.fit(data, target).transform(data)

    colors = ["navy", "turquoise", "darkorange"]

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            transformed_data[target == i, 0],
            transformed_data[target == i, 1],
            alpha=0.8,
            color=color,
            label=target_name,
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA of IRIS dataset")

    return pd.DataFrame(transformed_data, index=data.index)


if __name__ == "__main__":
    iris = datasets.load_iris(as_frame=True)

    new_data = LDA_2D(iris.data, iris.target, iris.target_names)
    display(new_data)
    plt.show()
