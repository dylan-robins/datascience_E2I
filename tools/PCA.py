# LIBRARY IMPORT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.decomposition import PCA

sns.set()


def PCA_2D(data: pd.DataFrame) -> pd.DataFrame:
    """Does the 2D principal component analysis of a dataset provided in the dataframe.

    Data is expressed according to it's two most appropriate axes to explain the variance,
    and this function generates the Pareto Diagram,
    """

    acp = PCA()
    transformed_data = acp.fit_transform(data)

    print(
        f"Variance explained by the two first dimensions: "
        f"{sum(acp.explained_variance_ratio_[:2]) * 100 :.2f} %"
    )

    print("Total variance:", sum(acp.explained_variance_))

    # Plot Pareto diagram to visualize explained variance contributions
    labels = [f"V{i}" for i, _ in enumerate(acp.explained_variance_ratio_)]
    plt.bar(labels, acp.explained_variance_ratio_, width=0.25, label="Variance ratio")
    plt.plot(
        labels, acp.explained_variance_ratio_.cumsum(), "r.-", label="Cumulative sum"
    )
    plt.title("Pareto Diagram")
    plt.legend()

    # Plot data in new base
    plt.figure()
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    for i, name in enumerate(data.index):
        plt.annotate(
            name,
            (transformed_data[i, 0], transformed_data[i, 1]),
            xytext=(2, 2),
            textcoords="offset points",
        )
    plt.title("Dataset in new axes")

    # Calculate correlations between new data (Xproj) and original columns (df):
    corvar = np.zeros((len(data.columns), 2))
    for i, _ in enumerate(data.columns):
        corvar[i, 0] = np.corrcoef(transformed_data[:, 0], data.iloc[:, i])[0, 1]
        corvar[i, 1] = np.corrcoef(transformed_data[:, 1], data.iloc[:, i])[0, 1]

    # Correlation circle
    _, axes = plt.subplots(figsize=(8, 8))
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)

    # Draw axes
    plt.plot([-1, 1], [0, 0], color="silver", linestyle="-", linewidth=1)
    plt.plot([0, 0], [-1, 1], color="silver", linestyle="-", linewidth=1)
    # Draw circle
    cercle = plt.Circle((0, 0), 1, color="blue", fill=False)
    axes.add_artist(cercle)
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title("Correlation circle")
    plt.scatter(corvar[:, 0], corvar[:, 1])
    # Draw data column names
    for j, _ in enumerate(data.columns):
        plt.annotate(
            data.columns[j],
            (corvar[j, 0], corvar[j, 1]),
            xytext=(2, 2),
            textcoords="offset points",
        )

    return pd.DataFrame(transformed_data, index=data.index)


if __name__ == "__main__":
    df = pd.read_excel(
        "datasets/Temperatures.xlsx", sheet_name=0, header=0, index_col=0
    )
    new_data = PCA_2D(df)
    display(new_data)

    plt.show()
