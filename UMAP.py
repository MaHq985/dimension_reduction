import pandas as pd
import umap
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    for drug in ['Cisplatin', 'Docetaxel', 'Etoposide']:
        sc_data = pd.read_csv('/home/mahq/SCAD-main/data/split_norm/' + drug + '_scFoundation/Target_expr_resp_z.' + drug + '.tsv',
                              header=0, index_col=0, sep='\t')

        x = sc_data.iloc[:, 1:]
        drug_response = sc_data.iloc[:, 0]

        # 调节超参数然后从中挑选出可视化效果最好的图片
        for n in range(2, 30):
            for random_stata in range(1, 100, 10):
                reducer = umap.UMAP(n_components=2, n_neighbors=n, random_state=random_stata, min_dist=0.1)
                y = reducer.fit_transform(x)

                labels = [0, 1]
                label_express = ['resistant', 'sensitive']
                colors = ['#007FFF', '#FF2400']

                plt.figure(figsize=(12, 9), dpi=200)  # figsize定义画布大小，dpi定义画布分辨率
                plt.title(f'UMAP Visualization for {drug}')
                plt.xlabel('UMAP component 1')
                plt.ylabel('UMAP component 2')

                for tlabel in labels:
                    # 读取数据
                    x_umap_data = y[drug_response == tlabel, 1]
                    y_umap_data = y[drug_response == tlabel, 2]
                    plt.scatter(x=x_umap_data, y=y_umap_data, s=25, c=colors[tlabel], label=label_express[tlabel])
                plt.legend(loc="upper right")
                plt.savefig(f'/home/mahq/MyProject/results/dimension reduction/UMAP/{drug}_{n}_{random_stata}.png')
                plt.show()


def main2():
    # 加载MNIST数据集
    mnist = datasets.load_digits()
    X = mnist.data
    y = mnist.target
    print(X.shape, y.shape)
    print(type(X), type(y))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reducer = umap.UMAP(n_components=2, n_neighbors=16, random_state=50, min_dist=0.1)
    embedding = reducer.fit_transform(X_scaled)

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    label_express = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    colors = ['#007FFF', '#FF2400', '#FF7F00', '#9932CD', '#D9D919', '#70DB93', '#238E68', '#8E236B', '#00FFFF', '#FF1CAE']

    plt.figure(figsize=(12, 9), dpi=200)  # figsize定义画布大小，dpi定义画布分辨率
    plt.title('UMAP Visualization for Mnist')
    plt.xlabel('UMAP component 1')
    plt.ylabel('UMAP component 2')

    for tlabel in labels:
        # 读取数据
        x_umap_data = embedding[y == tlabel, 0]
        y_umap_data = embedding[y == tlabel, 1]
        plt.scatter(x=x_umap_data, y=y_umap_data, s=25, c=colors[tlabel], label=label_express[tlabel])
    plt.legend(loc="upper right")
    plt.savefig('/home/mahq/MyProject/results/dimension reduction/UMAP/Mnist.png')
    plt.show()


if __name__ == '__main__':
    main()
