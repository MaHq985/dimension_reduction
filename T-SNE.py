import pandas as pd
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    for drug in ['Cisplatin', 'Docetaxel', 'Etoposide']:
        sc_data = pd.read_csv('/home/mahq/SCAD-main/data/split_norm/' + drug + '_scFoundation/Target_expr_resp_z.' + drug + '.tsv',
                              header=0, index_col=0, sep='\t')

        x = sc_data.iloc[:, 1:]
        drug_response = sc_data.iloc[:, 0]

        # 调节超参数并从中挑选出可视化效果最好的图片
        for perplexity in range(20, 100, 5):
            for seed in range(0, 100, 10):
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
                y = tsne.fit_transform(x)

                labels = [0, 1]
                label_express = ['resistant', 'sensitive']
                colors = ['#007FFF', '#FF2400']

                plt.figure(figsize=(12, 9), dpi=200)  # figsize定义画布大小，dpi定义画布分辨率
                plt.title(f'T-SNE Visualization for {drug}')
                plt.xlabel('T-SNE component 1')
                plt.ylabel('T-SNE component 2')

                for tlabel in labels:
                    # 读取数据
                    x_tsne_data = y[drug_response==tlabel, 0]
                    y_tsne_data = y[drug_response==tlabel, 1]
                    plt.scatter(x=x_tsne_data, y=y_tsne_data, s=25, c=colors[tlabel], label=label_express[tlabel])
                plt.legend(loc="upper right")
                plt.savefig(f'/home/mahq/MyProject/results/dimension reduction/T-SNE/{drug}_{perplexity}_{seed}.png')
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

    reducer = TSNE(n_components=2, perplexity=30, random_state=50)
    embedding = reducer.fit_transform(X_scaled)

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    label_express = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    colors = ['#007FFF', '#FF2400', '#FF7F00', '#9932CD', '#D9D919', '#70DB93', '#238E68', '#8E236B', '#00FFFF', '#FF1CAE']

    plt.figure(figsize=(12, 9), dpi=200)  # figsize定义画布大小，dpi定义画布分辨率
    plt.title('T-SNE Visualization for Mnist')
    plt.xlabel('T-SNE component 1')
    plt.ylabel('T-SNE component 2')

    for tlabel in labels:
        # 读取数据
        x_tsne_data = embedding[y == tlabel, 0]
        y_tsne_data = embedding[y == tlabel, 1]
        plt.scatter(x=x_tsne_data, y=y_tsne_data, s=25, c=colors[tlabel], label=label_express[tlabel])
    plt.legend(loc="upper right")
    plt.savefig('/home/mahq/MyProject/results/dimension reduction/T-SNE/Mnist.png')
    plt.show()


if __name__ == '__main__':
    main()
