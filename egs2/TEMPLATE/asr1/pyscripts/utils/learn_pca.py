import os
import re
import sys
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import torch


def extract_id(label):
    match = re.search(r"id(\d+)", label)
    return match.group(1) if match else label


def load_embeddings(embd_dir: str) -> dict:
    embd_dic = OrderedDict(np.load(embd_dir))
    embd_dic2 = {}
    for k, v in embd_dic.items():
        if len(v.shape) == 1:
            v = v[None, :]
        embd_dic2[k] = torch.nn.functional.normalize(
            torch.from_numpy(v), p=2, dim=1
        ).numpy()
    return embd_dic2


def main(args):
    embd_dir = args[0]
    out_dir = args[1]

    embd_dic = load_embeddings(embd_dir)

    print(f"Number of keys: {len(embd_dic)}")

    embeddings_for_pca = []
    ids = []
    for label, embedding in embd_dic.items():
        for embed in embedding:
            # if not np.isnan(embedding[0]).any():
            # embeddings_for_pca.append(embedding[0])
            embeddings_for_pca.append(embed)
            ids.append(extract_id(label))

    embeddings = np.array(embeddings_for_pca)
    print(f"filtered embedding shape: {embeddings.shape}")

    # Use PCA instead of t-SNE
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(embeddings)

    unique_ids = list(set(ids))
    print(f"total speakers: {len(unique_ids)}")
    color_map = plt.cm.get_cmap("tab20")
    id_to_color = {
        id: color_map(i / len(unique_ids)) for i, id in enumerate(unique_ids)
    }

    legend_elements = []
    max_legend_items = 20  # Adjust this number to control legend size
    for i, id in enumerate(unique_ids):
        if i >= max_legend_items:
            break
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"voice{int(id)+1}",
                markerfacecolor=id_to_color[id],
                markersize=8,
            )
        )

    # plt.figure(figsize=(10, 8))
    plt.figure(figsize=(12, 10))
    for i, id in enumerate(ids):
        color = id_to_color[id]
        plt.scatter(pca_results[i, 0], pca_results[i, 1], color=color)

    plt.title("PCA visualization")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.legend(
        handles=legend_elements,
        title="Speaker IDs",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
    )
    plt.tight_layout()

    if not os.path.exists(os.path.dirname(out_dir)):
        os.makedirs(os.path.dirname(out_dir))

    plt.savefig(out_dir)
    plt.close()

    print("Successfully done")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# Usage: python pyscripts/utils/learn_pca.py ./voxceleb1_test_embeddings.npz ./pca_test.png
