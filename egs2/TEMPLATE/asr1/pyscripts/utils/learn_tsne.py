import os
import re
import sys
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.manifold import TSNE

import torch

# def is_valid_embedding(embedding):
#     return not np.isnan(embedding).any()


def extract_id(label):
    match = re.search(r"id(\d+)", label)
    return match.group(1) if match else label


def load_embeddings(embd_dir: str) -> dict:
    embd_dic = OrderedDict(np.load(embd_dir))
    embd_dic2 = {}
    for k, v in embd_dic.items():
        if len(v.shape) == 1:
            v = v[None, :]
        embd_dic2[k] = torch.nn.functional.normalize(torch.from_numpy(v), p=2, dim=1).numpy()

    return embd_dic2


def main(args):
    # reducer = umap.UMAP()

    embd_dir = args[0]
    # trial_label = args[1]
    out_dir = args[1]  # 이미지가 저장될 곳

    embd_dic = load_embeddings(embd_dir)

    print(f"Number of keys: {len(embd_dic)}")  # 4708

    embeddings_for_umap = []
    ids = []
    for label, embedding in embd_dic.items():
        for embed in embedding:
            # if not np.isnan(embedding[0]).any():
            # embeddings_for_umap.append(embedding[0])
            embeddings_for_umap.append(embed)
            ids.append(extract_id(label))

    embeddings = np.array(embeddings_for_umap)

    # embd_dic = {key: value for key, value in embd_dic.items() if not np.isnan(value).any()}

    # for speaker, embedding in embd_dic.items():
    #     if is_valid_embedding(embedding[0]):
    #         valid_embeddings.append(embedding[0])
    #         valid_speakers.append(speaker)
    #     else:
    #         invalid_speakers.append(speaker)

    # embeddings = np.array(
    #     [embedding[0] for embedding in embd_dic.values() if is_valid_embedding(embedding[0])]
    # )

    # embeddings = np.array([embedding[0] for embedding in embd_dic.values()])
    print(f"filtered embedding shape: {embeddings.shape}")

    n_samples = embeddings.shape[0]
    perplexity = min(n_samples - 1, 30)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_results = tsne.fit_transform(embeddings)
    # umap_results = reducer.fit_transform(embeddings)

    unique_ids = list(set(ids))
    print(f"total speakers: {len(unique_ids)}")
    color_map = plt.cm.get_cmap("tab20")
    id_to_color = {id: color_map(i / len(unique_ids)) for i, id in enumerate(unique_ids)}

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
        # plt.scatter(umap_results[i, 0], umap_results[i, 1], color=color)
        # for i, result in enumerate(tsne_results):
        # plt.scatter(tsne_results[i, 0], tsne_results[i, 1])
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=color)
    # for i, (speaker, embedding) in enumerate(embd_dic.items()):
    #     plt.scatter(tsne_results[i, 0], tsne_results[i, 1], label=speaker)
    # plt.scatter(tsne_results[i, 0], tsne_results[i, 1])
    # plt.annotate(speaker, (tsne_results[i, 0], tsne_results[i, 1]))

    # plt.legend()
    # plt.title("UMAP visualization")
    plt.title("TSNE visualization")
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

    # with open(out_dir, "w") as f:
    #     pass
    # umap 이미지 저장


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# python pyscripts/utils/learn_tsne.py ./voxceleb1_test_embeddings.npz ./tsne_test.png
