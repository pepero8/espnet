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
    reducer = umap.UMAP()

    embd_dir = args[0]
    # trial_label = args[1]
    out_dir = args[1]  # 이미지가 저장될 곳

    embd_dic = load_embeddings(embd_dir)

    print(f"Number of keys: {len(embd_dic)}")  # 4708

    embeddings_for_umap = []
    ids = []
    for label, embedding in embd_dic.items():
        if not np.isnan(embedding[0]).any():
            embeddings_for_umap.append(embedding[0])
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

    # tsne = TSNE(n_components=2, random_state=42)
    # tsne_results = tsne.fit_transform(embeddings)
    umap_results = reducer.fit_transform(embeddings)

    unique_ids = list(set(ids))
    print(f"total speakers: {len(unique_ids)}")
    color_map = plt.cm.get_cmap("tab20")
    id_to_color = {id: color_map(i / len(unique_ids)) for i, id in enumerate(unique_ids)}

    plt.figure(figsize=(10, 8))
    for i, id in enumerate(ids):
        color = id_to_color[id]
        plt.scatter(umap_results[i, 0], umap_results[i, 1], color=color)
    # for i, result in enumerate(tsne_results):
    #     plt.scatter(tsne_results[i, 0], tsne_results[i, 1])
    # for i, (speaker, embedding) in enumerate(embd_dic.items()):
    #     plt.scatter(tsne_results[i, 0], tsne_results[i, 1], label=speaker)
    # plt.scatter(tsne_results[i, 0], tsne_results[i, 1])
    # plt.annotate(speaker, (tsne_results[i, 0], tsne_results[i, 1]))

    # plt.legend()
    plt.title("UMAP visualization")

    # Iterate through the dictionary
    # for key, value in embd_dic.items():
    # Print value type
    # print(f"Key: {key}")
    # print(f"  Value type: {type(value).__name__}")  # ndarray

    # Print shape if it's a numpy array or list
    # if np.isnan(value[0]).any():
    #     print(f"key: [{key}] has nan embedding")
    # if isinstance(value, np.ndarray):
    #     print(f"  Shape: {value.shape}")  # (3, 192)
    # else:
    #     print(f"  Shape: It's not an np.ndarray")
    # print()

    # with open(trial_label, "r") as f:
    #     lines = f.readlines()
    # trial_ids = [line.strip().split(" ")[0] for line in lines]
    # labels = [int(line.strip().split(" ")[1]) for line in lines]

    # enrolls = [trial.split("*")[0] for trial in trial_ids]
    # tests = [trial.split("*")[1] for trial in trial_ids]
    # assert len(enrolls) == len(tests) == len(labels)

    # scores = []
    # for e, t in zip(enrolls, tests):
    #     enroll = torch.from_numpy(embd_dic[e])
    #     test = torch.from_numpy(embd_dic[t])
    #     if len(enroll.size()) == 1:
    #         enroll = enroll.unsqueeze(0)
    #         test = enroll.unsqueeze(0)
    #     score = torch.cdist(enroll, test)
    #     score = -1.0 * torch.mean(score)
    #     scores.append(score.item())

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

# python pyscripts/utils/learn_umap.py ./voxceleb1_test_embeddings.npz ./umap_test.png
