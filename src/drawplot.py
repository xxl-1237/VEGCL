import argparse
import os.path as osp
import random
from typing import Dict
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import to_networkx
import os
from src import *
from utils1 import fileUtils
from src.reinsertion import reinsertion, get_gcc
import copy
from tqdm import tqdm
import networkx as nx
import numpy as np
from utils1.graphUtils import *
from scipy.linalg import expm, eigh

device = torch.device('cpu')
path = osp.join("./datasets", 'Cora')
dataset = get_dataset(path, 'Cora')
data = dataset[0] #得到第一个图的数据，因为使用的是一个图
data = data.to(device)
g = to_networkx(data, to_undirected=True)
# G = nx.Graph()  # 创建无向图
# # 添加节点和边，例如：
# G.add_edge(1, 2)
# G.add_edge(1, 3)
# G.add_edge(2, 4)
centrality = nx.degree_centrality(g)
# with open(f'results/{netname}/lg_VE_value.txt', 'r') as file:
#     # 读取文件内容
#     edge_weight_lines = file.readlines()
# 绘制网络图
nx.draw(g, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
plt.show()