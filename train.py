import argparse
import os.path as osp
import random
from typing import Dict
from torch_geometric.utils import dropout_edge, dropout_node
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

def generate_Belta(eigValue):
    gap = 100
    a = list(np.linspace(eigValue[0], 5/eigValue[1], gap))
    b = list(np.linspace(a[-1], 10/eigValue[1], gap))
    return a[0:-1] + b

def new_generate_beta(lambda2):
    # you could set a small gap to aacclerate computing!
    gap = 100
    a = list(np.linspace(0, 5 / lambda2, gap))
    b = list(np.linspace(a[-1], 10 / lambda2, gap))
    return a[0:-1] + b

def VertexEnt(G,belta=None, perturb_strategy='default', printLog=False):
    """
    近似计算的方法计算节点纠缠度
    :param G:
    :return:
    """
    # 邻接矩阵
    # nodelist: list, optional:
    # The rows and columns are ordered according to the nodes in nodelist.
    # If nodelist is None, then the ordering is produced by G.nodes().
    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).todense()
    assert 0 in G, "Node 0 should be in the input graph!"
    assert np.allclose(A, A.T), "adjacency matrix should be symmetric"
    L = np.diag(np.array(sum(A)).flatten()) - A
    N = G.number_of_nodes()

    eigValue,eigVector = np.linalg.eigh(L)
    print("Finish calucating eigen values!")
    eigValue = eigValue.real
    eigVector = eigVector.real

    if belta is None:

        num_components = nx.number_connected_components(G)
        belta = new_generate_beta(eigValue[num_components])

    S = np.zeros(len(belta))
    for i in range(0, len(belta)):
        b = belta[i]
        Z_ = np.sum(np.exp(eigValue * -b))
        t = -b * eigValue
        S[i] = -sum(np.exp(t) * (t / np.log(2) - np.log2(Z_)) / Z_)
        #S[i] = -np.log2(sum((np.exp(t)/Z_))**2)

    print("Finish calucating spectral entropy!")

    lambda_ral = np.zeros((N,N))
    if perturb_strategy == 'default':
        for v_id in tqdm(range(0, N), desc="Computing eigenValues", unit="node"):

            neibour = list(G.neighbors(v_id))
            kx = G.degree(v_id)
            A_loc = A[neibour][:,neibour]
            N_loc = kx+np.sum(A_loc)/2
            weight=2*N_loc/kx/(kx+1)
            if weight == 1:
                lambda_ral[v_id] = eigValue
            else:
                neibour.append(v_id)
                neibour = sorted(neibour)
                dA = weight-A[neibour][:,neibour]
                dA = dA - np.diag([weight]*(kx+1))
                dL = np.diag(np.array(sum(dA)).flatten()) - dA
                for j in range(0,N):
                    t__= eigVector[neibour,j].T@dL@eigVector[neibour,j]
                    if isinstance(t__, float):
                        lambda_ral[v_id, j] = eigValue[j] + t__
                    else:
                        lambda_ral[v_id, j] = eigValue[j] + t__[0,0]
    elif perturb_strategy == 'remove':
        for v_id in tqdm(range(0, N), desc="Computing eigenValues for removed networks", unit="node"):
            neibour = list(G.neighbors(v_id))
            pt_A = copy.deepcopy(A)
            pt_A[v_id, :] = 0
            pt_A[:, v_id] = 0
            dA = pt_A - A
            dA = dA[neibour][:,neibour]
            dL = np.diag(np.array(sum(dA)).flatten()) - dA
            for j in range(0, N):
                lambda_ral[v_id, j] = eigValue[j] + (eigVector[neibour, j].T @ dL @ eigVector[neibour, j])[0, 0]

    E=np.zeros((len(belta),N))
    for x in tqdm(range(0, N), desc="Searching minium entanglement", unit="node"):
        xl_=lambda_ral[x,:]
        for i in range(0,len(belta)):
            b=belta[i]
            Z_=np.sum(np.exp(-b*xl_))
            t = -b *xl_
            E[i,x] = -sum(np.exp(t) * (t / np.log(2) - np.log2(Z_)) / Z_) - S[i]
            #E[i, x] = -np.log2(sum((np.exp(t) / Z_)) ** 2) - S[i]


    VE = np.min(E, axis=0)
    mean_tau = np.mean(np.array(belta)[np.argmin(E, axis=0)])
    print(f"VE mean_tau={mean_tau}")
    return VE

def get_ve_nodeList(graph:nx, VE:np, dismantling_threshold=0.01):
    """
    当网络中有较多相同的VE值时用该函数更合适，否则建议使用 get_ve_nodeList_quick 函数
    :param graph: Note that nodes should be numbered starting from 0 #注意graph中要求节点从0开始编号
    :param VE:
    :param dismantling_threshold:
    :return:
    """
    assert graph.has_node(0), "Nodes should be numbered starting from 0!"
    G = copy.deepcopy(graph)
    target_size = int(dismantling_threshold * G.number_of_nodes())
    if target_size <= 2:
        target_size = 2

    delque = np.argsort(VE) #对VE从小到大进行排序，并获取排序后的索引列表

    index_lists = [] #用于存储相同元素的索引
    # 遍历排序后的索引列表
    for i, index in enumerate(delque):
        if i == 0 or VE[index] != VE[delque[i - 1]]:
            # 若当前元素与前一个元素不相同，则创建新的索引列表
            index_lists.append([index])
        else:
            # 若当前元素与前一个元素相同，则将索引添加到当前列表中
            index_lists[-1].append(index)

    remove_list = [] #最终的移除顺序
    gcc_list = []
    for same_VE_list in index_lists:
        while len(same_VE_list)>0:
            max_index = np.argmax([G.degree[v] for v in same_VE_list])
            remove_node = same_VE_list[max_index]
            G.remove_node(remove_node)
            temp_gcc = get_gcc(G)
            if temp_gcc <= target_size:
                break
            remove_list.append(remove_node+1) #+1是因为 G中节点从0开始，而输出结果节点从1开始
            gcc_list.append(temp_gcc)
            del same_VE_list[max_index]

    return remove_list, gcc_list

def get_ve_nodeList_quick(graph:nx, VE:np, dismantling_threshold=0.01):
    """ quick version """
    assert graph.has_node(0), "Nodes should be numbered starting from 0!"
    G = copy.deepcopy(graph)
    target_size = int(dismantling_threshold * G.number_of_nodes())
    if target_size <= 2:
        target_size = 2

    ds = np.array([G.degree(v) for v in range(G.number_of_nodes())])
    delque = np.argsort(VE - ds / 100000) # 对VE从小到大进行排序，并获取排序后的索引列表
    remove_list = []  # 最终的移除顺序
    gcc_list = []

    for v in tqdm(delque, desc="Computing gcc", unit="node"):
        G.remove_node(v)
        temp_gcc = get_gcc(G)
        if temp_gcc <= target_size:
            break
        remove_list.append(v + 1)
        gcc_list.append(temp_gcc)

    return remove_list, gcc_list


def train(epoch: int) -> int:
    #模型训练
    model.train()
    optimizer.zero_grad() #把梯度置零
    # 节点
    #node_index_1 = cnd(data.edge_index, node_weight, p=param['cnd_drop_rate_1'], threshold=args.cnd_thr)
    # 抽取数据集中的邻接矩阵，这里的邻接矩阵是一个边索引的方式，细节你可以看PYG（pytorch——geometric）这里面有介绍这个edge_index
    edge_index_1 = ced(data.edge_index, data.edge_weight, p=param['ced_drop_rate_1'], threshold=args.ced_thr)
    #node_index_2 = cnd(data.edge_index, node_weight, p=param['cnd_drop_rate_2'], threshold=args.cnd_thr)
    edge_index_2 = ced(data.edge_index, data.edge_weight, p=param['ced_drop_rate_2'], threshold=args.ced_thr)
    # # 随机删边，返回经过删除后保留的边的索引
    # edge_index_1 = dropout_edge(data.edge_index, p=drop_edge_rate_1)[0]  # p是删边的比例
    # z1 = model(data.x, node_index_1)
    #z2 = model(data.x, edge_index_1)
    z1 = model(data.x, edge_index_1)
    z2 = model(data.x, edge_index_2)
    loss = model.loss(z1, z2, batch_size=0)

    loss.backward() # 反向传播
    optimizer.step() # 梯度更新
    return loss.item()

def test() -> Dict:
    # 测试节点
    model.eval()

    # 将torch的计算图固定，也就是no_grad操作
    with torch.no_grad():
        z = model(data.x, data.edge_index)
    res = {}
    seed = np.random.randint(0, 32767)
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1,
                           generator=torch.Generator().manual_seed(seed))
    evaluator = MulticlassEvaluator()


    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            cls_acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}',
                                     num_epochs=800)
            accs.append(cls_acc['acc'])
        acc = sum(accs) / len(accs)
    else:
        cls_acc = log_regression(z, dataset, evaluator, split='rand:0.1',
                                 num_epochs=3000, preload_split=split)
        acc = cls_acc['acc']
        f1 = cls_acc['f1']
    res["acc"] = acc
    res["f1"] = f1
    return res

# 将数据转换为NetworkX图，并去除自环
def to_networkx_no_selfloops(data):
    g = to_networkx(data, to_undirected=True)
    node_mapping = {}  # 用于映射新旧节点编号关系
    new_node_id = 1
    edges_old = list(g.edges)
    with open(f"datasets/{netname}/edges.txt", 'w') as f:
        for edge in edges_old:
            if edge[0] != edge[1]:  # 确保边没有自环
                if edge[0] not in node_mapping:
                    node_mapping[edge[0]] = new_node_id
                    new_node_id += 1
                if edge[1] not in node_mapping:
                    node_mapping[edge[1]] = new_node_id
                    new_node_id += 1
                g.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
                f.write(f"{node_mapping[edge[0]]} {node_mapping[edge[1]]}\n")
    return g

def to_linegraph(data):
    g = to_networkx(data, to_undirected=True)
    l_g=nx.line_graph(g) #转换为线图
    G_int=nx.convert_node_labels_to_integers(l_g,first_label=-2)
    with open(f"datasets/{netname}/lg_edges.txt", 'w') as f:
        for edge in G_int.edges():
            # 将边写入文件，格式为 "node1 node2"
            f.write(f"{edge[0]} {edge[1]}\n")
    return l_g

# 将数据转换为NetworkX线图，并去除自环
def to_linenetworkx(data):
    g = to_networkx(data, to_undirected=True)
    l_g=nx.line_graph(g) #转换为线图
    node_mapping = {}  # 用于映射新旧节点编号关系
    new_node_id = 1
    edges_old = list(l_g.edges)
    lg=nx.Graph()
    with open(f"datasets/{netname}/lg_edges.txt", 'w') as f:
        for edge in edges_old:
            if edge[0] != edge[1]:  # 确保边没有自环
                if edge[0] not in node_mapping:
                    node_mapping[edge[0]] = new_node_id
                    new_node_id += 1
                if edge[1] not in node_mapping:
                    node_mapping[edge[1]] = new_node_id
                    new_node_id += 1
                lg.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
                f.write(f"{node_mapping[edge[0]]} {node_mapping[edge[1]]}\n")
    return lg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--dataset_path', type=str, default="./datasets")
    parser.add_argument('--param', type=str, default='local:cora.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--verbose', type=str, default='train,eval')
    parser.add_argument('--cls_seed', type=int, default=12345)
    parser.add_argument('--val_interval', type=int, default=100)
    parser.add_argument('--ced_thr', type=float, default=0.7)
    parser.add_argument('--cnd_thr', type=float, default=0.75)
    #VE参数
    parser.add_argument('--dth', default=0.01, help='dismantling_threshold')
    parser.add_argument('--sort_strategy', default='default', choices=['default', 'quick'])  # 从VE值获取节点序列的策略
    parser.add_argument('--perturb_strategy', default='default', choices=['default', 'remove'])  # 网络扰动策略
    parser.add_argument('--belta', default=None)

    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'ced_drop_rate_1': 0.3,
        'ced_drop_rate_2': 0.4,
        'cnd_drop_rate_1': 0.1,
        'cnd_drop_rate_2': 0.1,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        't0': 500,
        'gamma': 1.,
    }
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)
    comment = f'{args.dataset}_node_{param["cnd_drop_rate_1"]}_{param["cnd_drop_rate_2"]}'\
              f'_edge_{param["ced_drop_rate_1"]}_{param["ced_drop_rate_2"]}'\
              f'_t0_{param["t0"]}_gamma_{param["gamma"]}'
    if not args.device == 'cpu':
        args.device = 'cuda'

    print(f"training settings: \n"
          f"data: {args.dataset}\n"
          f"device: {args.device}\n"
          f"batch size if used: {args.batch_size}\n"
          f"communal edge dropping (ced) rate: {param['ced_drop_rate_1']}/{param['ced_drop_rate_2']}\n"
          f"communal node dropping (cnd) rate: {param['cnd_drop_rate_1']}/{param['cnd_drop_rate_2']}\n"
          f"gamma: {param['gamma']}\n"
          f"t0: {param['t0']}\n"
          f"epochs: {param['num_epochs']}\n"
          )

    random.seed(12345)
    torch.manual_seed(args.seed)
    # for node classification branch
    if args.cls_seed is not None:
        np.random.seed(args.cls_seed)

    device = torch.device(args.device)
    path = osp.join(args.dataset_path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0] #得到第一个图的数据，因为使用的是一个图
    data = data.to(device)
    netname = args.dataset
    # 创建一个一维张量，初始化所有权重为0并，赋值给 data.edge_weight
    data.edge_weight = torch.zeros(data.edge_index.size(1),dtype=torch.float)
    # # 重新编号的NetworkX图
    # g = to_networkx_no_selfloops(data)
    # G = to_networkx(data, to_undirected=True)
    # # print(f"edges已成功写入到文件")
    # # #计算节点的VE
    # for netname in ['Cora']:#'CiteSeer', 'Coauthor-CS', 'Cora', 'PubMed'
    #     path = os.path.join('..', 'results', netname)
    #     path = os.path.join('results', netname)
    #     edge_path = os.path.join('datasets', netname,'edges.txt')
    #     os.makedirs(path, exist_ok=True)
    #     N = G.number_of_nodes()
    #     print("===================================================")
    #     print(f"There are {N} nodes and {G.number_of_edges()} edges in {netname}")
    #
    #     if args.perturb_strategy == 'default':
    #         ve_value_path = os.path.join(path, 'VE_value.txt')
    #
    #     elif args.perturb_strategy == 'remove':
    #         #ve_value_path = os.path.join(path, 'VE_remove_value.txt')
    #
    #     if not os.path.exists(ve_value_path):
    #         VE = VertexEnt(G, belta=args.belta, perturb_strategy=args.perturb_strategy)
    #         np.savetxt(ve_value_path, VE, fmt='%.16f')
    #
    #     else:
    #         VE = np.loadtxt(ve_value_path).tolist()

    with open(f'results/{netname}/VE_value.txt', 'r') as file:
    # 读取文件内容
        node_weight_lines = file.readlines()
    # 移除每行末尾的换行符，并转换为浮点数
    node_weight_str = [float(line.strip()) for line in node_weight_lines]
    # 将列表转换为NumPy数组
    node_weight_array = np.array(node_weight_str, dtype=np.float32)
    node_weight =torch.from_numpy(node_weight_array).to(data.edge_index.device)

    # # # 计算边的VE
    # g = to_networkx(data, to_undirected=True)
    # l_g = nx.line_graph(g)  # 转换为线图
    # L_G=nx.convert_node_labels_to_integers(l_g)
    # print(f"lg_edges已成功写入到文件")
    # for netname in ['Cora']:
    #     path = os.path.join('results', netname)
    #     edge_path = os.path.join('datasets', netname,'lg_edges.txt')
    #     os.makedirs(path, exist_ok=True)
    #     N = L_G.number_of_nodes()
    #
    #     print("===================================================")
    #     print(f"In the LineGraph, there are {N} nodes and {L_G.number_of_edges()} edges in {netname}")
    #
    #     if args.perturb_strategy == 'default':
    #         ve_value_path = os.path.join(path, 'lg_VE_value.txt')
    #     elif args.perturb_strategy == 'remove':
    #         ve_value_path = os.path.join(path, 'lg_VE_remove_value.txt')
    #     if not os.path.exists(ve_value_path):
    #         VE = VertexEnt(L_G, belta=args.belta, perturb_strategy=args.perturb_strategy)
    #         np.savetxt(ve_value_path, VE, fmt='%.16f')
    #     else:
    #         VE = np.loadtxt(ve_value_path).tolist()

    with open(f'results/{netname}/lg_VE_value.txt', 'r') as file:
        # 读取文件内容
        edge_weight_lines = file.readlines()
    # 移除每行末尾的换行符，并转换为浮点数
    edge_weight_str = [float(line.strip()) for line in edge_weight_lines]
    # 将列表转换为NumPy数组
    edge_weight_array = np.array(edge_weight_str)

    g = to_networkx(data, to_undirected=True)
    l_g = nx.line_graph(g)
    node_mapping = {}  # 用于映射新旧节点编号关系
    new_edgeweight_id = 0
    edges_old = list(l_g.edges)
    for edge in edges_old:
        if edge[0] != edge[1]:  # 确保边没有自环
            if edge[0] not in node_mapping:
                node_mapping[edge[0]] = edge_weight_array[new_edgeweight_id]
                new_edgeweight_id += 1
            if edge[1] not in node_mapping:
                node_mapping[edge[1]] = edge_weight_array[new_edgeweight_id]
                new_edgeweight_id += 1

    for value, key in node_mapping.items():
        value1,value2=value
        key=float(key)
        i=0
        while i<data.edge_index.size(1):
            if data.edge_index[0,i]==value1 and data.edge_index[1,i]==value2:
                data.edge_weight[i]= key
            if data.edge_index[0,i]==value2 and data.edge_index[1,i]==value1:
                data.edge_weight[i] =key
            i=i+1


    # 接下来是构建model，先创建了一个GSGCL需要的一个encoder编码器，然后构建了Adam优化器来对model进行优化
    encoder = Encoder(dataset.num_features,
                      param['num_hidden'],
                      get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']),
                      k=param['num_layers']).to(device)
    model = CSGCL(encoder,
                  param['num_hidden'],
                  param['num_proj_hidden'],
                  param['tau']).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=param['learning_rate'],
                                 weight_decay=param['weight_decay'])
    last_epoch = 0
    log = args.verbose.split(',')

    # 加下来这些代表训练步骤，epoch代表要运行的轮数
    for epoch in range(1 + last_epoch, param['num_epochs'] + 1):
        loss = train(epoch) # 训练model，返回loss用于梯度更新
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
        if epoch % args.val_interval == 0:
            res = test()
            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {res["acc"]},avg_f1 = {res["f1"]}')
