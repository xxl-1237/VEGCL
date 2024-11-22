from torch_geometric.datasets import WikiCS, Coauthor, Amazon, Planetoid
import torch_geometric.transforms as T

# pyg输入格式的数据集，你也可以多加入几个数据集，只需要改这个参数，可以参考下torch_geometric.dataset这个函数
# 在百度上面搜就有的
def get_dataset(path, name):
    assert name in ['WikiCS', 'Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo','Cora','CiteSeer', 'PubMed']
    name = 'dblp' if name == 'DBLP' else name
    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())
    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    if name == 'Cora':
        return Planetoid(root=path, name='Cora', transform=T.NormalizeFeatures())
    if name == 'CiteSeer':
        return Planetoid(root=path, name='CiteSeer', transform=T.NormalizeFeatures())
    if name == 'PubMed':
        return Planetoid(root=path, name='PubMed', transform=T.NormalizeFeatures())
