import cv2
from copy import deepcopy
from Decision_tree.Node import *
from Util.Timing import Timing
from Util.Bases import ClassifierBase

def cvd_task(args):
    x, clf, n_cores = args
    return np.array([clf.root.predict_one(xx) for xx in x])

# 定义一个足够抽象的Tree结构的基类以适应我们Node结构的基类
class CvDBase(ClassifierBase):
    '''
    初始化结构
    self.nodes:记录所有Node的列表
    self.roots:主要用于CART剪枝的属性，可先按下不表:用于存储算法过程中产生的各个决策树
    self.max_depth:记录决策树最大深度的属性
    self.root，self.feature_sets:根节点和记录可选特征维度的列表
    self.label_dic:和朴素贝叶斯里面相应的属性意义一致、是类别的转换字典
    self.prune_alpha, self.layers:主要用于ID3和C4.5剪枝两个属性，可先按下不表：惩罚因子、记录每一层的Node
    self.whether_continuous：记录着各个维度的特征是否连续的列表
    '''
    CvDBaseTiming = Timing()

    def __init__(self, whether_continuous = None, max_depth = None, node = None, **kwargs):
        super(CvDBase, self).__init__(**kwargs)
        self.nodes, self.layers, self.roots = [], [], []
        self.max_depth = max_depth
        self.root = node
        self.feature_sets = []
        # self.label_dic = {}
        self.prune_alpha = 1
        self.whether_continuous = whether_continuous
        self.y_transformer = None

        self._params["alpha"] = kwargs.get("alpha", None)
        self._params["eps"] = kwargs.get("eps", 1e-8)
        self._params["cv_rate"] = kwargs.get("cv_rate", 0.2)
        self._params["train_only"] = kwargs.get("train_only", False)
        self._params["feature_found"] = kwargs.get("feature_only", None)

    '''
    def __str__(self):
        return "CvDTree ({})".format(self.root.height)
    __repr__ = __str__
    '''

    # 进行数据预处理时自动识别出连续特征对应的维度
    def feed_data(self, x, continuous_rate = 0.2):

        # 利用set获取各个维度特征的所有可能去职
        self.feature_sets = [set(dimension) for dimension in x.T]
        data_len, data_dim = x.shape
        # 判断是否连续
        if self.whether_continuous is None:
            self.whether_continuous = np.array(
                [len(feat) >= int(continuous_rate * data_len) for feat in self.feature_sets]
            )
        else:
            self.whether_continuous = np.asanyarray(self.whether_continuous)
        self.root.feats = [i for i in range(x.shape[1])]
        self.root.feed_tree(self)
    # Grow
    @CvDBaseTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight = None, alpha = None, eps = None, cv_rate = None, train_only = None, feature_bound = None):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]
        if alpha is None:
            alpha = self._params["alpha"]
        if eps is None:
            eps = self._params["eps"]
        if cv_rate is None:
            cv_rate = self._params["cv_rate"]
        if train_only is None:
            train_only = self._params["train_only"]
        if feature_bound is None:
            feature_bound = self._params["feature_found"]
        self.y_transformer, y = np.unique(y, return_inverse=True)
        x = np.atleast_2d(x)
        # 根据特征个数定出alpha
        self.prune_alpha = alpha if alpha is not None else x.shape[1]/2
        # 如果需要划分数据集的话
        if not train_only and self.root.is_cart:
            # 根据cv_rate将数据集随机分成训练集和交叉验证集
            _train_num = int(len(x) * (1 - cv_rate))
            _indices = np.random.permutation(np.arange(len(x)))
            _train_indices = _indices[:_train_num]
            _test_indices = _indices[_train_num:]
            if sample_weight is not None:
                # 注意对切分后的样本权重做归一化处理
                _train_weights = sample_weight[_train_indices]
                _test_weights = sample_weight[_test_indices]
                _train_weights /= np.sum(_train_weights)
                _test_weights /= np.sum(_test_weights)
            else:
                _train_weights = _test_weights = None
            x_train, y_train, _train_weights = x, y, sample_weight
            x_cv = y_cv = _test_weights = None
            self.feed_data(x_train)
            # 调用根节点的生成算法
            self.root.fit(x_train, y_train, _train_weights, feature_bound, eps)
            # 调用对Node简直算法的封装
            self.prune(x_cv, y_cv, _test_weights)

    @CvDBaseTiming.timeit(level=3, prefix="[Util] ")
    # 调动Tree的reduce_nodes方法将被剪掉的Node从nodes中除去
    def reduce_nodes(self):
        for i in range(len(self.nodes)-1, -1, -1):
            if self.nodes[i].pruned:
                self.nodes.pop(i)

    # prune
    @CvDBaseTiming.timeit(level=4)
    # 对update_layers函数进行封装
    def _update_layers(self):
        # 根据整棵决策树的高度，在self.layers里面放相应数量的列表
        self.layers = [[] for _ in range(self.root.height)]
        self.root.update_layers()

    @CvDBaseTiming.timeit(level=1)
    def _prune(self):
        self._update_layers()
        _tmp_nodes = []
        # 更新完决策树每一层Node后，从后往前向_tmp_nodes中加Node
        for _node_lst in self.layers[::-1]:
            for _node in _node_lst[::-1]:
                if _node.category is None:
                    _tmp_nodes.append(_node)
        _old = np.array([_node.cost() + self.prune_alpha * len(_node.leafs) for _node in _tmp_nodes])
        _new = np.array([_node.cost(pruned=True) + self.prune_alpha for node in _tmp_nodes])
        # 使用mask变量存储_old和_new对应位置的大小关系
        _mask = _old >= _new
        while True:
            # 若只剩下根节点就退出循环
            if self.root.height == 1:
                break
            p = np.argmax(_mask)
            # 如果_new中有比_old中对应损失小的损失，则进行局部剪枝
            if _mask[p]:
                _tmp_nodes[p].prune()
                # 根据被影响了的node，更新_old _mask对应位置的值
                for i, node in enumerate(_tmp_nodes):
                    if node.affected:
                        _old[i] = node.cost() + self.prune_alpha * len(node.leafs)
                        _mask[i] = _old >= _new[i]
                        node.affected = False
                # 根据被剪掉的Node， 将各个变量对应的位置除去
                for i in range(len(_tmp_nodes)-1, -1, -1):
                    if _tmp_nodes[i].pruned:
                        _tmp_nodes.pop(i)
                        _old = np.delete(_old, i)
                        _new = np.delete(_new, i)
                        _mask - np.delete(_mask, i)
            else:
                break
        self.reduce_nodes()

    @CvDBaseTiming.timeit(level=1)
    def _cart_prune(self):
        # 暂时将所有节点记录所属Tree的属性置为None
        self.root.cut_tree() # cut_tree同样利用递归实现
        _tmp_nodes = [node for node in self.nodes if node.category is None]
        _thresholds = np.array([node.get_threshold() for node in _tmp_nodes])
        while True:
            # 利用deepcopy对当前根节点进行深拷贝，存入self.roots列表，如果前面没有把记录Tree的属性置为None
            # 这里就要对整个Tree做深拷贝，会引发严重的内存问题，还会拖慢速度
            root_copy = deepcopy(self.root)
            self.roots.append(root_copy)
            if self.root.height == 1:
                break
            p =np.argmin(_thresholds)
            _tmp_nodes[p].prune()
            for i, node in enumerate(_tmp_nodes):
                # 更新被影响的Node阈值
                if node.affected:
                    _thresholds[i] = node.get_threshold()
                    node.affected = False
            for i in range(len(_tmp_nodes)-1, -1, -1):
                # 去除掉各列表相应位置的元素
                if _tmp_nodes[i].pruned:
                    _tmp_nodes.pop(i)
                    _thresholds = np.delete(_thresholds, i)
        self.reduce_nodes()

    @CvDBaseTiming.timeit(level=3, prefix="[Util] ")
    def prune(self, x_cv, y_cv, weights):
        if self.root.is_cart:
            # 如果该Node使用CART剪枝，那么只有在确实传入了交叉验证集的情况下才能调用相关函数，否则没有意义
            if x_cv is not None and y_cv is not None:
                self._cart_prune()
                _arg = np.argmax([CvDBase.acc(y_cv, tree.predit(x_cv), weights) for tree in self.roots])
                _tar_root = self.roots[_arg]
                # 由于Node的feed_tree方法会递归的更新nodes属性，所以要先重置
                self.nodes = []
                _tar_root.feed_tree(self)
                self.root = _tar_root
            else:
                self._prune()
    # Util
    @CvDBaseTiming.timeit(level=1, prefix="[API] ")
    def predict_one(self, x):
        return self.y_transformer[self.root.predict_one(x)]
    @CvDBaseTiming.timeit(level=3, prefix="[API] ")
    def predict(self, x, get_raw_results=False, **kwargs):
        return self.y_transformer[self._multi_data(x, cvd_task, kwargs)]
    @CvDBaseTiming.timeit(level=3, prefix="[API] ")
    def view(self):
        self.root.view()

    @CvDBaseTiming.timeit(level=2, prefix="[API] ")
    def visualize(self, radius=24, width=1200, height=800, padding=0.2, plot_num=30, title="CvDTree"):
        self._update_layers()
        units = [len(layer) for layer in self.layers]

        img = np.ones((height, width, 3), np.uint8) * 255
        axis0_padding = int(height / (len(self.layers) - 1 + 2 * padding)) * padding + plot_num
        axis0 = np.linspace(
            axis0_padding, height - axis0_padding, len(self.layers), dtype=np.int)
        axis1_padding = plot_num
        axis1 = [np.linspace(axis1_padding, width - axis1_padding, unit + 2, dtype=np.int)
                 for unit in units]
        axis1 = [axis[1:-1] for axis in axis1]

        for i, (y, xs) in enumerate(zip(axis0, axis1)):
            for j, x in enumerate(xs):
                if i == 0:
                    cv2.circle(img, (x, y), radius, (225, 100, 125), 1)
                else:
                    cv2.circle(img, (x, y), radius, (125, 100, 225), 1)
                node = self.layers[i][j]
                if node.feature_dim is not None:
                    text = str(node.feature_dim + 1)
                    color = (0, 0, 255)
                else:
                    text = str(self.y_transformer[node.category])
                    color = (0, 255, 0)
                cv2.putText(img, text, (x-7*len(text)+2, y+3), cv2.LINE_AA, 0.6, color, 1)

        for i, y in enumerate(axis0):
            if i == len(axis0) - 1:
                break
            for j, x in enumerate(axis1[i]):
                new_y = axis0[i + 1]
                dy = new_y - y - 2 * radius
                for k, new_x in enumerate(axis1[i + 1]):
                    dx = new_x - x
                    length = np.sqrt(dx**2+dy**2)
                    ratio = 0.5 - min(0.4, 1.2 * 24/length)
                    if self.layers[i + 1][k] in self.layers[i][j].children.values():
                        cv2.line(img, (x, y+radius), (x+int(dx*ratio), y+radius+int(dy*ratio)),
                                 (125, 125, 125), 1)
                        cv2.putText(img, str(self.layers[i+1][k].prev_feat),
                                    (x+int(dx*0.5)-6, y+radius+int(dy*0.5)),
                                    cv2.LINE_AA, 0.6, (0, 0, 0), 1)
                        cv2.line(img, (new_x-int(dx*ratio), new_y-radius-int(dy*ratio)), (new_x, new_y-radius),
                                 (125, 125, 125), 1)

        cv2.imshow(title, img)
        cv2.waitKey(0)
        return img

class CvDMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        _, _node = bases

        def __init__(self, whether_continuous=None, max_depth=None, node=None, **_kwargs):
            tmp_node = node if isinstance(node, CvDNode) else _node
            CvDBase.__init__(self, whether_continuous, max_depth, tmp_node(**_kwargs))
            self._name = name

        attr["__init__"] = __init__
        return type(name, bases, attr)


class ID3Tree(CvDBase, ID3Node, metaclass=CvDMeta):
    pass


class C45Tree(CvDBase, C45Node, metaclass=CvDMeta):
    pass


class CartTree(CvDBase, CartNode, metaclass=CvDMeta):
    pass