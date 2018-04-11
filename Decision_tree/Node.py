import numpy as np
from Decision_tree.Basic import Cluster

# 定义一个足够抽象的基类以囊括所有算法ID3、C4.5、CART
class CvDNode:
    '''
    初始化结构
    self._x, self._y:记录数据集的变量
    self._base, self.chaos:记录对数的底和当前的不确定性
    self.criterion, self.category:记录该Node计算信息增益的方法和所属类别
    self.left_child, self.right_child:针对连续型特征和CART、记录该Node的左右节点
    self._children, self.leafs:记录该Node的所有子节点和所有下属的叶节点
    self.sample_weight: 记录样本权重
    self.wc:记录各个维度的特征是否连续的列表(whether contiuous)
    self.tree:记录该Node所有的Tree
    self.feature_dim, self.tar, self.feats: 记录该Node划分标准的相关信息：
        self.feature_dim: 记录作为划分标准的特征所对应的维度j
        self.tar:针对连续型特征和CART，记录二分标准
        self.feats:记录该Node能进行选择的、作为划分标准的特征的维度
    self.parent, self.is_root:记录该Node的父节点及该节点是否为根节点
    self._depth, self.prev_feat: 记录该Node的深度和其父节点的划分标准
    self.is_cart:记录该Node是否使用了CART算法
    self.is_continuous: 记录该Node选择的划分标准对应的特征是否连续
    self.pruned:记录该Node是否已被剪掉，后面实现局部剪枝算法用到
    '''
    def __init__(self, tree = None, base = 2, chaos = None, depth = 0, parent = None, is_root = True, prev_feat = "Root"):
        self._x = self._y = None
        self.base, self.chaos = base, chaos
        self.criterion = self.category = None
        self.left_child = self.right_child = None
        self._children = self.leafs = {}, {}
        self.sample_weight = None
        self.wc = None
        self.tree = tree
        # 如果传入了Tree的话进行相应的初始化
        if tree is not None:
            # 由于数据预处理由Tree完成，所有各个维度的特征是否是连续型随机变量也是由Tree记录的
            self.wc = tree.whether_continuous
            # 这里的nodes变量是Tree中记录所有Node的列表
            tree.nodes.append(self)
        self.feature_dim, self.tar, self.feats = None, None, []
        self.parent, self.is_root = parent, is_root
        self._depth, self.prev_feat = depth, prev_feat
        self.is_cart = self.is_continuous = self.pruned = False
    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)
    # 重载__lt__方法， 使得Node之间可以比较谁更小、进行方便调试和可视化
    def __lt__(self, other):
        return self.prev_feat < other.prev_feat
    # 重载__str__和__repr__方法，同样为了便于调试和可视化
    def __str__(self):
        if self.category is None:
            return "CvDNode ({}) ({}->{})".format(self._depth, self.prev_feat, self.feature_dim)
        return "CvDNode ({}) ({}-> class:{})".format(self._depth, self.prev_feat, self.tree.label_dic[self.category])
    __repr__ = __str__
    # 定义children属性，区分开连续+CART情况和其余情况
    # 有了这个属性后，想要获得所有子节点就不用分情况讨论了
    @property
    def children(self):
        return {
            "left": self.left_child, "right": self.right_child
        }if (self.is_cart or self.is_continuous) else self._children
    # 递归定义height属性
    # 叶节点高度都定义为1，其余节点的高度定义为最高的子节点告诉+1
    @property
    def height(self):
        if self.category is not None:
            return 1
        return 1 + max([_child.height if _child is not None else 0 for _child in self.children.values()])
    # 定义info_dic(信息字典)属性，记录了该Node的主要信息
    # 在更新各个Node的叶节点时，被记录进各个self.leafs属性的就是该字典
    @property
    def info_dic(self):
        return {"chaos": self.chaos, "y":self._y}
    # 定义第一种停止准则：当特征维度为0或当前Node数据的不确定性小于阈值时停止
    # 同时，如果用户指定了决策树的最大深度，当该Node的深度太深也停止
    # 若满足停止条件，该函数会返回True，否则返回False
    def stop1(self, eps):
        if(
            self._x.shape[1] == 0 or (self.chaos is not None and self.chaos <= eps)
            or (self.tree.max_depth is not None and self._depth >= self.tree.max_depth)
        ):
            self._handle_terminate()
            return True
        return False
    def stop2(self, max_gain, eps):
        if max_gain <= eps:
            self._handle_terminate()
            return True
        return False
    # 利用bincount()方法定义根据该数据生成该Node所属类别的方法
    def get_category(self):
        return np.argmax(np.bincount(self._y))
    # 定义处理停止情况的方法，核心思想是把Node转为一个叶节点
    def _handle_terminate(self):
        # 首先要生成该Node所属类别
        self.category = self.get_category()
        # 然后一路回溯，更新父节点、父节点的父节点等，记录叶节点的属性leafs
        _parent = self.parent
        while _parent is not None:
            _parent.leafs[id(self)] = self.info_dic
            _parent = _parent.parent
    # 定义一个方法使其能将一个有子节点的Node转化为叶节点
    # 定义一个方法使其能挑选出最好的划分标准
    # 定义一个方法使其能根据划分标准进行生成
    def prune(self):
        # 调用相应方法进行计算该Node所属类别
        self.category = self.get_category()
        # 记录由于该Node转化为叶节点而被剪去的、下属叶节点
        _pop_lst = [key for key in self.leafs]
        # 一路回溯，更新各个parent的属性leafs
        _parent = self.parent
        while _parent is not None:
            for k in _pop_lst:
                # 删掉由于局部剪枝而被减掉的叶节点
                _parent.pop(k)
            _parent.leafs[id(self)] = self.info_dic
            _parent = _parent.parent
        # 调用mark_pruned方法将自己所有的子节点、子节点的子节点的pruned属性置为True，因为他们都被剪掉
        self.mark_pruned()
        # 重置各个属性
        self.feature_dim = None
        self.left_child = self.right_child = None
        self._children = {}
        self.leafs = {}
    # mark_pruned()用于给各个被局部剪枝剪掉的Node打一个标记、Tree可以根据这些标记将剪掉的Node从它记录所有Node的列表nodes删去
    def mark_pruned(self):
        self.pruned = True
        # 如果当前的子节点不是None的话，递归调用mark_pruned方法
        # 连续型特征和CART算法有可能导致children中出现None
        for _child in self.children.values():
            if _child is not None:
                _child.mark_pruned()
