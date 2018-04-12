import numpy as np
from Decision_tree.Basic import Cluster
import math
from Util.Metas import TimingMeta

# 定义一个足够抽象的基类以囊括所有算法ID3、C4.5、CART
class CvDNode(metaclass=TimingMeta):
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

    def __init__(self, tree = None, base = 2, chaos = None, depth = 0,
                 parent = None, is_root = True, prev_feat = "Root", **kwargs):
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
        return  self.__class__.__name__

    __repr__ = __str__

    @property
    def info(self):
        if self.category is None:
            return "CvDNode ({}) ({}->{})".format(self._depth, self.prev_feat, self.feature_dim)
        return "CvDNode ({}) ({}-> class:{})".format(self._depth, self.prev_feat, self.tree.label_dic[self.category])

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

    def fit(self, x, y, sample_weight, feature_bound = None,eps=1e-8):
        self._x, self._y = np.atleast_2d(x), np.array(y)
        self.sample_weight = sample_weight
        # 若满足第一停止条件，退出函数体
        if self.stop1(eps):
            return
        # 用该Node的数据实例化Cluster类以计算各种信息量
        _cluster = Cluster(self._x, self._y, sample_weight, self.base)
        # 对于根节点，需要额外计算其不确定性
        if self.is_root:
            if self.criterion == "gini":
                self.chaos = _cluster.gini()
            else:
                self.chaos = _cluster.ent()
        _max_gain, _chaos_lst = 0, []
        _max_feature = _max_tar = None
        feat_len = len(self.feats)
        if feature_bound is None:
            indices = range(0, feat_len)
        elif feature_bound == "log":
            indices = np.random.permutation(feat_len)[:max(1, int(math.log2(feat_len)))]
        else:
            indices = np.random.permutation(feat_len)[:feature_bound]
        tmp_feats = [self.feats[i] for i in indices]
        xt, feat_sets = self._x.T, self.tree.feature_sets
        bin_ig, ig = _cluster.bin_info_gain, _cluster.info_gain
        # 遍历还能选择的特征
        for feat in tmp_feats:
            # 如果是连续型特征或是CART算法，需要额外计算二分标准的取值集合
            if self.wc[feat]:
                _samples = np.sort(xt[feat])
                _set = (_samples[:-1] + _samples[1:]) * 0.5
            else:
                if self.is_cart:
                    _set = feat_sets[feat]
                else:
                    _set = None
            # 然后遍历这些二分标准并调用二类问题相关的计算信息量的方法
            if self.is_cart or self.wc[feat]:
                for tar in _set:
                    _tmp_gain, _tmp_chaos_lst = bin_ig(feat, tar, criterion=self.criterion,
                                                                       get_chaos_lst = True, continuous=self.wc[feat])
                    if _tmp_gain > _max_gain:
                        (_max_gain, _chaos_lst), _max_feature, _max_tar = (_tmp_gain, _tmp_chaos_lst), feat, tar
            # 对于离散型特征ID3和C4.5算法，调用普通的计算信息量的方法
            else:
                _tmp_gain, _tmp_chaos_lst = ig(
                    feat, criterion=self.criterion, get_chaos_lst=True, features=self.tree.feature_sets[feat])
                if _tmp_gain > _max_gain:
                    (_max_gain, _chaos_lst), _max_feature = (_tmp_gain, _tmp_chaos_lst), feat
        # 若满足第二种停止准则，则退出函数体
        if self.stop2(_max_gain, eps):
            return
        # 更新相关属性
        self.feature_dim = _max_feature
        if self.is_cart or self.wc[_max_feature]:
            self.tar = _max_tar
            # 调用根据划分标准进行生成的方法
            self._gen_children(_chaos_lst)
            # 如果该Node的左子节点和右子节点都是叶节点且所属类别一样，那么就将他们合并，局部剪枝
            if (self.left_child.category is not None and self.left_child.category == self.right_child.category):
                self.prune() # prune是对剪枝算法的封装
                # 调用Tree的相关方法，将被剪掉的该Node的左右子节点从Tree的记录所有Node列表nodes中除去
                self.tree.reduce_nodes()
        else:
            # 调用根据划分标准进行生成的方法
            self._gen_children(_chaos_lst)

    def _gen_children(self, chao_lst, feature_bound):
        feat, tar = self.feature_dim, self.tar
        self.is_continuous = continuous = self.wc[feat]
        features = self._x[..., feat]
        new_feats = self.feats.copy()
        if continuous:
            mask = features < tar
            masks = [mask, ~mask]
        else:
            if self.is_cart:
                mask = features == tar
                masks = [mask, ~mask]
                self.tree.feature_sets[feat].discard(tar)
            else:
                masks = None
        if self.is_cart or continuous:
            feats = [tar, "+"] if not continuous else ["{:6.4}-".format(tar), "{:6.4}+".format(tar)]
            for feat, side, chaos in zip(feats, ["left_child", "right_child"], chao_lst):
                new_node = self.__class__(
                    self.tree, self.base, chaos=chaos, depth=self._depth + 1, parent=self, is_root=False, prev_feat=feat
                )
                new_node.criterion = self.criterion
                setattr(self, side, new_node) # setattr用于为side设置属性值
            for node, feat_mask in zip([self.left_child, self.right_child], masks):
                if self.sample_weight is None:
                    local_weights = None
                else:
                    local_weights = self.sample_weight[feat_mask]
                    local_weights /= np.sum(local_weights)
                tmp_data, tmp_labels = self._x[feat_mask, ...], self._y[feat_mask]
                if len(tmp_labels) == 0:
                    continue
                node.feats = new_feats
                node.fit(tmp_data, tmp_labels, local_weights, feature_bound)
        else:
            new_feats.remove(self.feature_dim)
            for feat, chaos in zip(self.tree.feature_sets[self.feature_dim], chao_lst):
                feat_mask = features == feat
                tmp_x = self._x[feat_mask, ...]
                if len(tmp_x) == 0:
                    continue
                new_node = self.__class__(
                    tree=self.tree, base = self.base, chaos = chaos,
                    depth=self._depth + 1, parent=self, is_root=False, prev_feat=feat
                )
                new_node.feats = new_feats
                self.children[feat] = new_node
                if self.sample_weight is None:
                    local_weights = None
                else:
                    local_weights = self.sample_weight[feat_mask]
                    local_weights /= np.sum(local_weights)
                new_node.fit(tmp_x, self._y[feat_mask], local_weights, feature_bound)



    # 在Tree.py中调用feed_tree方法
    def feed_tree(self, tree):
        self.tree = tree # 让决策树中所有的Node记录他们所属的Tree结构
        self.tree.nodes.append(self) # 将自己记录在Tree中记录所有Node的列表nodes里
        self.wc = tree.whether_continuous # 根据Tree的相应属性更新记录连续特征的列表
        for child in self.children.values():
            if child is not None:
                child.feed_tree(tree)
    # 利用递归定义一个函数更新Tree的self.layers属性

    def update_layers(self):
        # 根据该node的深度，在self.layers对应位置的列表中记录自己
        self.tree.layers[self._depth].append(self)
        # 遍历所有子节点，完成递归
        for _node in sorted(self.children):
            _node = self.children[_node]
            if _node is not None:
                _node.update_layers()

    # 定义损失函数
    def cost(self, pruned = False):
        if not pruned:
            return sum([leaf["chaos"] * len(leaf["y"]) for leaf in self.leafs.values()])
        return self.chaos * len(self._y)


    # 在CART算法中，定义一个获取Node阈值的函数
    def get_threshold(self):
        return (self.cost(pruned=True) - self.cost()) / (len(self.leafs) -1)

    # 定义cut_tree
    def cut_tree(self):
        self.tree = None
        for child in self.children.values():
            if child is not None:
                child.cut_tree()


    def predict_one(self, x):
        if self.category is not None:
            return self.category
        if self.is_continuous:
            if x[self.feature_dim] < self.tar:
                return self.left_child.predict_one(x)
            return self.right_child.predict_one(x)
        if self.is_cart:
            if x[self.feature_dim] == self.tar:
                return self.left_child.predict_one(x)
            return self.right_child.predict_one(x)
        else:
            try:
                return self.children[x[self.feature_dim]].predict_one(x)
            except KeyError:
                return self.get_category()

    def predict(self, x):
        return np.array([self.predict_one(xx) for xx in x])

    def view(self, indent = 4):
        print(" " * indent * self._depth, self.info)
        for node in sorted(self.children):
            node = self.children[node]
            if node is not None:
                node.view()



class ID3Node(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = "ent"


class C45Node(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = "ratio"

class CartNode(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = "gini"
        self.is_cart = True

