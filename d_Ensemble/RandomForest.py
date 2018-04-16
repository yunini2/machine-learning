# 导入自己实现的决策树模型
from Decision_tree.Tree import *

from Util.Util import DataUtil
from Util.ProgressBar import ProgressBar

def rf_task(args):
    x, trees, n_cores = args
    return [tree.predict(x, n_cores = n_cores) for tree in trees]

class RandomForest(ClassifierBase):
    RandomForestTiming = Timing()
    # 建立一个决策树字典，以便调用
    _cvd_trees = {
        "id3": ID3Tree,
        "c45": C45Tree,
        "cart": CartTree
    }

    def __init__(self, **kwargs):
        super(RandomForest, self).__init__(**kwargs)
        self.tree, self._trees = "", []

        self._params["tree"] = kwargs.get("tree", "Cart")
        self._params["epoch"] = kwargs.get("epoch", 10)
        self._params["feature_bound"] = kwargs.get("feature_bound", "log")


    # 实现计算argmax(freq ck)的函数
    @staticmethod
    def most_appearance(arr):
        u, c = np.unique(arr, return_counts=True)
        return u[np.argmax(c)]

    # 默认使用10棵CART树，k=lod2d
    def fit(self, x, y, sample_weight = None, tree = "cart", epoch = 10, feature_bound = "log", *args, **kwargs):
        x, y = np.atleast_2d(x), np.array(y)
        n_sample = len(y)
        for _in in range(epoch):
            tmp_tree = RandomForest._cvd_trees[tree](*args, **kwargs)
            _indices = np.random.randint(n_sample, size=n_sample)
            if sample_weight is None:
                _local_weight = None
            else:
                _local_weight = sample_weight[_indices]
                _local_weight /= _local_weight.sum()

                tmp_tree.fit(x[_indices], y[_indices], sample_weight=_local_weight, feature_bound=feature_bound)
                self._trees.append(deepcopy(tmp_tree))

    # 对个体决策树进行简单组合
    def predict(self, x, get_raw_results=False, **kwargs):
        _matrix = np.array([_tree.predict(x) for _tree in self._trees]).T
        return np.array([RandomForest.most_appearance(rs) for rs in _matrix])


