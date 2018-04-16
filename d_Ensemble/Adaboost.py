from math import log
# 导入之前的朴素贝叶斯模型和决策树模型
from b_NaiveBayes.Vectorized.MultinomialNB import MultinomialNB
from b_NaiveBayes.Vectorized.GaussianNB import GaussianNB
from Decision_tree.Tree import *
from d_Ensemble.RandomForest import RandomForest
# 导入sklearn相应的模型
from _SKlearn.NaiveBayes import *
from _SKlearn.Tree import *

from Util.ProgressBar import ProgressBar

def boost_task(args):
    x, clfs, n_cores = args
    return [clf.predict(x, n_cores = n_cores) for clf in clfs]

class AdaBoost(ClassifierBase):
    AdaBoostTiming = Timing()
    # 弱分类器字典，如果想要测试新的弱分类器的话，只需将其加入该字典即可
    _weak_clf = {
        "SKMNB": SKMultinomialNB,
        "SKGNB": SKGaussianNB,
        "SKTree": SKTree,
        "MNB": MultinomialNB,
        "GNB": GaussianNB,
        "ID3": ID3Tree,
        "C45": C45Tree,
        "Cart": CartTree,
        "RF": RandomForest
    }

    """
    初始化结构
    AdaBoost框架的朴素实现
    使用的弱分类器需要有如下两个方法：
    'fit' 方法： 需要支持输入样本权重
    'predict'方法：用于返回预测的类别向量
    self._clf:记录弱分类器名称的变量
    self._clfs:记录弱分类器的列表
    self._clfs_weights:记录弱分类器"话语权"列表
    
    """

    def __init__(self, **kwargs):
        super(AdaBoost, self).__init__(**kwargs)
        self._clf, self._clfs, self._clfs_weights = "", [], []
        self._kwarg_cache = {}

        self._params["clf"] = kwargs.get("clf", None)
        self._params["epoch"] = kwargs.get("epoch", 10)
        self._params["eps"] = kwargs.get("eps", 1e-12)

    @property
    def params(self):
        rs = ""
        if self._kwarg_cache:
            tmp_rs = []
            for key, value in self._kwarg_cache.items():
                tmp_rs.append("{}:{}".format(key, value))
            rs += "(" + ";".join(tmp_rs) + ")"
        return rs
    @property
    def title(self):
        rs = "Classifier:{}; Num:{}".format(self._clf, len(self._clfs))
        rs += " " + self.params
        return rs

    @AdaBoostTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight = None, clf = None, epoch = None, eps = None, **kwargs):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]
        if clf is None:
            clf = self._params["clf"]
        if epoch is None:
            epoch = self._params["epoch"]
        if eps is None:
            eps = self._params["eps"]
        x, y = np.atleast_2d(x), np.asarray(y)
        # 默认使用10个CART决策树桩作为弱分类器
        if clf is None:
            clf = "Cart"
            kwargs = {"max_depth" : 1}
        self._clf = clf
        self._kwarg_cache = kwargs
        if sample_weight is None:
            sample_weight = np.ones(len(y)) / len(y)
        else:
            sample_weight = np.array(sample_weight)
        bar = ProgressBar(max_value=epoch, name="AdaBoost")
        # AdaBoost算法的主循环，epoch为迭代次数
        for _ in range(epoch):
            # 根据样本权重训练弱分类器
            tmp_clf = AdaBoost._weak_clf[clf](**kwargs)
            tmp_clf.fit(x, y, sample_weight)
            # 调用弱分类器的predict方法进行预测
            y_pred = tmp_clf.predict(x)
            # 计算加权错误率，考虑到数值的稳定性，在边值情况加了一个小的常熟
            em = min(max((y_pred != y).dot(sample_weight[:, None])[0], eps), 1 - eps)
            # 计算该弱分类器的话语权
            am = 0.5 * log(1 / em -1)
            # 更新样本权重并利用deepcopy将该弱分类器记录在列表总
            sample_weight *= np.exp(-am * y * y_pred)
            sample_weight /= np.sum(sample_weight)
            self._clfs.append(deepcopy(tmp_clf))
            self._clfs_weights.append(am)
            bar.update()
        self._clfs_weights = np.array(self._clfs_weights, dtype=np.float32)

    @AdaBoostTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results = False, bound = None, **kwargs):
        x = np.atleast_2d(x)
        if bound is None:
            clfs, clfs_weights = self._clfs, self._clfs_weights
        else:
            clfs, clfs_weights = self._clfs[:bound], self._clfs_weights[:bound]
        matrix = self._multi_clf(x, clfs, boost_task, kwargs)
        matrix *= clfs_weights
        rs = np.zeros(len(x))
        del matrix
        if not get_raw_results:
            return np.sign(rs)
        return rs



