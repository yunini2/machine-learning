import math
import numpy as np

class Cluster:
    '''
    初始化结构
    self._x, self._y:记录数据集的变量
    self._counters: 类别向量的计数器，记录第i类数据的个数
    self._sample_weight: 记录样本权重属性
    self._con_chaos_cache, self._ent_cache, self._gini_cache: 记录中间结果的属性
    self._base: 记录对数的底的属性
    '''

    def __init__(self, x, y, sample_weight = None, base =2):
        # 输入的是Numpy向量
        self._x, self._y = x.T, y
        # 利用样本权重对类别向量y进行计数
        if sample_weight is None:
            self._counters = np.bincount(self._y)
        else:
            self._counters = np.bincount(self._y, weights=sample_weight * len(sample_weight))
        self._sample_weight = sample_weight
        self._con_chaos_cache = self._ent_cache = self._gini_cache = None
        self._base = base
    # 定义计算不确定的信息熵函数、gini系数
    def ent(self, ent = None, eps = 1e-12):
        # 如果已经计算过且调用时没有额外给各类别样本的个数，直接调用结果
        if self._ent_cache is not None and ent is None:
            return self._ent_cache
        _len = len(self._y)
        # 如果调用时没有给各类别样本的个数，就利用结构本身的计数器来获取相应个数
        if ent is None:
            ent = self._counters
        # 使用eps来让算法的数值稳定性更好
        _ent_cache = max(eps, -sum([_c / _len * math.log(_c / _len, self._base) if _c != 0 else 0 for _c in ent]))
        # 如果调用时没有给各类别样本的个数，就将计算好的信息熵储存下来
        if ent is None:
            self._ent_cache = _ent_cache
        return  _ent_cache
    # 定义计算基尼系数的函数
    def gini(self, p = None):
        if self._gini_cache is not None and p is None:
            return self._gini_cache
        if p is None:
            p = self._counters
        _gini_cache = 1 - sum((p / len(self._y)) ** 2)
        if p is None:
            self._gini_cache = _gini_cache
        return _gini_cache
    # 定义计算H(y|A)和Gini(y|A)的函数
    

