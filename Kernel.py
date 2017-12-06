# 训练代码
def fit(...):
    ...
    self._alpha = np.zeros(len(x))
    self._b = 0.
    self._x = x
    # self._kernel 即为核函数，能够计算两组样本的核矩阵
    k_mat = self._kernel(x, x)
    for _ in range(epoch):
        err = -y * (self._alpha.dot(k_mat) + self._b)
        if np.max(err) < 0:
            continue
        mask = err >= 0
        delta = lr * y[mask]
        self._alpha += np.sum(delta[..., None] * k_mat[mask], axis=0)
        self._b += np.sum(delta)

# 预测代码
def predict(self, x, raw=False):
    x = np.atleast_2d(x).astype(np.float32)
    # 计算原样本与新样本的核矩阵并根据它来计算预测值
    k_mat = self._kernel(self._x, x)
    y_pred = self._alpha.dot(k_mat) + self._b
    if raw:
        return y_pred
    return np.sign(y_pred).astype(np.float32)