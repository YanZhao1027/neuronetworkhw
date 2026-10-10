"""实验一 / 第一部分：NumPy 基础操作。可单独运行。"""
import numpy as np

print("【1】import numpy as np")
print("NumPy version =", np.__version__)

print("\n【2】一维数组 a")
a = np.array([4, 5, 6])
print("a =", a, "; type =", type(a), "; shape =", a.shape, "; a[0] =", a[0])

print("\n【3】二维数组 b")
b = np.array([[4, 5, 6], [1, 2, 3]])
print("b =\n", b, "; shape =", b.shape)
print("b[0,0] =", b[0, 0], "; b[0,1] =", b[0, 1], "; b[1,1] =", b[1, 1])

print("\n【4】特殊矩阵 c, d, e")
c = np.zeros((3, 3))
d = np.ones((4, 5))
e = np.eye(4)
print("c =", c.shape, "\n", c)
print("d =", d.shape, "\n", d)
print("e =", e.shape, "\n", e)

print("\n【5】arange / reshape / slicing")
f = np.arange(12)
print("f (original) =", f, "; shape =", f.shape)
f = f.reshape(3, 4)
print("f (reshaped) =\n", f, "; shape =", f.shape)
print("second row f[1,:] =", f[1, :])
print("last two columns f[:,2:] =\n", f[:, 2:])
print("third row last item f[2,-1] =", f[2, -1])
