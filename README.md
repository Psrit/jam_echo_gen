# jam_echo_gen

为挑战赛问题一提供含噪声与干扰的AD数据仿真生成器。

## 安装

首先检查机器是否满足安装要求。当前支持的操作系统、CPU 架构和 Python 版本如下：


若满足上述要求，选择适用的 .whl 安装文件，并运行常规 `pip` 命令即可完成安装：
```shell
pip install jam_echo_gen.whl
```

## 运行

本扩展库提供一个公开函数接口：

```python
def echo_gen(case: int, num_subarrays: int, num_targets: int, num_jams: int) -> tuple
```

其参数含义如下：
- `case`: 测试场景类型编号，可选值为 0, 1, 2, 3，各测试类型含义请见下文说明；
- `num_subarrays`: 仿真的一维均匀等间距线阵所包含的子阵数，取值不能太小，具体下界可通过报错信息获知；
- `num_targets`: 场景中的目标个数，要求 >= 0；
- `num_jams`: 场景中的干扰个数，要求 >= 0；

返回值为如下信息组成的元组：
1. 载波频率 `freq` (Hz)；
2. 噪声功率 `noise_power` (Watts)；
3. 仿真生成的采样信号 `signal`，类型为复数 `numpy.ndarray`，形状为 `(num_working_subarrays, num_range_gates)`，其中 `num_working_subarrays` 为实际工作的子阵个数（取值范围：`[1, num_subarrays]`）；
4. 实际工作的子阵编号数组 `working_subarrays`，类型为整数 `numpy.ndarray`，形状为 `(num_working_subarrays,)`，其中元素与 `signal` 中每行分别一一对应。

仿真使用的一维均匀线阵始终取子阵间距为半波长。
四种测试场景含义分别如下：
- 场景 0：
- 场景 1：
- 场景 2：
- 场景 3：