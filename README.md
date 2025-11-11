# Dist Launch - 一键启动集群训练工具

## 功能说明

支持类似pai-dlc等k8s的pytorch-job任务，进行批量一键启动任务：集群启动后进入pending等待，在rank0节点上一键启动整个集群的训练任务，方便调试，不会因为命令失败则整个集群被杀掉。

## 安装

安装后可直接使用`dist-launch`命令：

```bash
pip install dist-launch
dist-launch wait              # 进入等待模式
dist-launch run train.sh      # 启动训练
```

## 项目结构

```
dist-launch/
├── dist-launch            # 统一命令行入口（开发模式）
├── setup.py               # Python包安装配置
├── dist_launch/           # Python包目录
│   ├── __init__.py
│   ├── cli.py             # 命令行入口（包安装后使用）
│   ├── run.py             # 主执行脚本（被run.sh和cli.py调用）
│   ├── scripts/            # 执行脚本目录
│   │   ├── init_cluster.py   # 集群初始化（发现hostname + 分发SSH密钥）
│   │   ├── wait.sh            # 等待脚本（保持节点alive）
│   │   ├── run.sh            # 一键启动脚本（在rank0执行）
│   │   ├── test_ssh.sh       # SSH连接测试脚本
│   │   └── verify_ssh_keys.sh # SSH密钥验证脚本
│   └── lib/               # 库文件目录
│       ├── __init__.py
│       ├── cluster_manager.py   # 集群节点管理
│       ├── node_executor.py      # SSH节点执行器
│       └── host_discovery.py     # 节点发现逻辑
└── README.md
```

## 使用流程

### 1. 集群启动阶段（自动执行）

在DLC任务启动时，所有节点执行初始化脚本：

```bash
# 在所有节点上执行，如果已安装包：
dist-launch wait
```

或者集成到启动脚本中：
```bash
# 在train.sh或其他启动脚本中
python3 /path/to/dist-launch/dist_launch/scripts/init_cluster.py && \
bash /path/to/dist-launch/dist_launch/scripts/wait.sh
```

**功能：**
- `init_cluster.py`: 
  - 通过PyTorch allgather收集所有节点的hostname
  - 自动分发SSH公钥到所有节点（用于免密登录）
  - 在rank0节点的`/etc/hosts`中添加`rank-0`、`rank-1`等别名，方便快速登录
  - 保存cluster info到`/tmp/cluster_info.json`
- `wait.sh`: 保持节点运行，等待调试（支持信号清理，避免僵尸进程）

### 2. Debug阶段

登录rank0节点，准备`train.sh`训练脚本，然后执行：

```bash
dist-launch run train.sh
```

**功能：**
- 自动读取保存的cluster info（hostnames）
- 通过SSH在所有节点上执行`train.sh`（免密登录）
- 在rank0本地执行`train.sh`
- 阻塞等待所有节点完成

**命令行使用：**
- `dist-launch wait` - 进入等待模式（集群初始化后等待调试）
- `dist-launch run <train.sh>` - 一键启动训练（train.sh作为第一个参数）
- `dist-launch kill [--force]` - 一键停止所有训练进程（不会停止wait进程）
- `dist-launch nccl-tests [options]` - 运行NCCL性能测试

### 3. NCCL性能测试

使用 `dist-launch nccl-tests` 命令可以测试集群的网络带宽性能：

```bash
# 基本使用（默认测试 allreduce，大小 1,10,100,1000 MB）
dist-launch nccl-tests

# 测试多个操作
dist-launch nccl-tests --operations allreduce,allgather,broadcast

# 自定义测试大小
dist-launch nccl-tests --sizes 10,100,1000,10000

# 2机4卡测试
dist-launch nccl-tests --nper-node 4 --world-size 2

# 使用 float16 数据类型
dist-launch nccl-tests --dtype float16

# 增加迭代次数以获得更准确的结果
dist-launch nccl-tests --iterations 50
```

**输出说明：**
- **Algorithm bandwidth (algo_bw)**: 算法带宽，表示每秒钟实际处理的数据量（GB/s）
- **Bus bandwidth (bus_bw)**: 总线带宽，表示在总线上实际传输的数据量（GB/s）
  - 对于 Allreduce: bus_bw ≈ algo_bw * 2 * (nRanks - 1) / nRanks
  - 对于 Allgather/Broadcast: bus_bw ≈ algo_bw * (nRanks - 1)

**选项说明：**
- `--operations`: 要测试的操作，可选值：`allreduce`, `allgather`, `broadcast`（默认：`allreduce`）
- `--sizes`: 测试的数据大小（MB），逗号分隔（默认：`1,10,100,1000`）
- `--iterations`: 每个测试的迭代次数（默认：`20`）
- `--dtype`: 数据类型，可选值：`float32`, `float16`, `int32`（默认：`float32`）
- `--nper-node`: 每个节点的GPU数量（默认：自动检测）

**注意事项：**
- 测试结果会显示算法带宽和总线带宽两种指标
- 总线带宽通常高于算法带宽，因为它反映了总线上传输的总数据量
- 对于多机测试，确保所有节点都能访问相同的网络路径

## 环境变量

集群信息文件位置（可选）：
```bash
export CLUSTER_INFO_FILE="/tmp/cluster_info.json"  # 默认值
```

SSH公钥路径（可选）：
```bash
export SSH_PUBLIC_KEY="/path/to/ssh-key/id_rsa.pub"  # 默认值：/path/to/ssh-key/id_rsa.pub
```

每个节点会自动使用镜像设置的环境变量：
- `MASTER_ADDR`: Master节点地址
- `WORLD_SIZE`: 总节点数
- `RANK`: 当前节点rank
- `MASTER_PORT`: Master端口（训练阶段使用）
- `INIT_MASTER_PORT`: 初始化阶段使用的端口（可选，默认：MASTER_PORT + 1，避免与训练端口冲突）

**nccl-socket自动设置**：
- `NCCL_SOCKET_IFNAME`: 网络接口名称（自动通过`ifconfig`或`ip addr`检测，比如使用GB200接口模式时`enP*`，失败则回退到默认值`enP22p3s0f0np0,enP6p3s0f0np0`）
- `NCCL_IB_DISABLE`: 禁用InfiniBand（默认：`1`）

这些NCCL环境变量会在集群初始化时自动设置，可通过环境变量覆盖。`NCCL_SOCKET_IFNAME`会自动检测当前系统的网络接口，无需手动配置。

## SSH配置

可通过环境变量配置SSH参数（在运行 `dist-launch run` 之前设置）：

```bash
export SSH_KEY="/path/to/ssh-key/id_rsa"
export SSH_PORT="2025"                        # 默认：2025
export SSH_USER="root"                        # 默认：root
```

或者通过命令行参数指定：
```bash
dist-launch run train.sh --ssh-port 2025 --ssh-key /path/to/key --ssh-user root
```

**默认配置：**
- SSH密钥：`/path/to/ssh-key/id_rsa`
- SSH端口：`2025`
- SSH用户：`root`

**注意：** 
- SSH公钥会在集群初始化时自动分发到所有节点，实现免密登录
- 如果设置了 `SSH_PORT` 环境变量，运行时会显示实际使用的端口，便于验证配置是否正确

### 快速登录（使用 rank 别名）

集群初始化后，rank0 节点的 `/etc/hosts` 会自动添加 `rank-0`、`rank-1` 等别名，方便快速登录：

```bash
# 使用 rank 别名登录（仅在 rank0 节点上可用）
ssh -i /path/to/ssh-key/id_rsa \
    -p 2025 \
    root@rank-1
```

**注意：** rank 别名仅在 rank0 节点上可用，因为 `/etc/hosts` 只在 rank0 上更新。

### 验证SSH密钥分发

集群初始化后，可以验证SSH密钥是否正确分发：

```bash
# 验证所有节点的SSH密钥配置
bash dist_launch/scripts/verify_ssh_keys.sh
```

## 进程管理

`wait.sh`使用信号处理机制，避免产生僵尸进程：
- 支持SIGTERM、SIGINT、SIGQUIT信号
- 自动清理子进程
- 使用`tail -f /dev/null`保持进程alive，可被信号中断
