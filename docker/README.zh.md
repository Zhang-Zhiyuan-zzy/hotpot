# Hotpot Docker 测试环境

这里的配置用于在独立的 Conda 环境中安装和测试 Hotpot，支持 Python
3.9、3.10、3.11、3.12、3.13 和 3.14，默认版本为 Python 3.13。

Docker 构建过程会自动完成以下工作：

1. 基于 Miniforge 初始化 Linux 镜像并安装系统依赖。
2. 使用 `conda-forge` 创建名为 `hp` 的 Conda 环境。
3. 在 `hp` 中安装指定 Python 版本、Cairo、pip、setuptools 和 wheel。
4. 以 editable 模式安装 `hotpot-zzy` 及其 `dev` 测试依赖。
5. 执行 `pip check`、Hotpot/Open Babel 导入检查和打包元数据测试。
6. 容器启动时默认执行 `tests/run_coverage.sh`。

镜像由 Docker 守护进程管理，导出的 `.tar` 归档默认保存在
`~/docker/image`，不会写入 Hotpot 仓库。构建脚本也会拒绝把归档目录设置到
仓库内部。

## 一键构建并安装

在 Hotpot 仓库根目录执行：

```bash
./docker/build-hotpot-test-images.sh
```

这条命令默认构建 `hotpot-test:py313-conda`，并导出为：

```text
~/docker/image/hotpot-test-py313-conda.tar
```

选择一个、多个或全部 Python 版本：

```bash
# 单个版本
./docker/build-hotpot-test-images.sh 3.11

# 多个版本
./docker/build-hotpot-test-images.sh 3.9 3.13 3.14

# 全部 3.9-3.14
./docker/build-hotpot-test-images.sh all
```

如需把归档放到其他仓库外目录：

```bash
HOTPOT_DOCKER_IMAGE_DIR=/data/docker-images \
  ./docker/build-hotpot-test-images.sh 3.13
```

## 只使用 Docker 命令行

下面这一条 `docker build` 命令会完成系统初始化、Conda 环境创建、Hotpot
安装及构建期自检：

```bash
docker build \
  --build-arg PYTHON_VERSION=3.13 \
  --tag hotpot-test:py313-conda \
  --file docker/test/Dockerfile \
  .
```

需要导出镜像时，再执行：

```bash
mkdir -p "$HOME/docker/image"
docker save \
  --output "$HOME/docker/image/hotpot-test-py313-conda.tar" \
  hotpot-test:py313-conda
```

构建其他版本时，同时修改 `PYTHON_VERSION` 和标签中的版本，例如 Python
3.10 对应 `PYTHON_VERSION=3.10` 与 `hotpot-test:py310-conda`。

## 创建容器并运行测试

默认在 Python 3.13 容器中运行维护中的覆盖率测试：

```bash
./docker/run-hotpot-test-container.sh
```

指定版本或自定义测试命令：

```bash
./docker/run-hotpot-test-container.sh --python 3.11

./docker/run-hotpot-test-container.sh --python 3.13 \
  python -m pytest -q tests/test_packaging_metadata.py
```

脚本会把当前 Hotpot 源码挂载到 `/workspace/hotpot`。如果本机 Docker 中没有
对应镜像，但 `~/docker/image` 中存在归档，脚本会先自动执行 `docker load`。

直接使用 Docker 命令运行默认测试：

```bash
docker run --rm \
  --name hotpot-test-py313 \
  --volume "$PWD:/workspace/hotpot" \
  --workdir /workspace/hotpot \
  --env MPLCONFIGDIR=/tmp/hotpot-matplotlib \
  hotpot-test:py313-conda
```

创建一个保留的交互式测试容器：

```bash
docker run -it \
  --name hotpot-test-py313-shell \
  --volume "$PWD:/workspace/hotpot" \
  --workdir /workspace/hotpot \
  hotpot-test:py313-conda bash
```

镜像的入口命令通过 `conda run -n hp` 执行，因此默认测试、自定义 Python
命令和交互式 shell 都位于 `hp` Conda 环境中。

## 文件说明

- `docker/test/Dockerfile`：初始化系统、创建 Conda 环境、安装并自检 Hotpot。
- `docker/build-hotpot-test-images.sh`：构建一个、多个或全部 Python 镜像并导出。
- `docker/run-hotpot-test-container.sh`：加载所选镜像、挂载当前源码并创建测试容器。
