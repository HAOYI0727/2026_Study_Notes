# Docker 快速配置深度学习环境 + 基础命令 + 常见报错

## 快速配置深度学习环境

- [镜像介绍](#镜像介绍)
  - [安装清单](#安装清单)
- [快速配置环境（两行命令）](#快速配置环境两行命令)
  - [1. 获取镜像（三选一）](#1-获取镜像三选一)
    - [国内镜像版](#国内镜像版)
    - [🪜科学上网版（直连）](#科学上网版直连)
    - [本地（网盘下载）](#本地网盘下载)
  - [2. 运行容器](#2-运行容器)
- [安装 Docker Engine](#安装-docker-engine)
  - [卸载旧版本](#卸载旧版本)
  - [使用 apt 仓库安装](#使用-apt-仓库安装)
- [GPU 驱动安装](#gpu-驱动安装)
- [安装 NVIDIA Container Toolkit](#安装-nvidia-container-toolkit)
- [拉取并运行 PyTorch Docker 镜像](#拉取并运行-pytorch-docker-镜像)

---

## 基础命令
- [Docker 快速配置深度学习环境 + 基础命令 + 常见报错](#docker-快速配置深度学习环境--基础命令--常见报错)
  - [快速配置深度学习环境](#快速配置深度学习环境)
  - [基础命令](#基础命令)
  - [常见报错](#常见报错)
- [Docker 快速配置深度学习环境](#docker-快速配置深度学习环境)
  - [镜像介绍](#镜像介绍)
    - [安装清单](#安装清单)
  - [快速配置环境（两行命令）](#快速配置环境两行命令)
    - [1. 获取镜像（三选一）](#1-获取镜像三选一)
      - [国内镜像版](#国内镜像版)
      - [🪜科学上网版（直连）](#科学上网版直连)
      - [本地（网盘下载）](#本地网盘下载)
    - [2. 运行容器](#2-运行容器)
  - [安装 Docker Engine](#安装-docker-engine)
    - [卸载旧版本](#卸载旧版本)
    - [使用 `apt` 仓库安装](#使用-apt-仓库安装)
  - [GPU 驱动安装](#gpu-驱动安装)
  - [安装 NVIDIA Container Toolkit](#安装-nvidia-container-toolkit)
  - [拉取并运行深度学习 Docker 镜像](#拉取并运行深度学习-docker-镜像)
- [Docker 基础命令](#docker-基础命令)
  - [镜像管理](#镜像管理)
    - [查看本地镜像](#查看本地镜像)
    - [拉取镜像](#拉取镜像)
    - [删除镜像](#删除镜像)
  - [创建容器](#创建容器)
    - [基础用法](#基础用法)
    - [挂载](#挂载)
    - [在容器中启动 Jupyter Lab](#在容器中启动-jupyter-lab)
  - [停止容器](#停止容器)
    - [在容器终端内停止](#在容器终端内停止)
    - [从主机停止](#从主机停止)
  - [重新连接到已存在的容器](#重新连接到已存在的容器)
  - [命名容器](#命名容器)
    - [使用 `--name` 参数](#使用---name-参数)
    - [使用容器名称的命令示例](#使用容器名称的命令示例)
  - [复制文件](#复制文件)
    - [从主机复制文件到容器](#从主机复制文件到容器)
    - [从容器复制文件到主机](#从容器复制文件到主机)
  - [删除容器](#删除容器)
    - [删除指定容器](#删除指定容器)
    - [删除所有已退出的容器](#删除所有已退出的容器)
  - [查看和调试容器状态](#查看和调试容器状态)
    - [查看容器日志](#查看容器日志)
    - [查看容器详细信息](#查看容器详细信息)
    - [查看容器资源使用情况](#查看容器资源使用情况)
  - [导出与加载镜像](#导出与加载镜像)
    - [使用 `docker commit` 提交容器为镜像](#使用-docker-commit-提交容器为镜像)
    - [导出镜像](#导出镜像)
    - [加载镜像](#加载镜像)
    - [压缩镜像文件](#压缩镜像文件)
- [Docker 常见报错](#docker-常见报错)
    - [报错 1：权限被拒绝（Permission Denied）](#报错-1权限被拒绝permission-denied)
      - [方法 1：使用 `sudo`](#方法-1使用-sudo)
      - [方法 2：将用户添加到 `docker` 用户组](#方法-2将用户添加到-docker-用户组)
    - [报错 2：无法连接到 Docker 仓库（Timeout Exceeded）](#报错-2无法连接到-docker-仓库timeout-exceeded)
      - [方法一：配置镜像](#方法一配置镜像)
      - [方法二：设置 HTTP/HTTPS 代理](#方法二设置-httphttps-代理)
    - [报错 3：磁盘空间不足（No Space Left on Device）](#报错-3磁盘空间不足no-space-left-on-device)
      - [更改 Docker 的数据目录](#更改-docker-的数据目录)
  
---
## 常见报错

- [解决常见报错](#解决常见报错)
  - [报错 1：权限被拒绝（Permission Denied）](#报错-1权限被拒绝permission-denied)
    - [方法 1：使用 sudo](#方法-1使用-sudo)
    - [方法 2：将用户添加到 docker 用户组](#方法-2将用户添加到-docker-用户组)
  - [报错 2：无法连接到 Docker 仓库（Timeout Exceeded）](#报错-2无法连接到-docker-仓库timeout-exceeded)
    - [方法一：配置镜像](#方法一配置镜像)
    - [方法二：设置 HTTP/HTTPS 代理](#方法二设置-httphttps-代理)
  - [报错 3：磁盘空间不足（No Space Left on Device）](#报错-3磁盘空间不足no-space-left-on-device)
    - [更改 Docker 的数据目录](#更改-docker-的数据目录)
- [参考链接](#参考链接)

---

# Docker 快速配置深度学习环境

> 版本说明：
> - **base** 版本基于 `pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel`，默认 `python` 版本为 3.11.10，可以通过 `conda install python==版本号` 直接修改版本。
> - **dl** 版本在 **base** 基础上，额外安装了深度学习框架和常用工具，具体查看[安装清单](#安装)。
> 
> 如果已经配置好了Docker，只需两行命令即可完成深度学习环境的搭建。命令在 Ubuntu 18.04/20.04/22.04 下可以顺利执行，其余系统可通过文内链接跳转安装。

## 镜像介绍

所有版本都预装了 `sudo`、`pip`、`conda`、`wget`、`curl` 和 `vim` 等常用工具，且已经配置好 `pip` 和 `conda` 的国内镜像源。同时，集成了 `zsh` 和一些实用的命令行插件（命令自动补全、语法高亮、以及目录跳转工具 `z`）。此外，已预装 `jupyter notebook` 和 `jupyter lab`，设置了其中的默认终端为 `zsh`，方便进行深度学习开发，并优化了容器内的中文显示，避免出现乱码问题。其中还预配置了 Hugging Face 的国内镜像地址。

**链接**：

- [quickstart](https://hub.docker.com/repository/docker/hoperj/quickstart/general)，位于 Docker Hub，对应于下方的 pull 命令。
- [百度云盘](https://pan.baidu.com/s/1RJDfc5ouTDeBFhOdbIAHNg?pwd=bdka)，直接下载对应的版本，跳过科学版的命令进行配置。

### 安装清单

<details> <summary> <strong>base</strong> </summary>
**基础环境**：

- python 3.11.10
- torch 2.5.1 + cuda 11.8 + cudnn 9

**Apt 安装**：

- `wget`、`curl`：命令行下载工具
- `vim`、`nano`：文本编辑器
- `git`：版本控制工具
- `git-lfs`：Git LFS（大文件存储）
- `zip`、`unzip`：文件压缩和解压工具
- `htop`：系统监控工具
- `tmux`、`screen`：会话管理工具
- `build-essential`：编译工具（如 `gcc`、`g++`）
- `iputils-ping`、`iproute2`、`net-tools`：网络工具（提供 `ping`、`ip`、`ifconfig`、`netstat` 等命令）
- `ssh`：远程连接工具
- `rsync`：文件同步工具
- `tree`：显示文件和目录树
- `lsof`：查看当前系统打开的文件
- `aria2`：多线程下载工具
- `libssl-dev`：OpenSSL 开发库

**pip 安装**：

- `jupyter notebook`、`jupyter lab`：交互式开发环境
- `virtualenv`：Python 虚拟环境管理工具，可以直接用 conda
- `tensorboard`：深度学习训练可视化工具
- `ipywidgets`：Jupyter 小部件库，用以正确显示进度条

**插件**：

- `zsh-autosuggestions`：命令自动补全
- `zsh-syntax-highlighting`：语法高亮
- `z`：快速跳转目录

</details>

<details> <summary> <strong>dl</strong> </summary>

**dl**（Deep Learning）版本在 **base** 基础上，额外安装了深度学习可能用到的基础工具和库：

**Apt 安装**：

- `ffmpeg`：音视频处理工具
- `libgl1-mesa-glx`：图形库依赖（解决一些深度学习框架图形相关问题）

**pip 安装**：

- **数据科学库**：
  - `numpy`、`scipy`：数值计算和科学计算
  - `pandas`：数据分析
  - `matplotlib`、`seaborn`：数据可视化
  - `scikit-learn`：机器学习工具
- **深度学习框架**：
  - `tensorflow`、`tensorflow-addons`：另一种流行的深度学习框架
  - `tf-keras`：Keras 接口的 TensorFlow 实现
- **NLP 相关库**：
  - `transformers`、`datasets`：Hugging Face 提供的 NLP 工具
  - `nltk`、`spacy`：自然语言处理工具

如果需要额外的库，可以通过以下命令手动安装：

```bash
pip install --timeout 120 <替换成库名>
```

这里 `--timeout 120` 设置了 120 秒的超时时间，确保在网络不佳的情况下仍然有足够的时间进行安装。如果不进行设置，在国内的环境下可能会遇到安装包因下载超时而失败的情况。

</details>

---

## 快速配置环境（两行命令）

### 1. 获取镜像（三选一）

假设已经安装并配置好了 Docker，那么只需两行命令即可完成深度学习的环境配置，以 **dl** 镜像为例，拉取：

#### 国内镜像版

```bash
sudo docker pull dockerpull.org/hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
```

#### 🪜科学上网版（直连）

```bash
sudo docker pull hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
```

> [!note]
>
> 如果镜像有更新版本，可通过 `docker pull` 拉取最新镜像。

#### 本地（网盘下载）

> 通过[百度云盘](https://pan.baidu.com/s/1RJDfc5ouTDeBFhOdbIAHNg?pwd=bdka)下载文件（阿里云盘不支持分享大的压缩文件）。
>
> 同名文件内容相同，`.tar.gz` 为压缩版本，下载后通过以下命令解压：
>
> ```bash
> gzip -d dl.tar.gz
> ```

假设 `dl.tar` 被下载到了 `~/Downloads` 中，那么切换至对应目录：

```bash
cd ~/Downloads
```

然后加载镜像：

```bash
sudo docker load -i dl.tar
```

### 2. 运行容器

```bash
sudo docker run --gpus all -it --name ai hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel /bin/zsh
```

如果需要使用 Jupyter，可以使用以下命令：

```bash
sudo docker run --gpus all -it --name ai -p 8888:8888 hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel /bin/zsh
```

> [!tip]
>
> **常用操作提前看**：
>
> - **启动容器**：`docker start <容器名>`
> - **运行容器**：`docker exec -it <容器名> /bin/zsh`
>   - **容器内退出**：`Ctrl + D` 或 `exit`。
> - **停止容器**：`docker stop <容器名>`
> - **删除容器**：`docker rm <容器名>`
> 
---

## 安装 Docker Engine

> 对于图形界面来说，可以跳过下面的命令直接安装 Desktop 版本（其中会提供 Docker Engine），这是最简单的方法。根据系统访问：
>
> - [Linux](https://docs.docker.com/desktop/setup/install/linux/)
> - [Mac](https://docs.docker.com/desktop/setup/install/mac-install/)
> - [Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
>
> 以下是命令行的安装命令，在 Ubuntu 上运行，其余系统参考[官方文档](https://docs.docker.com/engine/install)。

### 卸载旧版本

在安装 Docker Engine 之前，需要卸载所有有冲突的包，运行以下命令：

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```

`apt-get` 可能会报告没有安装这些包，忽略即可。

注意，卸载 Docker 的时候，存储在 /var/lib/docker/ 中的镜像、容器、卷和网络不会被自动删除。如果你想从头开始全新安装，请阅读 [Uninstall Docker Engine 部分](https://docs.docker.com/engine/install/ubuntu/#uninstall-docker-engine)。

### 使用 `apt` 仓库安装

首次安装 Docker Engine 之前，需要设置 Docker 的 `apt` 仓库。

1. **设置 Docker 的 `apt` 仓库**

   ```bash
   # 添加 Docker 的官方 GPG 密钥：
   sudo apt-get update
   sudo apt-get install ca-certificates curl
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc
   
   # 将仓库添加到 Apt 源：
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
     $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update
   ```

   > [!note]
   >
   > 如果你使用的是 Ubuntu 的衍生发行版，例如 Linux Mint，可能需要使用 `UBUNTU_CODENAME` 而不是 `VERSION_CODENAME`。
   >
   > 如果 `sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc` 执行失败，可以尝试以下命令：
   >
   > ```bash
   >sudo wget -qO- https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc
   > ```


2. **安装 Docker 包**

   ```console
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

3. **通过运行 `hello-world` 镜像来验证安装是否成功**

   ```console
   sudo docker run hello-world
   ```

---

## GPU 驱动安装

如果需要使用 GPU 的话，先安装适用于你的系统的 NVIDIA GPU 驱动程序，访问任一链接进行：

- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Official Drivers](https://www.nvidia.com/en-us/drivers/)

---

## 安装 NVIDIA Container Toolkit

> 为了在 Docker 容器中使用 GPU，需要安装 NVIDIA Container Toolkit。
> 注意，我们现在不再需要安装 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker?tab=readme-ov-file)，官方在 2023.10.20 指出其已被 [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) 所取代，过去的配置命令可能已不再适用。

以下命令使用 Apt 完成，Yum 等其他命令访问参考链接：[Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)。

1. **设置仓库和 GPG 密钥**

   设置 NVIDIA 的软件源仓库和 GPG 密钥，确保我们可以从官方源安装 NVIDIA Container Toolkit。

   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
     && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list。
   ```

2. **安装 NVIDIA Container Toolkit**

   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```

3. **配置 Docker**

   使用 `nvidia-ctk` 工具将 NVIDIA 容器运行时配置为 Docker 的默认运行时。

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   ```

4. **重启 Docker**

   ```bash
   sudo systemctl restart docker
   ```

---

## 拉取并运行深度学习 Docker 镜像

> 现在可以拉取深度学习（[dl](https://hub.docker.com/repository/docker/hoperj/quickstart/general)）镜像，命令和之前一致。

1. **拉取镜像**

   ```bash
   sudo docker pull hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
   ```

2. **运行镜像**

   ```bash
   sudo docker run --gpus all -it hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
   ```

3. **检查 GPU**

   在容器内运行：

   ```bash
   nvidia-smi
   ```

   如果正确显示代表成功。不过对于实际使用来说，还需要了解基础命令和报错的解决方法。

---

# Docker 基础命令

## 镜像管理

> 无需每次使用 `sudo` 的设置方法
> 默认情况下，非 root 用户执行 Docker 命令需要在前面加上 `sudo`。如果不想这样，可以将当前用户添加到 `docker` 用户组中：
> ```bash
> sudo groupadd docker  # 如果提示 group 已存在，可忽略
> sudo usermod -aG docker $USER
> newgrp docker
> ```

### 查看本地镜像

```bash
docker images
```
列出本地所有的 Docker 镜像，包括仓库名、标签、镜像 ID、创建时间和大小。

### 拉取镜像

```bash
docker pull <image_name>:<tag>
```

例如：

```bash
docker pull hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
```

> [!note]
>
> `docker pull` 可以从远程仓库更新镜像，如果镜像已存在，则只会拉取更新部分（层），不严谨地类比为 `git pull` 进行理解。

### 删除镜像

```bash
docker rmi <image_id_or_name>
```
在删除镜像前，请确保没有容器正在使用它。

---

## 创建容器

### 基础用法

```bash
docker run --gpus all -it [--rm] <image_name>:<tag>
```

- `--gpus all`：允许容器使用主机的所有 GPU 资源（如有）。
- `-it`：交互式终端。这是两个参数的组合，`-i` 表示“交互式”（interactive），`-t` 表示为容器分配一个伪终端（pseudo-TTY）。
- `--rm`：在容器退出后自动删除容器，避免试验产生无用容器。

以当前使用的深度学习镜像为例：

```bash
docker run --gpus all -it hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
```

> [!tip]
>
> 使用 `docker run --help` 可以查看更多参数的用法。

### 挂载

如果需要在容器内访问主机的文件，可以使用 `-v` 参数。

1. **挂载卷（Volume）**

   ```bash
   docker run --gpus all -it -v my_volume:/container/path hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
   ```

   - `my_volume`：Docker 卷的名称。
   - `/container/path`：容器中的路径。

   挂载卷可让数据在容器删除后仍保留。

2. **挂载主机目录到容器中**

   使用绝对路径挂载主机目录到容器中：

   ```bash
   docker run --gpus all -it -v /host/path:/container/path hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
   ```

   - `/host/path`：主机上的路径。
   - `/container/path`：容器中的路径。

以当前项目为例，假设已经在主机的 `~/Downloads` 文件夹克隆了项目并做了一些修改，那么所需要同步的目录为 `~/Downloads/AI-Guide-and-Demos-zh_CN`，想同步到容器的同名文件夹中，对应命令：

```bash
docker run --gpus all -it -v ~/Downloads/AI-Guide-and-Demos-zh_CN:/workspace/AI-Guide-and-Demos-zh_CN hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
```

容器中的 `/workspace/AI-Guide-and-Demos-zh_CN` 会与主机上的 `~/Downloads/AI-Guide-and-Demos-zh_CN` 目录同步，所有更改都会反映到主机的目录中。

### 在容器中启动 Jupyter Lab

如果需要在容器内启动 Jupyter Lab，并通过主机的浏览器进行访问，可以使用 `-p` 参数映射端口。Jupyter Lab 默认使用 8888 端口，使用以下命令：

```bash
docker run --gpus all -it -p 8888:8888 -v ~/Downloads/AI-Guide-and-Demos-zh_CN:/workspace/AI-Guide-and-Demos-zh_CN hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
```

- `-p 8888:8888` 将容器内的 8888 端口映射到主机的 8888 端口。

然后在容器内运行：

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

现在可以在主机浏览器中访问 `http://localhost:8888`。

如果需要映射多个端口，比如 7860，那么命令对应如下：

```bash
docker run --gpus all -it --name ai -p 8888:8888 -p 7860:7860 ...（后续一致）
```

- `7860` 端口在这里对应于 Gradio。

你可以根据实际情况重新指定端口号。

---

## 停止容器

### 在容器终端内停止

- 使用 `Ctrl+D` 或输入 `exit`：退出并**停止**容器（适用于通过 `docker run` 启动的情况）。
- 使用 `Ctrl+P` 然后 `Ctrl+Q`：仅退出容器的终端（detach），让容器继续在后台运行。

> [!note]
>
> 以上的“停止”行为适用于通过 `docker run` 启动的容器。如果容器是通过 `docker start` 启动的，`Ctrl+D` 或 `exit` 只会退出终端，而不会停止容器。通过 `docker ps` 可以察觉到这一点。

### 从主机停止

如果你想从主机停止正在运行的容器，可以使用：

```bash
docker stop <container_id_or_name>
```

替换 `<container_id_or_name>` 为容器的 ID 或名称。

---

## 重新连接到已存在的容器

在使用一段时间后，你可能会发现每次使用 `docker run` 去“运行”容器时，之前所做的改变都没有“保存”。

**这是因为每次运行 `docker run` 创建了新的容器。**

要找回在容器中的更改，需要重新连接到之前创建的容器，参考以下步骤：

1. **查看正在运行的容器**：

   ```bash
   docker ps
   ```

   如果容器已停止，可**查看所有容器**：

   ```bash
   docker ps -a
   ```

2. **启动已停止的容器**：

   ```bash
   docker start <container_id_or_name>
   ```

3. **连接到正在运行的容器**：

   ```bash
   docker exec -it <container_id_or_name> /bin/bash
   ```

   或者使用

   ```bash
   docker attach <container_id_or_name>
   ```

   区别在于 `exec` 启动一个新进程的终端，`attach` 附着到容器主进程终端（`PID 1`）。

> [!note]
>
> 在之前的命令中，我们使用了 `/bin/zsh`，这是因为该容器中已安装了 zsh。而在大多数容器中，默认的行为通常是 `/bin/bash` 或 `/bin/sh`。

---

## 命名容器

有没有什么方法可以指定名称呢？每次通过 `docker ps -a` 复制 `id` 太不优雅了。

### 使用 `--name` 参数

在创建容器时，可以使用 `--name` 参数为容器指定一个名称。例如：

```bash
docker run --gpus all -it --name ai hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
```

容器被命名为 `ai`，以后可通过该名称管理容器，不需要记住容器的 ID。

运行 `docker ps -a`

### 使用容器名称的命令示例

- **启动容器：**

  ```bash
  docker start ai
  ```

- **停止容器：**

  ```bash
  docker stop ai
  ```

- **重新连接到容器：**

  ```bash
  docker exec -it ai /bin/zsh
  ```

---

## 复制文件

### 从主机复制文件到容器

```bash
docker cp /host/path <container_id_or_name>:/container/path
```

### 从容器复制文件到主机

```bash
docker cp <container_id_or_name>:/container/path /host/path
```

---

## 删除容器

### 删除指定容器

如果想删除一个容器，可以使用 `docker rm` 命令：

```bash
docker rm <container_id_or_name>
```

删除前需先 `stop` 容器。

### 删除所有已退出的容器

```bash
docker container prune
```

这将删除所有已停止的容器（请谨慎使用，因为删除后无法恢复，适用于刚安装 Docker “不小心”创建了一堆容器的情况）。

---

## 查看和调试容器状态

### 查看容器日志

```bash
docker logs <container_id_or_name>
```

可加 `-f` 参数实时跟随日志输出。

### 查看容器详细信息

```bash
docker inspect <container_id_or_name>
```

输出容器的 JSON 配置信息（环境变量、卷挂载、网络信息等）。

### 查看容器资源使用情况

```bash
docker stats
```

显示所有容器的 CPU、内存、网络和存储 I/O 实时数据。

---

## 导出与加载镜像

有时候我们可能需要在没有网络环境、或者在不同机器之间迁移镜像，这时可以通过 `docker save` 和 `docker load` 来完成镜像的导出与导入。

### 使用 `docker commit` 提交容器为镜像

```bash
docker commit <container_id_or_name> <new_image_name>:<tag>
```

- `<container_id_or_name>`：容器的 ID 或名称。
- `<new_image_name>`：为生成的新镜像指定名称。
- `<tag>`（可选）：为镜像指定标签，默认是 `latest`。

假设容器名为 `ai`，我们在其中安装了一些软件并做了环境修改，现在希望将其保存为新镜像 `ai2:latest`：

```bash
docker commit ai ai2:latest
```

执行成功后，使用 `docker images` 查看：

```bash
docker images
```

此时，`ai2:latest` 就是基于容器 `ai` 保存的新镜像，其中包含了所有最近的修改。

### 导出镜像

使用 `docker save` 将指定镜像及其历史层打包成 `.tar` 文件：

```bash
docker save <image_name>:<tag> -o <output_file.tar>
```

例如，将刚才创建的 `ai2:latest` 镜像导出为 `quickstart_ai_image.tar`：

```bash
docker save ai2:latest -o quickstart_ai_image.tar
```

现在我们拥有了一个 `quickstart_ai_image.tar` 文件，可以将其迁移到其他机器上。

### 加载镜像

在另一台机器上，使用 `docker load` 来加载 `.tar` 文件中的镜像：

```bash
docker load -i <input_file.tar>
```

例如：

```bash
docker load -i quickstart_ai_image.tar
```

加载完成后可以使用 `docker images` 查看该镜像已成功导入。

> **区别于 export/import**：
>
> - `docker save` 与 `docker load` 针对镜像操作，并保留镜像的元数据（包括标签和镜像分层信息）。
> - `docker export` 与 `docker import` 针对容器操作，而非镜像本身（此时不需要 commit），将运行后的容器文件系统导出为单一文件系统快照，并不会保留完整的镜像层结构（如果只需要将容器环境打包并在另一端恢复为镜像，可以考虑这一对命令）。

### 压缩镜像文件

为了减少传输的文件大小，可以对导出的 `.tar` 文件进行压缩：

```bash
gzip quickstart_ai_image.tar
```

生成 `quickstart_ai_image.tar.gz` 后，在目标机器上解压：

```bash
gzip -d quickstart_ai_image.tar.gz
docker load -i quickstart_ai_image.tar
```

通过这种方式，可显著减少镜像备份文件的大小（接近 1/2）。

---
# Docker 常见报错

> 介绍在新环境中使用 Docker 时，可能会遇到的报错。

### 报错 1：权限被拒绝（Permission Denied）

当运行命令：

```python
docker ps
```

可能会遇到以下报错：

> permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.45/containers/json": dial unix /var/run/docker.sock: connect: permission denied

**解决方法**：

#### 方法 1：使用 `sudo`

在 Docker 命令前加上 `sudo`：

```bash
sudo docker ps
```

#### 方法 2：将用户添加到 `docker` 用户组

1. **创建 `docker` 用户组**

   ```bash
   sudo groupadd docker
   ```

2. **将当前用户添加到 `docker` 组**

   ```bash
   sudo usermod -aG docker $USER
   ```

3. **重新加载用户组设置**

   ```bash
   newgrp docker
   ```

4. **验证**

   运行 Docker 命令，如果不提示权限错误（permission denied），说明配置成功。

   ```bash
   docker ps	
   ```

### 报错 2：无法连接到 Docker 仓库（Timeout Exceeded）

> Error response from daemon: Get "https://registry-1.docker.io/v2/": net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)

**原因：** 由于国内网络限制，无法直接连接到 Docker Hub。

**解决方法**：

#### 方法一：配置镜像

> 镜像参考：[目前国内可用Docker镜像源汇总（截至2024年11月）](https://www.coderjia.cn/archives/dba3f94c-a021-468a-8ac6-e840f85867ea)

**临时使用**：

直接在原 `<image_name>:<tag>` 前增加网址，比如：

```bash
docker pull dockerpull.org/<image_name>:<tag>
```

快速测试可用性：

```bash
docker pull dockerpull.org/hello-world
```

**永久使用**：

运行以下命令配置文件，如果有一天突然拉（pull）不动了，说明链接挂了需要更新。

```bash
# 创建目录
sudo mkdir -p /etc/docker

# 写入配置文件
sudo tee /etc/docker/daemon.json > /dev/null <<-'EOF'
{
    "registry-mirrors": [
        "https://docker.unsee.tech",
        "https://dockerpull.org",
        "https://docker.1panel.live",
        "https://dockerhub.icu"
    ]
}
EOF

# 重启 Docker 服务
sudo systemctl daemon-reload
sudo systemctl restart docker
```

#### 方法二：设置 HTTP/HTTPS 代理

> 这一项提供给🪜科学上网的同学进行配置。对于本项目来说，**所有文件都会提供网盘链接**和对应的国内镜像命令。

1. **创建并编辑 Docker 的系统服务配置文件**

   ```bash
   sudo mkdir -p /etc/systemd/system/docker.service.d
   sudo vim /etc/systemd/system/docker.service.d/http-proxy.conf
   ```

2. **添加代理配置**

   在 `http-proxy.conf` 文件中添加以下内容（将 `http://localhost:7890/` 替换为你自己的代理地址）：

   ```ini
   [Service]
   Environment="HTTP_PROXY=http://localhost:7890/"
   Environment="HTTPS_PROXY=http://localhost:7890/"
   ```

   使用 `ESC` + `:wq` 回车保存配置。

   > 如果不熟悉 `vim` 的操作，也可以使用直接运行（将 `http://localhost:7890/` 替换为你自己的代理地址）：
   >
   > ```bash
   > sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null <<EOF
   > [Service]
   > Environment="HTTP_PROXY=http://localhost:7890/"
   > Environment="HTTPS_PROXY=http://localhost:7890/"
   > EOF
   > ```

3. **重新加载配置并重启 Docker 服务**

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart docker
   ```

### 报错 3：磁盘空间不足（No Space Left on Device）

> write /var/lib/docker/tmp/...: no space left on device

**原因：** Docker 默认使用 `/var/lib/docker` 作为数据存储目录，如果该分区空间不足，就会出现此错误。

**解决方法：**

#### 更改 Docker 的数据目录

1. **查看当前的磁盘空间**

   检查 `/var/lib/docker` 所在分区的剩余空间：

   ```bash
   sudo df -h /var/lib/docker
   ```

2. **选择具有足够空间的目录**

   创建文件夹：

   ```bash
   mkdir -p ~/Downloads/Docker && cd ~/Downloads/Docker && pwd
   ```

   复制输出。

3. **修改 Docker 的配置文件**

   编辑 `/etc/docker/daemon.json` 文件（如果不存在会自动创建）：

   ```bash
   sudo vim /etc/docker/daemon.json
   ```

   添加或修改以下内容（将 `Path/to/Docker` 替换为你的新数据目录的绝对路径，也就是刚刚复制的输出）：

   ```json
   { 
      "data-root": "Path/to/Docker"
   }
   ```

   `ESC` + `:wq`保存并退出。

4. **重启 Docker 服务并验证**

   ```bash
   sudo systemctl restart docker
   docker info -f '{{ .DockerRootDir}}'
   ```

---
