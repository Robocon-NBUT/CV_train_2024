# Task 3 Virtue Machine
## 学习内容
 - Docker的安装与使用
 - Linux操作系统基础入门
 - opencv cpp版环境安装
## 任务要求
 - 安装Docker 10%
 - 在Docker中创建ubuntu容器 10%
 - 学习以下命令 20%
    - cd 切换工作目录
    - mv 移动
    - ls 枚举文件
    - mkdir 创建目录
    - poweroff 关机
    - sudo 以管理员权限运行
    - reboot 重启
    - nmtui 管理网络
    - ifconfig 查看ip地址
    - top/htop 任务管理器
    - nano/vim 文本编辑器
    - apt 包管理器
 - 安装opencv cpp环境并编译 40%
 - 在ubuntu容器中安装rknn toolkit模型转换工具 20%


# 基本命令

~~~
cd：在容器的终端中输入 cd / 会切换到根目录，cd /home 会切换到 /home 目录。

mv：touch test.txt 创建一个名为 test.txt 的文件。
mv test.txt /home/test.txt 将 test.txt 文件从当前目录移动到 /home 目录。

Is:ls 显示当前目录的文件和目录列表。
ls /home 显示 /home 目录下的文件和目录列表。

mkdir:
mkdir new_directory  mkdir 命令用于创建新目录，这里创建了一个名为 new_directory 的目录。

poweroff:exit 命令用于退出容器的 bash 终端。在容器内使用 poweroff 会因为容器的运行环境限制而无法正常执行关机操作。

sudo:apt update 更新包列表。
apt install -y sudo 安装 sudo 工具，-y 选项自动回答 yes 确认安装。
sudo apt update 使用 sudo 权限来更新包列表。

reboot:exit 退出容器。
docker start <container_id> 启动之前创建的容器，<container_id> 是容器的唯一标识符，可以使用 docker ps -a 查看。
docker exec -it <container_id> /bin/bash 重新进入容器的 bash 终端。

nmtui:apt install -y network-manager 安装网络管理器。
nmtui 启动网络管理的文本界面。

ifconfig:apt update 更新包列表。
apt install -y net-tools 安装 net-tools 包，包含 ifconfig 工具。
ifconfig 查看网络接口信息。

top/htop:apt update 更新包列表。
apt install -y htop 安装 htop 任务管理器。
htop 启动 htop 任务管理器，显示系统进程信息。

nano/vim:apt update 更新包列表。
apt install -y nano 安装 nano 文本编辑器。
nano test.txt 打开 test.txt 文件进行编辑，如果文件不存在会创建并打开。


~~~