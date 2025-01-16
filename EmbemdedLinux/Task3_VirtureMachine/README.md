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
      显示当前目录： cd
      返回上一层目录： cd ..
      切换到指定文件夹： cd 文件夹名 
    - mv 移动
      mv 文件名1 文件名2  （文件移动到文件（文件重命名））
      mv 文件名 文件夹名 （文件移动到文件夹下）
      mv 文件夹名1 文件夹名2 （文件夹1存在）
      mv 文件夹名1 文件夹名2 （两文件夹均存在）
    - ls 枚举文件
      ls 路径 列出路径下的文件
      ls 不跟东西 当前目录的文件和目录
    - mkdir 创建目录
      mkdir 名字（创建文件夹）
    - poweroff 关机
    - sudo 以管理员权限运行
    - reboot 重启
    - nmtui 管理网络
    - ifconfig 查看ip地址
    - top/htop 任务管理器
    - nano/vim 文本编辑器
    - apt 包管理器
      apt update：更新软件包
      apt upgrade：将已安装的包升级到最新版本
      apt install xxx：安装 xxx 软件包
      apt remove xxx：删掉名为 xxx 的软件包，保留配置文件
      apt purge xxx：删掉名为 xxx 的软件包，不保留配置文件
      apt search xxx：搜索包含 xxx 的软件包
      apt show xxx：显示 xxx 的详细信息
 - 安装opencv cpp环境并编译 40%
 - 在ubuntu容器中安装rknn toolkit模型转换工具 20%