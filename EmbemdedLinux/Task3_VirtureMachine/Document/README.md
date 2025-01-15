# Task 3 Virtue Machine

## 学习内容

- cd 切换工作目录
  ```
  cd ..返回上一级
  cd - 切换到上一个所在目录
  cd 不跟东西 主目录
  ```
- mv 移动
  ```
  mv xxx1 xxx2 重命名xxx1为xxx2
  mv xxx1 路径 移动xxx1到路径
  ```
- ls 枚举文件
  ```
  ls 路径 列出路径下的文件
  ls 不跟东西 当前目录的文件和目录
  ```
- mkdir 创建目录
  ```
  mkdir xxx 在当前目录下创建xxx
  ```
- poweroff 关机
- sudo 以管理员权限运行
  ```
  sudo 命令 在docker下不需要sudo
  ```
- reboot 重启
- nmtui 管理网络（带图形界面）
- ifconfig 查看ip地址
- top/htop 任务管理器
  ```
  top简略一点
  htop详细一点
  ```
- nano/vim 文本编辑器
  ```
  nano xxx 打开/创建文件
  ctrl+o保存，+x退出
  
  vim xxx：打开/创建文件
   i 进入插入模式，输入文本  Esc 键退出插入模式
   输入 :wq 保存并退出，输入 :q! 不保存退出
  ```
- apt 包管理器
  ```
  apt update：更新软件包
  apt upgrade：将已安装的包升级到最新版本
  apt install xxx：安装 xxx 软件包
  apt remove xxx：删掉名为 xxx 的软件包，保留配置文件
  apt purge xxx：删掉名为 xxx 的软件包，不保留配置文件
  apt search xxx：搜索包含 xxx 的软件包
  apt show xxx：显示 xxx 的详细信息
  ```
## 安装docker

需要开启wsl，使用docker run -it --name xxx(取名) ubuntu直接安装     
先使用/bin/bash进入主目录