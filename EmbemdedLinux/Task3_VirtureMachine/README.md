#学习成果

     cd 切换工作目录
-cd /home             #进入 /home 目录
-cd ..                 # 返回上一级目录
-cd ~                  # 返回当前用户的主目录
-cd /path/to/directory # 进入指定路径的目录

     mv 移动
-mv file.txt /home/user/       # 将 file.txt 移动到 /home/user/
-mv oldname.txt newname.txt    # 将 oldname.txt 重命名为 newname.txt

     ls 枚举文件
-ls                    # 列出当前目录的文件
-ls -l                 # 显示详细信息
-ls /path/to/directory # 列出指定目录内容

     mkdir 创建目录
-mkdir new_folder              # 创建名为 new_folder 的目录
-mkdir -p parent/child/grandchild # 创建多层目录

     poweroff 关机
-sudo poweroff                # 需要管理员权限

     sudo 以管理员权限运行
-sudo apt update              # 以管理员权限更新软件包
-sudo nano /etc/hosts         # 以管理员权限编辑 /etc/hosts 文件

     reboot 重启
 -sudo reboot                  # 需要管理员权限

     nmtui 管理网络
-sudo nmtui                   # 启动网络管理工具

     ifconfig 查看ip地址
-ifconfig                     # 显示所有网络接口信息
-ifconfig eth0                # 查看特定接口（如 eth0）的信息

     top/htop 任务管理器
-top                          # 启动任务管理器
-htop                         # 更友好的任务管理界面（需安装）

     nano/vim 文本编辑器
-nano file.txt                # 打开 file.txt 进行编辑
-vim file.txt                 # 打开 file.txt 进行编辑

     apt 包管理器
-sudo apt update              # 更新包列表
-sudo apt upgrade             # 升级系统中的所有软件包
-sudo apt install package     # 安装指定的软件包
-sudo apt remove package      # 删除指定的软件包
