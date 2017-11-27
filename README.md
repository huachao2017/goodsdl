#启动方式
python manage.py runserver 0.0.0.0:8000

#系统安装
yum -y install lrzsz
yum -y install gcc-c++
下载四个加粗安装文件到本地/root/download
用root用户进入下列安装：

##0、检测显卡
rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org<br>
(centos6)rpm -Uvh http://hkg.mirror.rackspace.com/elrepo/elrepo/el6/x86_64/RPMS/elrepo-release-6-8.el6.elrepo.noarch.rpm<br>
(centos7)rpm -Uvh http://www.elrepo.org/elrepo-release-7.0-2.el7.elrepo.noarch.rpm<br>
yum -y install nvidia-detect<br>
nvidia-detect -v<br>

##1、安装驱动
禁用nouveau<br>
lsmod | grep nouveau<br>
echo -e "blacklist nouveau\noptions nouveau modeset=0" > /etc/modprobe.d/blacklist.conf<br>
lsmod | grep nvidiafb<br>
echo -e "blacklist nvidiafb\noptions nvidiafb modeset=0" > /etc/modprobe.d/blacklist.conf<br>
mv /boot/initramfs-$(uname -r).img /boot/initramfs-$(uname -r).img.bak<br>
dracut /boot/initramfs-$(uname -r).img $(uname -r)<br>
reboot<br>

yum -y install kernel-devel kernel-headers<br>
sh **NVIDIA-Linux-x86_64-384.90.run** --kernel-source-path=/usr/src/kernels/TAB<br>
nvidia-smi<br>

##2、安装cuda
sh **cuda_8.0.44_linux.run** --kernel-source-path=/usr/src/kernels/TAB<br>
修改.bash_profile<br>
PATH=$PATH:$HOME/bin:/usr/local/cuda/bin:/root/anaconda3/bin<br>
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/<br>
CUDA_HOME=/usr/local/cuda<br>
export PATH<br>
export LD_LIBRARY_PATH<br>
export CUDA_HOME<br>

nvcc -V<br>

##3、安装cudnn
tar -xvf **cudnn-8.0-linux-x64-v6.0**.solitairetheme8 -C /usr/local<br>

##4、安装Anaconda3
yum -y install bzip2<br>
sh **Anaconda3-5.0.0.1-Linux-x86_64.sh**<br>

##5、安装TensorFlow
/root/anaconda3/bin/pip install tensorflow_gpu<br>
/root/anaconda3/bin/pip install  https://pypi.python.org/packages/08/47/bc3ccd2ecae8f9f18a93c683a66339714090a36e1b69307787fb28e72e2b/tensorflow_gpu-1.4.0-cp36-cp36m-manylinux1_x86_64.whl#md5=110fbd34ae26089a3d966ecd4bf27455<br>
用import tensorflow as tf测试<br>
glibc、glibcxx、libstdc的问题见http://blog.csdn.net/u011832895/article/details/53731199<br>

##6、安装tensorflow models
###6.1、安装git
yum -y install git
###6.2、下载models
在/home/source下：
git clone https://github.com/tensorflow/models.git
###6.3、增加PYTHONPATH
修改.bash_profile
export PYTHONPATH=$PYTHONPATH:/home/source/models/research:/home/source/models/research/slim
###6.4、生成protobuf
wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
unzip protoc-3.3.0-linux-x86_64.zip
cd /home/source/models/research
~/bin/protoc object_detection/protos/*.proto --python_out=. 

##7、安装web
###7.1、下载代码
在/home/src下：
git clone https://github.com/huachao2017/goodsdl.git
###7.2、进入goodsdl，安装依赖包
yum -y install libX11-devel
yum -y install libXext-devel
cd dl
/root/anaconda3/bin/pip install -r requirements.txt
上传frozen_inference_graph.pb（或者训练生成）到model目录下的各个类目录里
###7.3、开启防火墙
(centos7)
firewall-cmd --permanent --add-port=80/tcp
service firewalld restart
(centos6)
iptables -I INPUT -p tcp --dport 80 -j ACCEPT
/etc/rc.d/init.d/iptables save
###7.4、初始化数据库，启动web服务
python3 /home/src/goodsdl/manage.py migrate
###7.5、创建服务
修改rc.local
vi /etc/rc.d/rc.local
PATH=$PATH:$HOME/bin:/usr/local/cuda/bin:/root/anaconda3/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
CUDA_HOME=/usr/local/cuda
export PYTHONPATH=$PYTHONPATH:/home/source/models/research:/home/source/models/research/slim:/home/source/models/research/object_detection
export PATH
export LD_LIBRARY_PATH
export CUDA_HOME
python3 /home/src/goodsdl/manage.py runserver 0.0.0.0:80
修改rc.local的权限
chmod +x /etc/rc.d/rc.local
###7.6、查看日志
tail -f /home/src/goodsdl/logs/debug.log
###7.7、安装中文字体
http://blog.csdn.net/wlwlwlwl015/article/details/51482065
vi /home/source/models/research/object_detection/utils/visualization_utils.py
164行修改为：
font = ImageFont.truetype('simsun.ttc', 24)