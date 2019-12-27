#  安装环境要求
- Linux Ubuntu 16.04 +
- CUDA 9.0
- NCCL 2+
- GCC 4.9+
  
#  需要安装的关联的python包
- Python 3.5
- PyTorch 1.1.0
  
    ```
    pip install torch torchvision
    ```
- mmcv
 
    ```
    pip install mmcv
    ```
- opencv-python
  
    ```
    pip install opencv-python
    ```
- PyQt5
   
    ```
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PyQt5
    ```
- PyQt5 tools
  
    ```
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PyQt5-tools
    ```
- cython

- pyrealsense2
  
    ```
    pip install pyrealsense2 -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

## 安装步骤
a. 创建Conda环境并激活，然后用conda安装cython以及安装前面提到的关联的python包，但是先保证安装环境已经正确安装配置

```shell
conda create -n wood python=3.5 -y
source activate wood

conda install cython
```

b. 在当前的我们的开发包Wood文件夹中 编译cuda扩展程序.

```shell
./compile.sh
```

c. 然后安装 woodev (其他依赖的包将自动进行安装).

```shell
   python setup.py develop
```
d. 当前仓库的模型是基于以下仓库进行训练，感兴趣的读者也可以自己训练得到对应的模型，该仓库的链接如下，仓库可能有更新，请参照进行修改
```
    https://github.com/open-mmlab/mmdetection
```
## 安装摄像头环境
- 摄像头的SDK在librealsense文件夹下，需要使用摄像头程序，需要将该文件夹下的.so文件加入到变量搜索空间中，具体指令为:
```
    export LD_LIBRARY_PATH=Your_work_path/librealsense:$LD_LIBRARY_PATH  其中Your_work_path为你的绝对工作路径
    echo $LD_LIBRARY_PATH 可查看你的电脑的LD_LIBRARY_PATH路径下已经添加的搜索路径
```
- realsense 主要基于intel 的realsense 摄像头开发，该开发套件请参照以下官方套件的Linux版本
    ```
   https://github.com/IntelRealSense/librealsense
    ```

## 安装sticher配置环境
### Step1 
- .so文件配置:
    ```
    export LD_LIBRARY_PATH=Your_work_path/libsticher:$LD_LIBRARY_PATH
    ```
- sticher为拼接库，请参照我的另外一个仓库如下
    ```
    https://github.com/Hedlen/NISwGSP_Ubuntu
    ```
### Step2 
对于缺失本地库的安装（本地可能已经安装，没有安装的情况可以按照以下指令进行安装)
- libjpeg.so.9 安装
    ```
    在http://jpegclub.org/reference/reference-sources/ 下载 jpegsrc.v9b.tar.gz
    tar xvf jpegsrc.v9b.tar.gz 解压
	cd jpeg-9b
	./configure
	make check
	sudo make install
	sudo ldconfig
    ```
- libjasper.so.1 安装
    ```
	sudo add-apt-repository 'deb http://security.ubuntu.com/ubuntu xenial-security main'
	sudo apt update
	sudo apt install libjasper-dev
    ```
- libtbb.so.2 安装
    ```
	sudo apt-get install libtbb2
    ```
# 运行demo
在保证前面的步骤已经完成，并已经得到正确的安装的情况下，可以执行以下的指令，运行demo！！！
```
python run_demo.py
```

# 检测单图和视频

```
具体功能类似之前的版本
```
# 采集拍摄
具体步骤：
- 插入摄像头（！！！！）
- 点击界面拍摄选项
- 点击播放，可以查看当前拍摄画面
- 点击暂停，可以暂停当前视频流，点击继续可以继续播放
- 输入采集排数，会将不同排的数据采集结果进行分类，方便进行选择。输入排数之前，请先拍完当前木头堆后，点击重置，然后再输入排数，然后点击播放，并点击保存即可保存对应排数的木头数据。
- 点击保存，可以保存当前画面看到的视频帧，点击保存时，确保显示保存成功后，然后再继续点击保存保存下一帧。请严格按照采集提示中提到的拍摄顺序来进行拍摄。
- 在拍摄完当前木头堆后，再点击重置，可以接着拍另外一堆木头堆，请保证采集完当前木头堆后再点击重置，又不会使采集数据出现问题。
- 点击设置，可以调节亮度，饱和度，曝光时间等参数，在比较黑的场景，可以通过设置来提亮，方便图像中目标可见。
- 
# 多视图检测
具体步骤：
- 点击界面右上角多视图选项
- 点击输入，选择包含数据的文件夹，选中该文件夹，然后点击确认即可
- 点击检测，即可输出结果，检测信息将包含木头个数和每根木头的长短轴的长度

# 界面展示

![木头检测系统](ui/ui.jpeg)



