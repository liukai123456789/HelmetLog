# 昇腾平台YOLOv5-安全帽识别



## 一.样例使用日志

1.Image decode failed![image-20210628101536174](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210628101536174.jpg)


## 二.模型转换（2021.6.28-2021.7.3）

#### 0.模型下载

 0.1 github代码开源地址：[PeterH0323/Smart_Construction: Head Person Helmet Detection on Construction Sites，基于目标检测工地安全帽和禁入危险区域识别系统，🚀😆附 YOLOv5 训练自己的数据集超详细教程🚀😆2021.3新增可视化界面❗❗ (github.com)](https://github.com/PeterH0323/Smart_Construction)

 0.2 模型下载：helmet_head_person_l.pt /helmet_head_person_m.pt/helmet_head_person_s.pt

YOLOv5s `epoch = 50`

|        | P     | R     | mAP0.5 |
| ------ | ----- | ----- | ------ |
| 总体   | 0.884 | 0.899 | 0.888  |
| 人体   | 0.846 | 0.893 | 0.877  |
| 头     | 0.889 | 0.883 | 0.871  |
| 安全帽 | 0.917 | 0.921 | 0.917  |

YOLOv5m `epoch = 100`

| 分类   | P     | R     | mAP0.5 |
| ------ | ----- | ----- | ------ |
| 总体   | 0.886 | 0.915 | 0.901  |
| 人体   | 0.844 | 0.906 | 0.887  |
| 头     | 0.9   | 0.911 | 0.9    |
| 安全帽 | 0.913 | 0.929 | 0.916  |

YOLOv5l  `epoch = 100`

| 分类   | P     | R     | mAP0.5 |
| ------ | ----- | ----- | ------ |
| 总体   | 0.892 | 0.919 | 0.906  |
| 人体   | 0.856 | 0.914 | 0.897  |
| 头     | 0.893 | 0.913 | 0.901  |
| 安全帽 | 0.927 | 0.929 | 0.919  |

 0.3 数据集下载：[Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)

#### 1.act安装目录：

/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/atc/bin/atc

注意：

*使用export方式设置环境变量后，环境变量只在当前窗口有效。如果用户之前在.bashrc文件中设置过ATC软件包安装路径的环境   变量，则在执行上述命令之前，需要先手动删除原来设置的ATC安装路径环境变量。*
*若开发环境架构为Arm（aarch64），模型转换耗时较长，则可以参考[开发环境架构为Arm（aarch64），模型转换耗时较长解决](https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0129.html)。*

#### 2.环境变量设置

1.指定python环境版本 py3.7.5 ： export PATH=/usr/local/python3.7.5/bin:$PATH

2.非root用户安装toolkit包： . ${HOME}/Ascend/ascend-toolkit/set_env.sh

3.非必选环境变量：日志落盘与打屏 算子并行编译能力。打印模型转换的图信息

#### 3.   .pt转 .ONNX

*Open Neural Network Exchange（ONNX，开放神经网络交换）格式，是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。它使得不同的人工智能框架（如Pytorch, MXNet）可以采用相同格式存储模型数据并交互。 ONNX的规范及代码主要由微软，亚马逊 ，Facebook 和 IBM 等公司共同开发，以开放源代码的方式托管在Github上。目前官方支持加载ONNX模型并进行推理的深度学习框架有： Caffe2, PyTorch, MXNet，ML.NET，TensorRT 和 Microsoft CNTK*

#####   3.1AMTC 

通过昇腾模型压缩工具（AMCT）将.pt文件量化并转换为.onnx文件

#####   3.2 直接转换

1.加载pt文件

```python
import argparse
import os
import sys
# 使用绝对路径import，对应教程的 export PYTHONPATH="$PWD"
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from models.common import *
from utils import google_utils
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./helmet_head_person_l.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt
# Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size(1,3,320,192) iDetection
    # Load PyTorch model
    google_utils.attempt_download(opt.weights)
    model = torch.load(opt.weights, map_location=torch.device('cpu'))['model'].float()
    model.eval()
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run
```

2.onnx导出

```python
import onnx
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = opt.weights.replace('.pt', '.onnx')  # filename
    model.fuse()  # only for ONNX
    torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
                      output_names=['classes', 'boxes'] if y is None else ['output'])
```

3.验证

```python
onnx_model = onnx.load(f)  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model
print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
print('ONNX export success, saved as %s' % f)

```

4.onnx文件：helmet_head_person_l.onnx：**/home/sd_xiong2/ExampleProject/HelmetIdentification/Models/helmet_head_person_l.onnx**

helmet_head_person_m.onnx：**/home/sd_xiong2/ExampleProject/HelmetIdentification/Models/helmet_head_person_m.onnx**

helmet_head_person_s.onnx：**/home/sd_xiong2/ExampleProject/HelmetIdentification/Models/helmet_head_person_s_1.7.0_op11_dbs.onnx**

先尝试使用最小的helmet_head_person_s.onnx进行开发。

* | 输入数据 | 大小        | 数据类型 | 数据排布格式 |
  | -------- | ----------- | -------- | ------------ |
  | images   | 1x3x640x640 | RGB      | NCHW         |

  | 输出数据      | size        | datatype | 数据排布格式 |
  | ------------- | ----------- | -------- | ------------ |
  | output        | 1x3x80x80x8 | FLOAT32  | NCHW         |
  | 462（未采用） | 1x3x40x40x8 | FLOAT32  | NCHW         |
  | 482（未采用） | 1x3x20x20x8 | FLOAT32  | NCHW         |

  输出尺寸为1x3x80x80x8，1表示为批大小，3表示每个cell预测三个bbox，80x80代表特征图尺寸，8表示为4个坐标+1个置信度+3分类概率



#### 4.  .ONNX 转 .OM

1.以ATC软件包运行用户登录开发环境，并将模型转换过程中使用到的模型文件 （*.onnx）等上传到开发环境任意路径。

2.执行如下命令：atc --model=$HOME/···/xxx.onnx --framework=5 --output=$HOME/···/onnx_xxx -- soc_version=${soc_version}

atc --model=${Home}/Models/helmet_head_person_s_1.7.0_op11_dbs.onnx --framework=5 --output=${Home}/Models/helmet_head_person_s_1.7.0_op11_dbs --input_format=NCHW --log=info --soc_version=Ascend310 --input_shape="images:1,3,640,640"

* 模型中的所有层算子除const算子外，输入和输出需要满足dim!=0

* 支持原始框架类型为Caffe、TensorFlow、MindSpore、ONNX的模型转换，当原始框架类型为Caffe、MindSpore、ONNX时，输入数据类型为FP32、FP16（通过 设置入参--input_fp16_nodes实现，MindSpore框架不支持该参数）、UINT8（通 过配置数据预处理实现）；

* | 参数名           | 参数描述                                                     |
  | ---------------- | :----------------------------------------------------------- |
  | --  framework    | 原始框架类型。当取值为5时，即为ONNX网络模型，仅支持ai.onnx算子域中opset v11版本的算 子。用户也可以将其他opset版本的算子（比如opset v9），通过PyTorch转换成 opset v11版本的onnx算子 |
  | --model          | 原始模型文件路径与文件名                                     |
  | --output         | 如果是开源框架的网络模型，存放转换后的离线模型的路径以及文件名。 |
  | --soc_version    | 模型转换时指定芯片版本。昇腾AI处理器的版本，可从ATC工具安装路径的“/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/atc/data/platform_config”目录下 查看。 ".ini"文件的文件名即为对应的${soc_version} |
  | --insert_op_conf | 插入算子的配置文件路径与文件名， 例如aipp预处理算子。        |
  | --input_shape    | 模型输入数据的 shape。                                       |
  | --out_nodes      | 指定输出节点,如果不指定输出节点（算子名称），则模型的输出默认为最后一层的算子信息，如果 指定，则以指定的为准 |

3.提示 **ATC run success** 说明转换成功

#### 5.错误日志

##### 1.E19010 

![image-20210630083240461](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210630083240461.jpg)



* 错误信息：Check op[%s]'s type[%s] failed, the type is unsupported. 

* 处理:onnx opset_version设置为11 onnx=1.7.0

##### 2.E10016

![image-20210630104846984](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210630104846984.jpg)

* 错误信息： Input parameter[--%s]'s opname[%s] is not exist in model 
* 处理： env.sh中--input_shape="inputname:1,3,640,640" 名称要与入参算子名称相同，修改为images

##### 3.E19999

![image-20210630111512259](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210630111512259.jpg)

  命令行：atc -log=error

  ![image-20210630141351906](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210630141351906.jpg)

处理：onnx模型中含有动态resize算子进行上采样，所以通过计算维度变化，改为静态算子。

  运行脚本：

```python
python replace_resize.py helmet_head_person_s_1.7.0_op11.onnx
```

##### 4.W11001

![image-20210701110903188](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210701110903188.jpg)

警告信息:High-priority service of op[%s] is invalid, low-priority service is used. It can work normally but may affect performance，在模型编译中无法选择高优先级引擎，使用低优先级引擎替代，不影响模型生成，可能对性能有影响。



● 处理建议 一般是算子的输入不满足要求，导致aicore引擎到aicpu引擎的切换，如需要可联系华 为工程师协助解决



## 三. 第二周开发（2021.7.5-2021.7.11）



### 1.样例：AllObjectsStructuring 插件流程

1.mxpi_rtspsrc0（mxpi_rtspsrc）：拉流，接收外部调用接口的输入视频路径，对视频进行拉流，并将拉取的 裸流存储到缓冲区(buffer)中，并发送到下游插件，目前只支持H264拉流。输出：buffer（数据类型“Mxpibuffer”）、metadata（数据类型 “MxpiFrame”）

2.mxpi_videodecoder0（mxpi_videodecoder）：视频解码，将视频转换为YUV420SP_NV12格式的图片

3.mxpi_skipframe0（mxpi_skipframe）（**自定义插件**）：图片跳帧 相隔两帧跳一次。

4.mxpi_parallel2serial0（mxpi_parallel2serial）：串行化插件，多个端口输入数据通过一个端口按顺序输出

5.mxpi_imageresize0（mxpi_imageresize）：图片缩放，设置为适合模型输入的尺寸

6.queue0（queue）：此插件输出时为后续处理过程另创建一个线程，用于将输入数据与输出数据解耦，并创建缓存队列，存储尚未输出到下游插件的数据。

7.mxpi_modelinfer0（mxpi_modelinfer）：推理插件，用于目标分类或检测，

8.mxpi_objectdistributor0（mxpi_distributor）：向不同端口发送指定类别或通道的数据。在配置文件中填写类别索引或通道索引来选择发送需要输出的结果

9.1 queue1（queue）：创建缓存队列

9.2 queue33（queue）：创建缓存队列

10.1.mxpi_imagecrop1（mxpi_imagecrop）：抠图，*支持根据目标检测的（x,y）坐标和（width,height）宽高进行图像裁剪（抠图）。支持指定上下左右四个方向的扩边比例，扩大目标框的区域进 行图像裁剪。支持指定缩放后的宽高将裁剪（抠图）的图像缩放到指定宽 高*。

11.1 mxpi_channeldistributor_pfm（mxpi_distributor）：向不同端口发送指定类别或通道的数据

12.1.1 queue_landmark0（queue）：创建缓存队列

12.1.2 queue_landmark1（queue）：创建缓存队列

12.1.3 queue_landmark2（queue）：创建缓存队列

13.1.1 mxpi_facelandmark0（mxpi_modelinfer）：推理插件

13.1.2 mxpi_facelandmark1（mxpi_modelinfer）：推理插件

13.1.3 mxpi_facelandmark2（mxpi_modelinfer）：推理插件

14 mxpi_facelandmark（mxpi_parallel2serial）：串行化插件，多个端口输入数据通过一个端口按顺序输出

15 queue2（queue）：创建缓存队列

**···**

![AllObjectsStructuring 插件流程](https://github.com/liukai123456789/HelmetLog/blob/main/images/AllObjectsStructuring%20%E6%8F%92%E4%BB%B6%E6%B5%81%E7%A8%8B.jpg)



### 2.安全帽识别插件流程

<img src="https://github.com/liukai123456789/HelmetLog/blob/main/images/%E5%AE%89%E5%85%A8%E5%B8%BD%E8%AF%86%E5%88%AB%E6%8F%92%E4%BB%B6%E6%B5%81%E7%A8%8Bv3.jpg" alt="安全帽识别插件流程v3" style="zoom:75%;" />



1.拉流（mxpi_rtspsrc）：接收外部调用接口的输入视频路径，对视频进行拉流，并将拉取的 裸流存储到缓冲区(buffer)中，并发送到下游插件，目前只支持H264拉流。输出：buffer（数据类型“Mxpibuffer”）、metadata（数据类型 “MxpiFrame”）

2.视频解码（mxpi_videodecoder）：将视频转换为YUV420SP_NV12格式的图片

3.图片跳帧（mxpi_skipframe）：**自定义插件**，图片跳帧 相隔两帧跳一次，实现一路每秒8frame

4.数据串行化（mxpi_parallel2serial）：串行化插件，将接受到的两路数据串行化，通过输出端口输出。

5.图片缩放（mxpi_imageresize）：将图片设置为适合模型输入的尺寸

6.数据缓存（queue）：此插件输出时为后续处理过程另创建一个线程，用于将输入数据与输出数据解耦，并创建缓存队列，存储尚未输出到下游插件的数据。

7.安全帽识别推理（mxpi_modelinfer）：推理插件，用于目标分类或检测。本设计采用YOLOv5模型，输入尺寸为  1x3x640x640的JPEG图片，输出大小为1x3x80x80x8，1表示为批大小，3表示每个cell预测三个bbox，80x80代表特征图尺寸，8表示为4个坐标+1个置信度+3分类概率。输入输出数据排布格式都为NCHW

8.图像后处理：**动态链接文件**，用于模型输出后的数据处理 包括NMS,跟踪去重等。

9.数据分流（mxpi_distributor）：将经过推理模型识别后的数据分为原两路数据

10.序列化（mxpi_dataserialize）：将业务流结果序列化组装成json字符串输出。

11.appsink ：业务流的输出插件，最终会在appsink0获取推理结果。

12. 告警与画框：调用第三方库cv2，通过解析序列化结果，对未佩戴安全帽的对象做出告警，对已告警的对象不再重复告警，把没戴安全帽情况画框保存为图片。



### 3.接口调用方式

**StreamManagerApi => InitManager => CreateMultipleStreams => GetProtobuf**

StreamManagerApi：用于业务流管理的接口，该类主要用于对流程的基本管理，包括加载流程配置、创建流程、向流程上发送数据、获得执行结果。

InitManager ：初始化当前StreamManagerApi

CreateMultipleStreams ：此接口根据指定的配置文件创建多个Stream，用于加载pipline文件。

GetProtobuf：本套接口没有使用appsrc，而是使用视频取流元件mxpi_rtspsrc，因此不需要通过SendProtobuf发送数据，输出结果仅用GetProtobuf获取。元件处理完数据后，以元件名为key将处理结果保存至元数据中，最后通过GetProtobuf接口从元数据中取出想要获取的元件结果，输入一组key，便能获取key对应的protobuf数据。

![接口调用流程](https://github.com/liukai123456789/HelmetLog/blob/main/images/%E6%8E%A5%E5%8F%A3%E8%B0%83%E7%94%A8%E6%B5%81%E7%A8%8B.jpg)



### 4.onnx模型修改与简化

#### 1.resize算子修改

已知resize算子属于动态算子，atc模型转换不支持，会导致系统报错E19999，需要将算子的size属性用scales属性代替。通过 **https://convertmodel.com/** 可以将onnx文件的详细信息显示，可以得知scales的值为[1,1,2,2]

![image-20210707203326392](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210707203326392.jpg)

#### 2.onnx模型简化

推断整个计算图，然后用其不变的输出替换冗余运算符简化模型。

1. 安装方式：pip3 install onnx-simplifier

2. 运行命令：python -m onnxsim --skip-optimization helmet_head_person_s_1.7.0_op11_dbs.onnx helmet_head_person_s_1.7.0_op11_dbs_sim.onnx

3. error：ImportError: Microsoft Visual C++ Redistributable for Visual Studio 2019 not installed on the machine

​       处理：install  Microsoft Visual C++ Redistributable for Visual Studio 2019



#### 3.slice算子修改

旧版使用slice算子之后会报警告W10010，在模型编译中无法选择高优先级引擎，使用低优先级引擎替代，可能会影响模型性能。修改后不再报警。

运行命令：python modify_yolov5s_slice.py  helmet_head_person_s_1.7.0_op11_dbs_sim.onnx 

生成: helmet_head_person_s_1.7.0_op11_dbs_sim_t.onnx

修改前：

![image-20210708110432746](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210708110432746.jpg)

修改后：

![image-20210708110338510](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210708110338510.jpg)



### 5.AIPP config文件

```python
aipp_op{  # 算子标识
    aipp_mode:static  # 静态aipp
    input_format : YUV420SP_U8   # 输入format：YUV420SP_NV12

    src_image_size_w : 640  # images size
    src_image_size_h : 640

    crop: false  # 不使用crop padding
    load_start_pos_h : 0
    load_start_pos_w : 0
    crop_size_w : 640
    crop_size_h: 640

    csc_switch : true   # 开启色域转换
    rbuv_swap_switch : false # 是否通道交换

    # 色域转换  YUV420SP_U8转RGB
    matrix_r0c0: 256
    matrix_r0c1: 0
    matrix_r0c2: 359
    matrix_r1c0: 256
    matrix_r1c1: -88
    matrix_r1c2: -183
    matrix_r2c0: 256
    matrix_r2c1: 454
    matrix_r2c2: 0
    input_bias_0: 0
    input_bias_1: 128
    input_bias_2: 128

    # 均值归一化
    min_chn_0 : 0
    min_chn_1 : 0
    min_chn_2 : 0
    var_reci_chn_0: 0.003921568627451
    var_reci_chn_1: 0.003921568627451
    var_reci_chn_2: 0.003921568627451}
```



### 6.benchmark推理

#### 1.下载软件包

Ascend-cann-benchmark_20.3.0-Linux-aarch64.zip  解压至服务器/home/sd_xiong2/ExampleProject/HelmetIdentification/benchmark_tools

#### 2.准备文件

推理模型 **helmet_head_person_s_1.7.0_op11_dbs_sim_t.om**

​                      coco2014数据集   

​                      模型预处理文件 **yolo_tf_preprocess.py**

​                      运行脚本 **benchmark.aarch64**

​                      数据集解析文件 **parse_COCO.py**

​                      推理数据集生成文件 **get_yolo_info.py**

​                      mAP精度统计脚本 **map_calculate.py**

#### 3.纯推理场景

运行命令：cd  /home/sd_xiong2/ExampleProject/HelmetIdentification/benchmark_tools

​                   chmod +x benchmark.aarch64

​                   source env.sh

​                  ./benchmark.aarch64 -om_path=./helmet_head_person_s_1.7.0_op11_dbs_sim_t.om -batch_size=1 -round=30

ERROR：账号没有root权限，显示缺少so文件，切换至root账号。

测试成功：![image-20210709193732310](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210709193732310.jpg)

参数说明：

-batch_size：执行一次模型推理所处理的数据量。

-om_path：经过ATC转换后的模型OM文件所 在的路径

-round：执行模型推理的次数，取值范围为 1~1024



ave_throughputRate：模型的平均吞吐率。单位为samples/s

ave_latency： 模型执行的平均时间。单位为ms

**注**：若不修改slice算子，则无法使用benchmark推理，报错为：[ERROR] RUNTIME(24810)aicpu kernel execute failed, device_id=0, stream_id=628, task_id=13, fault so_name=, fault kernel_name=, extend_info=(info_type:4, info_len:7, msg_info:Slice_9).

#### 4.推理场景

##### 1.数据准备（废弃）

*1.1.解析原始数据集标签文件 和原始图片，生成类别文件、图片信息文件和真实标签*

*运行命令：python3.7 parse_COCO.py --json_file instances_val2017.json  -- img_path val2017 --classes coco2017.names --info coco2017.info --gtp ground-truth/*

*运行成功，在相同目录下生成生成coco2017.names，coco2017.info ，./ground-truth/xxx.txt*



*1.2.通过模型对应的预处理脚本对数据进行预处理，并保存成bin格式的文件。以保证模型输入层数据类型大小与预处理的数据类型大小一致*

*运行命令：python3.7.5 yolo_tf_preprocess.py --src_info ./coco2017.info --save_path ./input_bin*

*运行成功：![image-20210710100044911](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210710100044911.jpg)*



*参数说明：src_info：生成的图片信息文件所在的路径*

​                   *save_path：经过预处理后的数据保存路径，文件名用户可以自定义。*



*1.3.生成数据集文件*

*运行命令：python3.7 get_yolo_info.py input_bin coco2017.info yolov5.info*

*运行成功，数据集文件yolov5.info保存在当前目录*



##### 2.性能推理



运行命令：/benchmark.aarch64 -model_type=yolocaffe -batch_size=1 -device_id=0  -om_path=./helmet_head_person_s_1.7.0_op11_dbs_sim_t.om -input_width=460 -input_height=460 input_text_path=./yolov5.info -useDvpp=false -output_binary=true

error:

![image-20210710151853941](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210710151853941.jpg)

 处理：**input_text_path输入为二进制文件，而模型的输入为图片，所以修改为使用input_imgFiles_path参数，直接输入图片，不在使用数据集文件，且使用DVpp=true**。使用数据图片：    /home/sd_xiong2/ExampleProject/HelmetIdentification/benchmark_tools/test_imgFiles

命令输入：  ./benchmark.aarch64 -model_type=yolocaffe -batch_size=1 -device_id=0 -om_path=./helmet_head_person_s_1.7.0_op11_dbs_sim_t.om -input_width=640 -input_height=640 -input_imgFiles_path=./test_imgFiles -useDvpp=true -output_binary=False 

运行成功:

 ![image-20210710170200571](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210710170200571.jpg)                            

输出参数说明：

[e2e]  throughputRate：端到端总吞吐率。公式为sample个 数/一次推理时间。 latency：端到端时延，即从第一个sample到最后一个sample的完成时间

[data read] [preprocess] [post]  throughputRate： 当前模块的吞吐率      moduleLatency ：执行一次当前模块的时延。

[infer] throughputRate：推理模块的吞吐率，公式为sample个 数/执行一次推理的时间。 moduleLatency： 推理模块的平均时延。公式为执行一次推理的时间/batch size。Interface throughputRate： aclmdlExecute接口的吞吐率。公式为 sample个数/aclmdlExecute接口的平均 执行时间。

同时在“result/dumpOutput_device0”下生成各数据的推理结果文件

##### 3. mAP精度





## 四. 第三周开发（2021.7.12-2021.7.17）

### 1.pipline编写

```python
# helmet detection pipline
{
    "Detection":{
      "stream_config":{
        "deviceId":"0"
       },
       "mxpi_rtspsrc0":{
            "factory":"mxpi_rtspsrc",
            "props":{
                "rtspUrl":"rtsp://xxx.xxx.xxx.xxx:xxxx/xxxx.264",
                "channelId":"0"
            },
            "next":"mxpi_videodecoder0"
        },
        "mxpi_rtspsrc1":{
            "factory":"mxpi_rtspsrc",
            "props":{
                "rtspUrl":"rtsp://xxx.xxx.xxx.xxx:xxxx/xxxx.264",
                "channelId":"1"
            },
            "next":"mxpi_videodecoder1"
        },
        "mxpi_videodecoder0":{
            "factory":"mxpi_videodecoder",
            "props":{
                "inputVideoFormat":"H264",
                "outputImageFormat":"YUV420SP_NV12",
                "vdecChannelId":"0"
            },
            "next":"mxpi_selectedframe0"
        },
        "mxpi_videodecoder1":{
            "factory":"mxpi_videodecoder",
            "props":{
                "inputVideoFormat":"H264",
                "outputImageFormat":"YUV420SP_NV12",
                "vdecChannelId":"1"
            },
            "next":"mxpi_selectedframe1"
        },
        "mxpi_selectedframe0":{
            "factory":"mxpi_selectedframe",
            "next":"mxpi_parallel2serial0:0",
            "props":{
                "frameNum":"2"
            }
        },
        "mxpi_selectedframe1":{
            "factory":"mxpi_selectedframe",
            "next":"mxpi_parallel2serial0:1",
            "props":{
                "frameNum":"2"
            }
        },
        "mxpi_parallel2serial0":{
            "factory":"mxpi_parallel2serial",
            "props":{
                "dataSource":"mxpi_videodecoder0,mxpi_videodecoder1"
            },
            "next":"mxpi_imageresize0"
        },
        "mxpi_imageresize0":{
            "props":{
                "dataSource":"mxpi_parallel2serial0",
                "resizeType": "Resizer_KeepAspectRatio_Fit",
                "resizeHeight":"640",
                "resizeWidth":"640"
            },
            "factory":"mxpi_imageresize",
            "next":"queue0"
        },
        "queue0":{
            "props":{
                "max-size-buffers":"500"
            },
            "factory":"queue",
            "next":"mxpi_modelinfer0"
        },
        "mxpi_modelinfer0":{
            "props":{
                "dataSource":"mxpi_imageresize0",
                "modelPath":"./Models/helmet_head_person_s_1.7.0_op11_dbs_sim_t.om",
                "postProcessConfigPath":"./Models/Helmet_yolov5.cfg",
                "labelPath":"./Models/imgclass.names",
                "postProcessLibPath":"/home/sd_xiong2/MindX_SDK/mxVision/lib/libMpYOLOv5PostProcessor.so"
            },
            "factory":"mxpi_modelinfer",
            "next":"mxpi_objectdistributor0"
        },
        "mxpi_objectdistributor0":{
            "props":{
                "channelIds":"0,1",
                "dataSource":"mxpi_modelinfer0"
            },
            "factory":"mxpi_distributor",
            "next":["mxpi_dataserialize0","mxpi_dataserialize1"]
        },
        "mxpi_dataserialize0": {
            "props": {
                "outputDataKeys": "ReservedFrameInfo,mxpi_modelinfer0"
            },
            "factory": "mxpi_dataserialize",
            "next": "appsink0"
        },
        "mxpi_dataserialize1": {
            "props": {
                "outputDataKeys": "ReservedFrameInfo,mxpi_modelinfer1"
            },
            "factory": "mxpi_dataserialize",
            "next": "appsink1"
        },
        "appsink0": {
            "props": {
                "blocksize": "4096000"
            },
            "factory": "appsink"
        }
        "appsink1": {
            "props": {
                "blocksize": "4096000"
            },
            "factory": "appsink"
        }

    }
}
```



### 2.插件参数

1. mxpi_rtspsrc：

   ①若pipeline中使用了拉流插件（mxpi_rtspsrc），建议在运行前设置环境变量GST_DEBUG，这样 取流地址配置不正确时，有warning日志提示。设置环境变量的命令为： **export GST_DEBUG=3** 

   ②**rtspUrl**为rtsp取流地址(可以从网络摄像机获取， 也可通过Live555等工具将本地视频文 件转换为rtsp流），**channelId**为表示视频拉流的路数标识，默认值为 0。

2. mxpi_videodecoder：

   ①该插件当前只支持H264/H265格式，**inputVideoFormat**为输入视频流的格式，默认为H264。**outputImageFormat**为解码的输出图像格式，默认为 YUV420SP_NV12，暂时只能填写 YUV420SP_NV12或者 YUV420SP_NV21。**vdecChannelId**为视频解码通道号，默认为0，取值范围 为[0,31]。每个视频拉流插件应使用不 同的解码通道号。

4. mxpi_parallel2serial

   ①**dataSource**为输入数据对应索引（通常情况下为上游元件名称），可以配置多个，以逗号隔开。当不需要在该插件中挂载元数据时，可以不使用该属性。配置dataSource属性，串行化插件会挂载元数据，并将数据按照接收的顺序发送 给下游插件。假定串行化插件mxpi_parallel2serial0接受数据的顺序为demoA0，demoA1， demoA1，demoA0，demoA0...，那么串行化插件将以demoA0为key在demoA传 递的buffer上获取元数据，然后以mxpi_parallel2serial0为key挂载前面获取的元数据，最后将buffer发送给下游插件demoB

5. mxpi_imageresize

   ①**removeParentData**为删除原Buffer数据，默认否。**dataSource**为输入数据对应索引（通常情况下为上游 元件名称）。默认为上游插件对应输出 端口的key值。**resizeHeight、resizeWidth**为 Resizer_Stretch 和 Resizer_KeepAspectRatio_Fit 缩放模式 中，指定缩放后的高和宽。

   ②**resizeType** 缩放方式： ● Resizer_Stretch ：拉伸缩放，默认缩 放方式。支持opencv和ascend。● Resizer_KeepAspectRatio_Fit ：等比 缩放，使图片等比缩放至在指定宽高 的区域内面积最大化。只支持 ascend。

   ③**paddingType** 为补边方式。 ● Padding_NO(默认):不补边 ● Padding_RightDown：右下方补边 ● Padding_Around：上下左右补边。paddingHeight、paddingWidth为补边后的高和宽。

6. mxpi_dataserialize

   ①**outputDataKeys** 为指定需要输出的数据的索引（通常情况 下为元件名称），以逗号隔开。此插件根据用户选择的元件名，将元件数据拼接成json字符串。该json字符串用于根据 插件的依赖关系输出组装结果

7. mxpi_modelinfer

   ①**modelPath**指定推理模型om文件路径。**postProcessConfigPath**为后处理配置文件路径。**postProcessConfigContent**为后处理配置。labelPath为后处理类别标签路径。 **postProcessLibPath** 后处理动态链接库so文件路径。如果不指定，则不进行后处理，直接将模型推 理结果写入元数据 MxpiTensorPackageList 并将内存拷贝到 outputDeviceId指定位置。**tensorFormat**值为0时采用NHWC，值为1时采用 NCHW，默认为0。

   ② postProcessConfigPath和postProcessConfigContent两个属性的目的都是获取后处理的配置内容，不同点在于一个是直接将内容写出来，另一个是以文件的形式给出，实际使用中只用使用其中一个属性即可。



### 3.抽帧插件（自定义）

#### 1.准备文件

CMakeLists.txt ：主要用于设置插件名、添加生成插件动态库的目标文件以及链接相关的第三方库。

 MxpiSelectedFrame.cpp： 主文件

MxpiSelectedFrame.h： 头文件

文件放置在/home/sd_xiong2/ExampleProject/HelmetIdentification/plugins/MxpiSelectedFrame

#### 2.命令

mkdir build

cmake ..

make -j

#### 3.ERROR

**error1**：![image-20210713170424169](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210713170424169.jpg)

处理：添加环境变量

```c++
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/include)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/opensource/include)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/opensource/include/gstreamer-1.0)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/opensource/include/glib-2.0)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/opensource/lib/glib-2.0/include)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/include/MxTools/PluginToolkit)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/include/MxTools/PluginToolkit)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/include/MxTools/PluginToolkit)

link_directories(/home/sd_xiong2/MindX_SDK/mxVision/lib)
link_directories(/home/sd_xiong2/MindX_SDK/mxVision/opensource/lib)
```



**error2**：![image-20210713211321067](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210713211321067.jpg)

处理：CMakeLists中代码的顺序问题，改为样例中的顺序运行成功

```c++
cmake_minimum_required(VERSION 3.5.2)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set(PLUGIN_NAME "mxpi_selectedframe")
set(TARGET_LIBRARY ${PLUGIN_NAME})

include_directories(/home/sd_xiong2/MindX_SDK/mxVision/include)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/opensource/include)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/opensource/include/gstreamer-1.0)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/opensource/include/glib-2.0)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/opensource/lib/glib-2.0/include)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/include/MxTools/PluginToolkit)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/include/MxTools/PluginToolkit)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/include/MxTools/PluginToolkit)
include_directories(/home/sd_xiong2/MindX_SDK/mxVision/lib)

link_directories(/home/sd_xiong2/MindX_SDK/mxVision/lib)
link_directories(/home/sd_xiong2/MindX_SDK/mxVision/opensource/lib)

add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")
add_library(${TARGET_LIBRARY} SHARED MxpiSelectedFrame.cpp)

target_link_libraries(${TARGET_LIBRARY} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0 gmodule-2.0)
target_link_libraries(${TARGET_LIBRARY} plugintoolkit mxbase mxpidatatype)
target_link_libraries(${TARGET_LIBRARY} -Wl,-z,relro,-z,now,-z,noexecstack -s)

install(TARGETS ${TARGET_LIBRARY} LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/dist/lib)
```

编译成功，产生**libmxpi_selectedframe.so**文件存放在/home/sd_xiong2/ExampleProject/HelmetIdentification/plugins/MxpiSelectedFrame/build，将文件copy至SDK的插件库中。

### 4.后处理文件xxx.so

sdk中已经包含有YOLOv5的后处理文件：libMpYOLOv5PostProcessor.so

参数配置为：

| 参数名            | 描述                                                         | 设置参数值                                                   |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| CLASS_NUM         | 类别数量                                                     | 3                                                            |
| BIASES_NUM        | anchor宽高的数量（18表 示9个anchor，每个对应 一对宽高值）。  | 18                                                           |
| BIASE             | 每两个数组成一个anchor 的宽高值，例如10、13表 示第一个anchor的宽、高 值。 | 10，13，16，30，33， 23，30，61，62，45， 59，119，116，90， 156，198，373，326 |
| SCORE_THRESH      | 目标是否为某种类别物体 的阈值，大于阈值即认为 是该目标。     | 0.4                                                          |
| OBJECTNESS_THRESH | 是否为目标的阈值，大于 阈值即认为是目标。                    | 0.6                                                          |
| IOU_THRESH        | 两个框的IOU阈值，超过 阈值即认为同一个框。                   | 0.5                                                          |
| YOLO_TYPE         | 表示输出tensor的个数，3 表示有三个feature map 输出。         | 3                                                            |
| ANCHOR_DIM        | 每个feature map对应的 anchor框数量。                         | 3                                                            |
| MODEL_TYPE        | 数据排布格式，0表示 NHWC，1表示NCHW。                        | 1                                                            |





### 5.搭建一个简单地推流服务器

1.下载VLC media player：[VLC media player for Windows](https://www.videolan.org/vlc/download-windows.html)

在 媒体/流 中将添加本地视频 生成rstp流。流地址为rtsp://10.10.11.201:8554/

选择另外客户端 添加流地址。在服务端点击【流】即可直播。



### 6.调用主文件

#### 1.主文件

```python
from StreamManagerApi import *

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("/home/sd_xiong2/ExampleProject/HelmetIdentification/Models/HelmetDetection.pipline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Obtain the inference result by specifying streamName and uniqueId.
    streamName = b'Detection'
    for i in range(2):
      inferResult[i] = streamManagerApi.GetResult(streamName, i, 3000)
      if inferResult[i].errorCode != 0:
          print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            inferResult[i].errorCode, inferResult[i].data.decode()))
          exit()
      print(inferResult.data.decode())  # print the infer result

    # destroy streams
    streamManagerApi.DestroyAllStreams()
```

#### 2.运行

source /home/sd_xiong2/.bashrc

source /home/sd_xiong2/ExampleProject/HelmetIdentification/Models/main-env.sh

 python3.7.5 main.py



#### 3.ERROR

##### error1.



![image-20210717141906198](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210717141906198.jpg)

处理：环境变量设置问题

①.bashrc 文件中设置如下：

```c++
export MX_SDK_HOME=/home/sd_xiong2/MindX_SDK/mxVision
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:${install_path}/acllib/lib64:/usr/local/Ascend/ascend-toolkit:${install_path}/arm64-linux/atc/lib64:/usr/local/Ascend/driver/lib64:${MX_SDK_HOME}/python:${LD_LIBRARY_PATH}"
export PYTHONPATH="${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:${MX_SDK_HOME}/python:${PYTHONPATH}"
export PATH="/usr/local/python3.7.5/bin:${install_path}/arm64-linux/atc/ccec_compiler/bin:${install_path}/arm64-linux/atc/bin:${install_path}/atc/bin:${MX_SDK_HOME}/python:$PATH"
```

②main-env.sh中设置如下：

```c++
export MX_SDK_HOME=/home/sd_xiong2/MindX_SDK/mxVision
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/arm64-linux/atc/ccec_compiler/bin:${install_path}/arm64-linux/atc/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:/usr/local/python3.7.5/bin:/usr/local/lib/python3.7/dist-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:${install_path}/acllib/lib64:/usr/local/Ascend/driver/lib64:${install_path}/arm64-linux/atc/lib64:${install_path}/acllib_linux.arm64/lib64:${MX_SDK_HOME}/include:${MX_SDK_HOME}/python${LD_LIBRARY_PATH}


export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
export ASCEND_OPP_PATH=${install_path}/opp
export GST_DEBUG=3
echo "successfully!!"
```



##### error2.

报错： undefined symbol: _ZN6google10LogMessage9SendToLogEv

处理：在编译插件CMakeLists.txt添加如下编译参数 set(PLUGIN_NAME "mxpi_selectedframe")



##### error3.

[6005] [stream invaldid config]  Invalid stream config. Parse json value of stream failed. Error message: (* Line 46, Column 13 Syntax error: Malformed object literal). Failed to create Stream, ret=6005

处理： 语法格式错误。

##### error4.

![image-20210717162923226](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210717162923226.jpg)

处理：将插件mxpi_motsimplesortV2修改为mxpi_motsimplesort

##### error5.

![image-20210717200635763](https://github.com/liukai123456789/HelmetLog/blob/main/images/image-20210717200635763.jpg)




