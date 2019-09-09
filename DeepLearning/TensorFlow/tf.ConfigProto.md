### `TensorFlow`中使用`tf.ConfigProto()`配置`session`参数

`tf.ConfigProto()`函数在创建`session`的时候，用来对`session`进行参数配置：
```
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.4 # 占用40%显存
sess = tf.Session(config=config)
```

#### 记录设备指派情况:`log_device_placement`

设置`tf.ConfigProto()`中参数`log_device_placement=True`，可以获取到`operations`和`Tensor`被指派到哪个设备(几号CPU或几号GPU)上运行并在终端上进行输出

#### 自动选择运行设备:`allow_soft_placement`

在`tf`中，通过命令`with tf.device('/cpu:0'):`允许手动设置操作运行的设备，如果手动设置的设备不存在或者不可用，就会导致`tf`程序等待或异常，为了防止这种情况可以设置`tf.ConfigProto()`中参数`allow_soft_placement=True`，允许`tf`自动选择一个存在且可用的设备进行操作

- 限制GPU资源使用:
为了加快工作效率，`TensorFlow`在初始化时会尝试分配所有可用的GPU显存资源给自己，这在多人使用的服务器上工作会导致GPU占用，别人无法使用GPU工作的情况

`tf`提供了两种控制GPU资源使用的方法，一是**让`TensorFlow`在运行过程中动态申请显存，需要多少就申请多少**；二是**限制GPU的使用率**

1. 动态申请显存
```
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
```

2. 限制GPU使用率
```
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存
session = tf.Session(config=config)
```
或者：
```
gpu_options = tf.GPUOptions(per_process_gpu_memory_fracion=0.4)
config = tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config)
```

#### 设置使用哪块GPU
1. 在python程序中设置：
```
os.environ['CUDA_VISIBLE_DIVICES'] = '0' # 使用GPU0
os.environ['CUDA_VISIBLE_DIVICES'] = '0, 1' # 使用GPU 0, 1
```

2. 在执行`python`程序时：
```
$ CUDA_VISIBLE_DEVICES=0,1 python yourcode.py
```
