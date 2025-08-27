# 2025起重机赛视觉识别系统

## 个人贡献

该项目全部代码均由本人完成，使用AI辅助。包含摄像头控制、YOLO目标检测、串口通信和系统集成四个核心模块。

## 代码结构

### 主要模块

**main.py - 主程序模块**

- `VisionSystemV2`类：系统主控制器
- `continuous_mode_with_signal_control()`：传统连续识别模式（投票机制）
- `continuous_matching_mode()`：连续匹配模式（要求连续N次相同结果）
- `visual_mode_with_signal_control()`：可视化调试模式
- `debug_mode()`：详细调试模式
- `is_perfect_result()`：完美结果检测函数
- `on_request_perfect_result()`：处理STM32请求完美数据的回调

**camera_module.py - 摄像头模块**

- `CameraModule`类：摄像头管理器
- `init_cameras()`：初始化货架和区域摄像头
- `set_v4l2_controls()`：通过v4l2-ctl设置摄像头参数
- `capture_images_with_interval()`：按指定间隔批量拍摄
- `get_images_for_yolo()`：为YOLO提供格式化图像数据

**yolo_module.py - YOLO识别模块**
- `YOLOModule`类：YOLO模型管理器
- `detect_shelf_numbers()`/`detect_area_numbers()`：货架和区域识别
- `map_shelf_position()`/`map_area_position()`：坐标到位置的映射
- `get_consensus_result()`：多次识别的共识算法
- `format_result_string()`：格式化最终识别结果

**uart_module.py - 串口通信模块**

- `UARTModule`类：串口通信管理器
- `process_received_signal()`：处理STM32发送的控制信号
- `set_request_perfect_result_callback()`：设置完美数据请求回调
- `send_result()`：发送识别结果到STM32

**sample_collector.py - 图像采集工具**

- 用于X11转发环境下的图像采集和标注

### 配置文件

**config.json**

- `camera`：摄像头参数配置，包括分辨率、曝光、对比度等v4l2参数
- `yolo`：模型路径、推理分辨率、置信度阈值等
- `uart`：串口配置和信号字符串定义
- `system`：系统运行参数和发送次数配置

### 模块调用关系

主程序main.py作为控制中心，初始化并协调三个核心模块：
1. CameraModule负责图像获取
2. YOLOModule接收图像进行识别
3. UARTModule处理与STM32的通信

连续模式流程：`continuous_mode_with_signal_control()` → `single_recognition_with_signal_control()` → `camera.get_images_for_yolo()` → `yolo.process_camera_data()` → `uart.send_result()`

连续匹配模式流程：`continuous_matching_mode()` → `ContinuousMatchingTracker.add_result()` → 达到连续匹配要求后发送结果

### 关键机制

#### 完美数据检测与缓存机制

系统实现了自动完美数据识别功能，通过`is_perfect_result()`函数检测识别结果是否符合比赛标准：

- 货架位置（1-6）：必须包含数字1-6各一次，无重复，无缺失
- 区域位置（a-f）：必须有且仅有一个空位（0或x），其余5个位置为不重复的数字1-6

完美结果在确认发送后自动缓存至`last_perfect_result`，STM32可通过"PERFECT"信号随时请求获取。

#### 信号控制机制

通过`process_received_signal()`处理STM32发送的控制信号：

- 状态控制信号：START/PAUSE/RESUME/STOP改变系统运行状态
- 数据请求信号：REQUEST获取最近成功结果，PERFECT获取最近完美结果
- 异步监听：独立线程`signal_listener_thread`监听串口输入，避免阻塞主识别流程

#### 共识算法机制

`get_consensus_result()`实现多次识别的共识判断：

- 投票模式（voting）：选择出现次数最多的识别结果
- 置信度模式（confidence）：选择置信度最高的识别结果
- 最小共识数量：需达到配置的最小检测次数才认为结果有效

### 运行模式流程

#### 连续识别模式（continuous）

采用传统的投票共识机制：

1. 系统进入运行状态（响应START信号）
2. 循环执行：拍摄 → YOLO识别 → 位置映射 → 共识算法
3. 每次识别后立即发送结果（无需等待连续匹配）
4. 支持信号控制的暂停/恢复功能
5. 自动保存成功结果和完美结果供后续请求

#### 连续匹配模式（continuous-matching）

采用连续匹配确认机制：

1. 使用`ContinuousMatchingTracker`跟踪连续识别结果
2. 每次识别后与历史结果比较，维护连续匹配计数
3. 只有连续N次（配置的required_matches）识别结果相同才确认发送
4. 确认后的每次匹配都会重复发送（可配置send_every_match）
5. 不同结果会重置匹配计数，重新开始统计

## 调试问题与解决方案

### 1. 摄像头曝光问题

**问题描述**：初期摄像头拍摄的图像存在严重过曝或欠曝现象，影响识别准确率。

**解决方案**：通过`v4l2-ctl --list-ctrls`查看摄像头所有可调参数，在`camera_module.py`的`set_v4l2_controls()`函数中实现了完整的v4l2参数控制。关键是启用了`auto_exposure=3`（自动曝光）和`exposure_dynamic_framerate=1`（动态帧率），解决了曝光问题。

### 2. YOLO模型迁移适配

**问题描述**：从YOLOv5迁移到YOLOv8+版本时，模型输出格式发生变化导致解析错误。

**解决方案**：借助AI修改了`detect_single_image()`函数中的输出解析逻辑。YOLOv8+使用`result.boxes`对象，需要通过`.xyxy[0].cpu().numpy()`、`.conf[0].cpu().numpy()`等方式获取坐标、置信度和类别信息。

### 3. 识别准确率优化

**问题描述**：识别偶尔会出错，希望把错误率降到0

**解决方案**：

- 从YOLOv5升级到YOLOv12，准确率和速度同时大幅提升
- 测试了yolov5\v8\v11\v12版本的s\m\l模型在640\960\1280\1440分辨率下的各种搭配组合，最终选定了以下使用的模型：
  - 货架模型：yolov12m_shelf_960.pt（960分辨率，中等模型）
  - 区域模型：yolov12m_area_1280.pt（1280分辨率，中等模型）

## 系统环境配置

主要是：加快启动速度、设置代码能够自动运行、进行全盘备份

### 系统基础配置
- **操作系统**：切换至Raspberry Pi OS Lite（命令行版本），移除桌面环境减少资源占用
- **启动优化**：`config.txt`中禁用彩虹屏、音频、蓝牙等非必要功能，`boot_delay=0`实现快速启动
- **串口配置**：启用UART0和UART1，禁用蓝牙释放GPIO14/15给串口使用

### 摄像头设备映射
`99-usb-cameras.rules`：

- 通过USB控制器路径区分两个摄像头：`1f00300000.usb`映射为`/dev/camera_shelf`，`1f00200000.usb`映射为`/dev/camera_area`
- 避免了因设备插拔顺序变化导致的摄像头混乱问题

### 服务化运行
**systemd服务配置**（`vision.service`）：

- 开机自启动视觉识别系统，运行`--continuous-matching --quiet`模式
- 日志输出重定向至systemd journal，便于调试和监控

**终端自动显示**（`.bashrc`）：
- tty1终端自动执行`journalctl -u vision.service -f`显示实时运行日志
- 系统启动后直接看到视觉系统运行状态，无需手动操作

### 环境管理与备份
- **Python环境**：使用pyenv管理Python 3.9.13环境，隔离系统Python避免依赖冲突
- **系统备份**：利用rpi-backup工具对整个SD卡进行完整备份
