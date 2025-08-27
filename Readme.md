# LingXiao-Crane2025-RPi-STM32

2025年起重机赛“凌霄”作品，BCC编写的视觉和控制代码，包括树莓派端视觉系统代码、STM32接收处理视觉信息模块和STM32机械臂分控代码（包含封装好的小米cybergear关节微电机高级控制函数）。

## 项目结构

- `vision_system_raspberrypi/` - 树莓派视觉识别系统
  - 详见 [vision_system_raspberrypi/README.md](vision_system_raspberrypi/README.md)
  
- `receive_from_pi_module/` - STM32视觉数据接收解析模块
  - 详见 [receive_from_pi_module/README.md](receive_from_pi_module/README.md)
  
- `jixiebi/` - STM32机械臂分控代码
  - 详见 [jixiebi/README.md](jixiebi/README.md)
