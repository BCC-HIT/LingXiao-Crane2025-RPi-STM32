#ifndef __STEPPER_AND_GRIPPER_H
#define __STEPPER_AND_GRIPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h" // 包含此文件以获取HAL库的类型定义
#ifndef PI
#define PI 3.14159265358979323846264f
#endif

/* Public Macros and Constants -----------------------------------------------*/

// --- 步进电机物理参数定义 ---
// 假设步进电机通过周长为62.8mm的同步轮带动，减速比为30:1
// (0.628cm * 30 = 18.84cm) <- 注意：这里的物理意义需要您根据实际结构核对
// 此处我们遵循您原始代码的计算逻辑
#define STEPS_PER_CM            ((2800.0f * 4.0f) / (0.628f * 30.0f))

// --- 步进电机脉冲延时 (值越小速度越快) ---
#define PULSE_DELAY_US_LOADED   32  // us, 有负载时的脉冲延时
#define PULSE_DELAY_US_UNLOADED 28  // us, 空载时的脉冲延时

// --- 夹爪舵机控制参数 ---
#define GRIPPER_TIM_HANDLE      htim3       // 定义夹爪舵机使用的TIM句柄
#define GRIPPER_TIM_CHANNEL   TIM_CHANNEL_1 // 定义夹爪舵机使用的TIM通道
#define GRIPPER_OPEN_PWM      1825        // 舵机“打开”状态的PWM比较值
#define GRIPPER_OPEN_SMALLER_PWM      1725        // 舵机“打开”状态的PWM比较值
#define GRIPPER_CLOSE_PWM     1225        // 舵机“闭合”状态的PWM比较值

// --- 动作序列参数 ---
#define GRIPPER_ACTION_DELAY_MS 500         // ms, 夹爪开合后的等待时间
#define SEQUENCE_ACTION_DELAY_MS 100        // ms, 动作序列中的通用延时
#define PLACE_RETRACT_CM        5.0f        // cm, 放置动作中，松爪后抬升的距离
#define SEQUENCE_ACTION_DELAY_MS 100

// --- 升降电机物理与控制参数 ---
// 1. 物理参数 (务必根据您的实际结构进行测量和修改)
#define LIFT_WINCH_DIAMETER_CM  (7.6f)   // 缠绕线的线盘直径 (单位: cm)
#define LIFT_DEGREES_PER_CM     (360.0f / (PI * LIFT_WINCH_DIAMETER_CM))
#define LIFT_MAX_HEIGHT_CM      (47.0f)  // 爪子能到达的最高点高度 (单位: cm)，零点为最低点

// 2. 速度参数 (单位: cm/s, 分为四种工况)
#define LIFT_SPEED_UP_UNLOADED_CMS   (60.0f) // 空载上升速度
#define LIFT_SPEED_DOWN_UNLOADED_CMS (90.0f) // 空载下降速度
#define LIFT_SPEED_UP_LOADED_CMS     (50.0f) // 带载上升速度
#define LIFT_SPEED_DOWN_LOADED_CMS   (60.0f) // 带载下降速度

#define LIFT_HOLD_SPEED_CMS     (20.0f)   // 运动结束后用于保持位置的速度
#define LIFT_START_END_SPEED_CMS (20.0f)  // 梯形规划的起始/结束速度

// 3. 梯形规划与超时参数 (单位: cm)
#define LIFT_ACCEL_UP_CM        (10.0f) // 上升时的加速距离
#define LIFT_DECEL_UP_CM        (4.0f)  // 上升时的减速距离
#define LIFT_ACCEL_DOWN_CM      (3.0f)  // 下降时的加速距离
#define LIFT_DECEL_DOWN_CM      (8.0f)  // 下降时的减速距离
#define LIFT_TOLERANCE_CM       (1.5f) // 位置误差容忍度
#define LIFT_TIMEOUT_BUFFER_MS  (600)  // 在理论时间上增加的额外超时缓冲 (毫秒)

/* Public Function Prototypes ------------------------------------------------*/

/**
  * @brief  底层步进电机脉冲控制函数
  * @param  dir: 运动方向 (0: 向下, 1: 向上)
  * @param  posi_cm: 移动距离 (单位: cm)
  * @param  is_loaded: 是否有负载 (0: 空载, 1: 有负载)
  * @retval None
  */
void bujin(int dir, double posi_cm, int is_loaded);

/**
  * @brief  控制升降电机移动到指定高度，带动态速度和超时
  * @param  target_cm: 目标高度 (单位: cm)，0为最低点
  * @param  is_loaded: 是否带载 (0: 空载, 1: 带载)
  * @retval uint8_t: 1 表示成功, 0 表示超时
  */
uint8_t Lift_MoveTo_cm(float target_cm, int is_loaded);

/**
  * @brief  执行一次完整的“取物”动作序列
  * @param  posi_cm: 需要下降的距离 (单位: cm)
  * @retval None
  */
void get(double posi_cm);

/**
  * @brief  执行一次完整的“放置”动作序列
  * @param  down_cm: 需要下降的距离 (单位: cm)
  * @retval None
  */
void place(double down_cm);


#ifdef __cplusplus
}
#endif

#endif /* __STEPPER_AND_GRIPPER_H */
