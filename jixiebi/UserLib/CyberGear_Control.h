#ifndef __CYBERGEAR_CONTROL_H
#define __CYBERGEAR_CONTROL_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "cybergear.h" // 包含此文件以获取 MI_Motor 类型定义

/* Public Typedefs -----------------------------------------------------------*/
/**
  * @brief  定义电机ID，用于访问状态变量
  */
typedef enum {
    MOTOR_ID_ROTATION = 0, // 旋转电机 (对应 mi_motor[0])
    MOTOR_ID_LIFT     = 1  // 升降电机 (对应 mi_motor[1])
} Motor_ID_t;

/**
  * @brief  定义不同的增益参数档案（Profile）
  */
typedef enum {
    GAIN_SET_LOADED,    // 带载增益
    GAIN_SET_UNLOADED   // 空载增益
} GainProfile_t;

/* Public Function Prototypes ------------------------------------------------*/

// --- 获取和设置内部状态 ---
// 【已修改】增加 Motor_ID_t 参数以选择电机
float CyberGear_GetLastAngle(Motor_ID_t motor_id);
void CyberGear_SetLastAngle(Motor_ID_t motor_id, float angle_deg);

// --- 高级控制逻辑 ---
//void MiAngleControl_MultiStage(float target_angle_deg);
//void MiAngleControl_Dynamic(float target_angle_deg);
// 使用混合模式
void Move_and_Hold_Hybrid(MI_Motor* Motor, float target_pos_deg, GainProfile_t profile);
//【V3版】基于6节点5段的精细化动态PD控制器（带完全可调参数）
void MiAngleControl_MultiStage_V3(MI_Motor* Motor, float target_angle_deg, GainProfile_t profile, 
                                  uint8_t enable_timeout, uint32_t timeout_ms,
                                  float stop_tolerance_deg, float stop_speed_dps);

// --- 中层封装函数 (接口单位已统一为 度 和 度/秒) ---

/**
  * @brief  设置位置模式的目标位置和目标速度
  */
void set_position_target_and_speed(MI_Motor *Motor, float target_pos_deg, float target_speed_dps);

/**
  * @brief  移动到指定角度，等待指定时间后设置保持速度
  */
void CyberGear_Move_Then_Hold(MI_Motor* Motor, float target_pos_deg, float move_speed_dps, 
                              float hold_speed_dps, uint32_t timeout_ms);

/**
  * @brief  使用位置模式同步移动到指定角度
  */
uint8_t CyberGear_MoveTo_Sync_PosMode(MI_Motor* Motor, float target_pos_deg, float move_speed_dps, 
                                      float tolerance_deg, uint32_t stable_ms, 
                                      uint8_t enable_timeout, uint32_t timeout_ms);

// --- 带梯形速度规划的同步移动函数 ---
//uint8_t CyberGear_MoveTo_Trapezoidal(MI_Motor* Motor, float target_pos_deg,
//                                     float cruise_speed_dps, float hold_speed_dps,
//                                     float accel_decel_distance_deg, float start_end_speed_dps,
//                                     float tolerance_deg, uint32_t stable_ms,
//                                     uint8_t enable_timeout, uint32_t timeout_ms);

 // --- 带可调加减速距离的梯形速度规划同步移动函数 V2 ---
 uint8_t CyberGear_MoveTo_Trapezoidal_V2(Motor_ID_t motor_id, MI_Motor* Motor, float target_pos_deg,
                                        float cruise_speed_dps, float hold_speed_dps,
                                        float accel_distance_deg, float decel_distance_deg,
                                        float start_end_speed_dps,
                                        float tolerance_deg, uint32_t stable_ms,
                                        uint8_t enable_timeout, uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* __CYBERGEAR_CONTROL_H */
