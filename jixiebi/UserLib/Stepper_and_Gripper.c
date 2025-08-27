/**
  ******************************************************************************
  * @file       : stepper_control.c
  * @brief      : 步进电机及夹爪控制函数的实现
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "Stepper_and_Gripper.h"
#include "tim.h"      // 需要包含tim.h以使用舵机PWM功能
#include "Delay_us.h" // 需要包含Delay_us.h以使用微秒延时
#include "CyberGear.h"
#include "CyberGear_Control.h"
#include "math.h"

/* Function Definitions ------------------------------------------------------*/

/**
  * @brief  底层步进电机脉冲控制函数
  * @param  dir: 运动方向 (0: 向下, 1: 向上)
  * @param  posi_cm: 移动距离 (单位: cm)
  * @param  is_loaded: 是否有负载 (0: 空载, 1: 有负载)
  * @retval None
  */
void bujin(int dir, double posi_cm, int is_loaded)
{
    // 设置电机方向引脚
    HAL_GPIO_WritePin(GPIOB, DIRECT_Pin, (dir == 0) ? GPIO_PIN_SET : GPIO_PIN_RESET);
    
    // 根据距离(cm)计算所需的脉冲数
    uint32_t pulse_count = (uint32_t)(posi_cm * STEPS_PER_CM);
    
    // 根据负载情况选择不同的速度（脉冲延时）
    if (is_loaded == 1) {
        for (uint32_t i = 0; i < pulse_count; ++i) {
            HAL_GPIO_WritePin(GPIOB, PUL_Pin, GPIO_PIN_RESET);
            Delay_us(PULSE_DELAY_US_LOADED);
            HAL_GPIO_WritePin(GPIOB, PUL_Pin, GPIO_PIN_SET);
            Delay_us(PULSE_DELAY_US_LOADED);
        }
    } else {
        for (uint32_t i = 0; i < pulse_count; ++i) {
            HAL_GPIO_WritePin(GPIOB, PUL_Pin, GPIO_PIN_RESET);
            Delay_us(PULSE_DELAY_US_UNLOADED);
            HAL_GPIO_WritePin(GPIOB, PUL_Pin, GPIO_PIN_SET);
            Delay_us(PULSE_DELAY_US_UNLOADED);
        }
    }
}

/**
  * @brief  执行一次完整的“取物”动作序列
  * @param  posi_cm: 需要下降的距离 (单位: cm)
  * @retval None
  */
//void get(double posi_cm)
//{
//    __HAL_TIM_SetCompare(&GRIPPER_TIM_HANDLE, GRIPPER_TIM_CHANNEL, GRIPPER_OPEN_PWM); // 打开夹爪
//    HAL_Delay(GRIPPER_ACTION_DELAY_MS);
//    
//    bujin(0, posi_cm, 0); // 空载下降
//    HAL_Delay(SEQUENCE_ACTION_DELAY_MS);
//    
//    __HAL_TIM_SetCompare(&GRIPPER_TIM_HANDLE, GRIPPER_TIM_CHANNEL, GRIPPER_CLOSE_PWM); // 闭合夹爪
//    HAL_Delay(GRIPPER_ACTION_DELAY_MS);
//    
//    bujin(1, posi_cm, 1); // 带载上升
//    HAL_Delay(10); // 短暂延时
//}

/**
  * @brief  执行一次完整的“放置”动作序列
  * @param  down_cm: 需要下降的距离 (单位: cm)
  * @retval None
  */
//void place(double down_cm)
//{
//    bujin(0, down_cm, 1); // 带载下降
//    HAL_Delay(SEQUENCE_ACTION_DELAY_MS);
//    
//    __HAL_TIM_SetCompare(&GRIPPER_TIM_HANDLE, GRIPPER_TIM_CHANNEL, GRIPPER_OPEN_PWM); // 打开夹爪
//    HAL_Delay(GRIPPER_ACTION_DELAY_MS);
//    
//    bujin(1, PLACE_RETRACT_CM, 0); // 空载上升一小段距离
//    __HAL_TIM_SetCompare(&GRIPPER_TIM_HANDLE, GRIPPER_TIM_CHANNEL, GRIPPER_CLOSE_PWM); // （可选）闭合夹爪
//    
//    bujin(1, down_cm - PLACE_RETRACT_CM, 0); // 空载上升剩余距离
//    HAL_Delay(10); // 短暂延时
//}


/**
  * @brief  【V2版】控制升降电机移动到指定高度，带动态速度和超时
  * @param  target_cm: 目标高度 (单位: cm)，0为最低点
  * @param  is_loaded: 是否带载 (0: 空载, 1: 带载)
  * @retval uint8_t: 1 表示成功, 0 表示超时
  */
uint8_t Lift_MoveTo_cm(float target_cm, int is_loaded)
{
    // 1. 输入参数安全检查
    if (target_cm < 0) target_cm = 0;
    if (target_cm > LIFT_MAX_HEIGHT_CM) target_cm = LIFT_MAX_HEIGHT_CM;

		// 2. 根据负载和方向选择速度
    // a. 获取起始位置(cm)
    float start_cm = CyberGear_GetLastAngle(MOTOR_ID_LIFT) / LIFT_DEGREES_PER_CM;
		// b. 判断方向
    uint8_t is_going_up = (target_cm > start_cm);
    // c. 选择对应的速度
    float move_speed_cms;
    if (is_going_up) {
        move_speed_cms = is_loaded ? LIFT_SPEED_UP_LOADED_CMS : LIFT_SPEED_UP_UNLOADED_CMS;
    } else {
        move_speed_cms = is_loaded ? LIFT_SPEED_DOWN_LOADED_CMS : LIFT_SPEED_DOWN_UNLOADED_CMS;
    }
		// 3. 动态计算超时时间
    float distance_cm = fabs(target_cm - start_cm);
    uint32_t timeout_ms = (move_speed_cms > 0) ? 
                          ((uint32_t)((distance_cm / move_speed_cms) * 1000.0f) + LIFT_TIMEOUT_BUFFER_MS) : 
                          LIFT_TIMEOUT_BUFFER_MS;
		
    // 4. 根据运动方向，选择不同的加减速距离(cm)
    float accel_cm;
    float decel_cm;
    if (is_going_up) {
        accel_cm = LIFT_ACCEL_UP_CM;
        decel_cm = LIFT_DECEL_UP_CM;
    } else { // is_going_down
        accel_cm = LIFT_ACCEL_DOWN_CM;
        decel_cm = LIFT_DECEL_DOWN_CM;
    }

    // 5. 在调用底层函数前，将所有cm和cm/s单位转换为电机需要的度(deg)和度/秒(dps)
    float target_deg = target_cm * LIFT_DEGREES_PER_CM;
    float cruise_dps = move_speed_cms * LIFT_DEGREES_PER_CM;
    float hold_dps = LIFT_HOLD_SPEED_CMS * LIFT_DEGREES_PER_CM;
    float accel_deg = accel_cm * LIFT_DEGREES_PER_CM;
		float decel_deg = decel_cm * LIFT_DEGREES_PER_CM;
    float start_end_dps = LIFT_START_END_SPEED_CMS * LIFT_DEGREES_PER_CM;
    float tolerance_deg = LIFT_TOLERANCE_CM * LIFT_DEGREES_PER_CM;
    
    // 6. 调用底层的梯形速度规划函数来执行移动
    uint8_t result = CyberGear_MoveTo_Trapezoidal_V2(
				MOTOR_ID_LIFT,									// 升降电机是1
        &mi_motor[1],       // 控制升降电机 mi_motor[1]
        target_deg,
        cruise_dps,
        hold_dps,
        accel_deg,
        decel_deg,
        start_end_dps,
        tolerance_deg,
        200,                // 稳定时间 (ms)
        1,                  // 启用超时
        timeout_ms
    );
		
		CyberGear_SetLastAngle(MOTOR_ID_LIFT, target_cm * LIFT_DEGREES_PER_CM); // 更新目标角度
		
		return result;
}

/**
  * @brief  【V2版】执行一次完整的“取物”动作序列
  * @param  target_height_cm: 下降的的高度 (单位: cm)
  */
void get(double target_height_cm)
{
    // 1. 打开夹爪
    __HAL_TIM_SetCompare(&GRIPPER_TIM_HANDLE, GRIPPER_TIM_CHANNEL, GRIPPER_OPEN_PWM);
    HAL_Delay(GRIPPER_ACTION_DELAY_MS);
    
    // 2. 空载(0)下降到目标物体高度
    Lift_MoveTo_cm(target_height_cm, 0);
    
    // 3. 闭合夹爪
    __HAL_TIM_SetCompare(&GRIPPER_TIM_HANDLE, GRIPPER_TIM_CHANNEL, GRIPPER_CLOSE_PWM);
    HAL_Delay(GRIPPER_ACTION_DELAY_MS);
    
    // 4. 带载(1)上升回到最高点
    Lift_MoveTo_cm(LIFT_MAX_HEIGHT_CM, 1);
}

/**
  * @brief  【V2版】执行一次完整的“放置”动作序列
  * @param  target_height_cm: 目标放置点的高度 (单位: cm)
  */
void place(double target_height_cm)
{
    // 1. 带载(1)下降到目标放置高度
    Lift_MoveTo_cm(target_height_cm, 1);
    
    // 2. 打开夹爪
    __HAL_TIM_SetCompare(&GRIPPER_TIM_HANDLE, GRIPPER_TIM_CHANNEL, GRIPPER_OPEN_SMALLER_PWM);
    HAL_Delay(GRIPPER_ACTION_DELAY_MS);
    
    // 3. 空载(0)上升一小段距离，以完全脱离物体
    //Lift_MoveTo_cm(target_height_cm+PLACE_RETRACT_CM, 0);

    // 4. 闭合夹爪，防止爪子碰到铝管
//    __HAL_TIM_SetCompare(&GRIPPER_TIM_HANDLE, GRIPPER_TIM_CHANNEL, GRIPPER_CLOSE_PWM);
//		HAL_Delay(GRIPPER_ACTION_DELAY_MS);
    
    // 5. 空载(0)回到起始高度
		Lift_MoveTo_cm(LIFT_MAX_HEIGHT_CM, 0);
}
