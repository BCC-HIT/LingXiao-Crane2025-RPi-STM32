/**
  ******************************************************************************
  * @file       : CyberGear_Control.c
  * @brief      : 包含所有自定义的应用层电机控制算法
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "CyberGear_Control.h"
#include "main.h" // 包含 main.h 以便使用 HAL_Delay 等
#include "math.h"

#define PI 3.1415926f
/* Private Constants and Variables -------------------------------------------*/

// 定义一个静态全局变量，用于存储电机的上一次目标角度
// static 关键字确保此变量只在当前.c文件内可见
// 【已修改】将单个变量扩展为数组，以分别存储两个电机的状态
#define NUM_CYBERGEAR_MOTORS 2
static float g_last_angle_deg[NUM_CYBERGEAR_MOTORS] = {0.0f, 0.0f};

// --- MiAngleControl_MultiStage V3 使用的变量 ---
// --- 6节点5段动态增益调度参数 ---
// 当前的速度实测：带载90度：1.6秒，带载180度：2.3秒，空载90度：0.7秒，空载180度：1.0秒
// 【第1套：带载参数 (稍快)】
const float g_error_nodes_deg_loaded[6] = {150.0f, 80.0f, 60.0f, 20.0f, 5.0f, 0.0f};
const float g_kp_nodes_loaded[6]      = {1.0f, 2.4f, 3.6f, 8.0f, 13.5f, 2.5f};
const float g_kd_nodes_loaded[6]      = {0.8f, 0.8f, 0.8f, 1.8f, 4.0f, 5.0f};

// 【第2套：空载参数 (新增，数值需要调试)】
const float g_error_nodes_deg_unloaded[6] = {150.0f, 80.0f, 60.0f, 20.0f, 5.0f, 0.0f};
const float g_kp_nodes_unloaded[6]      = {2.0f, 3.0f, 5.0f, 12.0f, 27.0f, 10.0f}; 
const float g_kd_nodes_unloaded[6]      = {0.7f, 0.7f, 0.7f, 1.6f, 3.5f, 4.5f}; // Kd值比带载时略小

// 模拟积分项参数 (两套配置共用)
const float g_ki = 0.1f;
const float g_integral_active_error = 1.0f / 180.0 * 3.1415926;
const float g_integral_max_accum = 5.0f;


/* Private Function Prototypes (Static Functions) ----------------------------*/
// interpolate 函数只在本文件内部使用，所以声明为 static
static float interpolate(float current_val, float start_val, float end_val, float start_gain, float end_gain);

/* Function Definitions ------------------------------------------------------*/

// --- 线性插值辅助函数 ---
static float interpolate(float current_val, float start_val, float end_val, float start_gain, float end_gain) {
    if (start_val == end_val) return start_gain;
    float scale = (start_val - current_val) / (start_val - end_val);
    return start_gain + (end_gain - start_gain) * scale;
}

/**
  * @brief  【已修改】获取指定电机的上一次目标角度值
  * @param  motor_id: 电机ID (MOTOR_ID_ROTATION 或 MOTOR_ID_LIFT)
  */
float CyberGear_GetLastAngle(Motor_ID_t motor_id)
{
    // 增加安全检查，防止数组越界
    if (motor_id < NUM_CYBERGEAR_MOTORS) {
        return g_last_angle_deg[motor_id];
    }
    return 0.0f; // 如果ID错误，返回0
}

/**
  * @brief  【已修改】设置（更新）指定电机的上一次目标角度值
  * @param  motor_id: 电机ID (MOTOR_ID_ROTATION 或 MOTOR_ID_LIFT)
  * @param  angle_deg: 新的目标角度 (单位: 度)
  */
void CyberGear_SetLastAngle(Motor_ID_t motor_id, float angle_deg)
{
    if (motor_id < NUM_CYBERGEAR_MOTORS) {
        g_last_angle_deg[motor_id] = angle_deg;
    }
}

/**
 * @brief  按“角度位移 + 载荷档案”估算超时时间（毫秒）
 *
 * 设计思路（基于实测数据）：
 *  - 带载：90°≈1.6s，180°≈2.25s。用线性拟合得到：t_loaded ≈ 0.95 + 0.007222*θ（单位：秒）。
 *  - 空载：90°≈0.6~0.8s（取均值0.7s），180°≈1.0s。拟合：t_unload ≈ 0.40 + 0.003333*θ。
 *  - 为保证鲁棒性，再叠加两类冗余：
 *      1）安全系数（默认 15%）：覆盖摩擦波动、瞬态扰动；
 *      2）固定开销（默认 ~200ms）：模式切换、总线往返、固件调度等常量时间。
 *  - 上下限夹紧：
 *      空载：最短不低于 300ms，上限 1.8s
 *      带载：最短不低于 600ms，上限 3.0s
 *
 * 调参建议：
 *  - 若后续测得“短行程过保守”，可降低安全系数（例如从 1.15 调到 1.10）或下调固定开销；
 *  - 若“重载长行程偶发超时”，可略升安全系数或提高上限；
 *  - 需要更精细也可做分段：小角度一套系数，大角度另一套。
 *
 * @param  delta_deg   角度位移（度）。推荐由 fabsf(target - CyberGear_GetLastAngle()) 计算得到，
 *                    即“目标对目标”的差，不使用当前反馈角度，以贴合“计划运动量”的超时预估。
 * @param  profile     载荷档案：GAIN_SET_LOADED（带载） / GAIN_SET_UNLOADED（空载）
 * @return 估算得到的超时（毫秒）
 */
static uint32_t CyberGear_EstimateTimeoutMs(float delta_deg, GainProfile_t profile)
{
    // --- 1) 基于档案选择线性模型系数（单位换成 ms） ---
    //   带载：t = 0.95s + 7.222ms/deg * θ
    //   空载：t = 0.40s + 3.333ms/deg * θ
    float a_ms_per_deg;   // 斜率：每度需要的时间（毫秒/度）
    float b_ms;           // 截距：启动/切模/响应等“与角度无关的基础时间”（毫秒）
    if (profile == GAIN_SET_LOADED) {
        a_ms_per_deg = 7.222f;
        b_ms         = 950.0f;
    } else {
        a_ms_per_deg = 3.333f;
        b_ms         = 400.0f;
    }

    // --- 2) 线性估算（对角度取绝对值，模型仅关心位移大小） ---
    float abs_deg = (delta_deg >= 0.0f) ? delta_deg : -delta_deg;
    float t_ms    = b_ms + a_ms_per_deg * abs_deg;

    // --- 3) 上下限夹紧：防止过短（来不及完成）或过长（影响节拍） ---
    if (profile == GAIN_SET_LOADED) {
        if (t_ms < 600.0f)  t_ms = 600.0f;   // 带载小角度至少 0.6s
        if (t_ms > 3000.0f) t_ms = 3000.0f;  // 带载上限 3.0s
    } else {
        if (t_ms < 300.0f)  t_ms = 300.0f;   // 空载小角度至少 0.3s
        if (t_ms > 1800.0f) t_ms = 1800.0f;  // 空载上限 1.8s
    }

    // --- 4) 叠加冗余：安全系数（默认 15%） + 固定开销（默认 200ms） ---
    t_ms = t_ms * 1.15f + 200.0f;

    // --- 5) 返回毫秒整数 ---
    return (uint32_t)(t_ms);
}


/**
  * @brief  【V3版】基于6节点5段的精细化动态PD控制器（带完全可调参数）
  * @param  target_angle_deg: 目标角度 (单位: 度)
  * @param  profile: 选择要使用的增益参数档案 (带载/空载)
  * @param  enable_timeout: 是否启用超时 (1:启用, 0:禁用)
  * @param  timeout_ms: 最大等待时间/超时 (单位: 毫秒)
  * @param  stop_tolerance_deg: 退出循环的位置误差容忍度 (单位: 度)
  * @param  stop_speed_dps: 退出循环的速度误差容忍度 (单位: 度/秒)
  * @retval None
  */
void MiAngleControl_MultiStage_V3(MI_Motor* Motor, float target_angle_deg, GainProfile_t profile, 
                                  uint8_t enable_timeout, uint32_t timeout_ms,
                                  float stop_tolerance_deg, float stop_speed_dps)
{
    // --- 初始化 ---
    uint32_t start_time = HAL_GetTick();
    float target_rad = target_angle_deg / 180.0f * PI;
    static float error_accumulator = 0.0f;
    error_accumulator = 0.0f;
    
    float stop_speed_rad_s = stop_speed_dps / 180.0f * PI;
    
    // --- 根据profile选择参数组  ---
    const float* p_error_nodes;
    const float* p_kp_nodes;
    const float* p_kd_nodes;

    if (profile == GAIN_SET_LOADED) {
        p_error_nodes = g_error_nodes_deg_loaded;
        p_kp_nodes = g_kp_nodes_loaded;
        p_kd_nodes = g_kd_nodes_loaded;
    } else { // GAIN_SET_UNLOADED
        p_error_nodes = g_error_nodes_deg_unloaded;
        p_kp_nodes = g_kp_nodes_unloaded;
        p_kd_nodes = g_kd_nodes_unloaded;
    }

    // 将角度阈值转换为弧度
    static float rad_error_nodes[6];
    static uint8_t is_init = 0;
    if (!is_init) {
        for (int i = 0; i < 6; ++i) {
            rad_error_nodes[i] = g_error_nodes_deg_loaded[i] / 180.0f * PI;
        }
        is_init = 1;
    }

    // --- 主控制循环 ---
    float current_rad, error, error_abs;
    do {
        // 使用传入的 Motor 指针获取当前角度
        current_rad = Motor->Angle / 180.0f * PI;
        error = target_rad - current_rad;
        error_abs = fabs(error);
        
        float dynamic_kp = 0.0f;
        float dynamic_kd = 0.0f;

        // --- 增益计算  ---
        if (error_abs > rad_error_nodes[0]) {
            dynamic_kp = p_kp_nodes[0];
            dynamic_kd = p_kd_nodes[0];
        } else if (error_abs <= rad_error_nodes[4]) {
            if (error_abs < rad_error_nodes[5]) {
                dynamic_kp = p_kp_nodes[5];
                dynamic_kd = p_kd_nodes[5];
            } else {
                dynamic_kp = interpolate(error_abs, rad_error_nodes[4], rad_error_nodes[5], p_kp_nodes[4], p_kp_nodes[5]);
                dynamic_kd = interpolate(error_abs, rad_error_nodes[4], rad_error_nodes[5], p_kd_nodes[4], p_kd_nodes[5]);
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                if (error_abs <= rad_error_nodes[i] && error_abs > rad_error_nodes[i+1]) {
                    dynamic_kp = interpolate(error_abs, rad_error_nodes[i], rad_error_nodes[i+1], p_kp_nodes[i], p_kp_nodes[i+1]);
                    dynamic_kd = interpolate(error_abs, rad_error_nodes[i], rad_error_nodes[i+1], p_kd_nodes[i], p_kd_nodes[i+1]);
                    break;
                }
            }
        }

        // --- 积分项计算  ---
        float dynamic_target_rad = target_rad;
        if (error_abs < g_integral_active_error) {
            error_accumulator += error;
            if (error_accumulator > g_integral_max_accum) error_accumulator = g_integral_max_accum;
            if (error_accumulator < -g_integral_max_accum) error_accumulator = -g_integral_max_accum;
            dynamic_target_rad += g_ki * error_accumulator;
        }

        // --- 使用传入的 Motor 指针发送指令 ---
        motor_controlmode(Motor, 0, dynamic_target_rad, 0, dynamic_kp, dynamic_kd);
        
        // --- 超时检查 ---
        if (enable_timeout == 1) {
            if (HAL_GetTick() - start_time > timeout_ms) {
                break;
            }
        }

        HAL_Delay(2);

    // --- 使用传入的 Motor 指针获取当前速度和角度 ---
    } while (fabs(target_rad - Motor->Angle / 180.0f * PI) > (stop_tolerance_deg / 180.0f * PI) || fabs(Motor->Speed) > stop_speed_rad_s);
}



/**
  * @brief  使用混合模式，先精确移动，然后切换到位置模式保持
  * @param  Motor: 目标电机结构体指针
  * @param  target_pos_deg: 期望移动到的目标角度 (单位: 度)
  * @param  profile: 选择用于移动阶段的增益参数档案 (带载/空载)
  * @retval None
  */
void Move_and_Hold_Hybrid(MI_Motor* Motor, float target_pos_deg, GainProfile_t profile)
{
    // --- 阶段一: 使用“运控模式”进行高精度移动 ---

    // 1. 停止电机 (这是切换模式前的强制要求) 
    stop_cybergear(Motor, 0); // 第二个参数0代表不清除错误
    HAL_Delay(20);
    
    // 2. 切换到运控模式
    set_mode_cybergear(Motor, Motion_mode); // Motion_mode 是 motor.h 中定义的枚举，值为0
    HAL_Delay(20); // 模式切换后等待一小段时间

    // 3. 启动电机
    start_cybergear(Motor);
    HAL_Delay(20);

    // 4. 调用V3版高精度移动函数，并传入增益档案选择
    //    为V3函数的其他参数设定了在此场景下合理的默认值
	
	    // 以“上一次目标角”而非当前反馈角，计算本次计划位移
    float delta_deg       = fabsf(target_pos_deg - CyberGear_GetLastAngle(MOTOR_ID_ROTATION));
	    // 按档案与位移估算动态超时
    uint32_t dyn_timeout_ms = CyberGear_EstimateTimeoutMs(delta_deg, profile);
	
    MiAngleControl_MultiStage_V3(
				Motor,
        target_pos_deg,      // 1. 目标角度
        profile,             // 2. 传入的增益档案 (GAIN_SET_LOADED 或 GAIN_SET_UNLOADED)
        1,                   // 3. 启用超时
        dyn_timeout_ms,                // 4. 超时时间2.8秒
        1.5f,                // 5. 位置容忍度1.5度
        3.0f                 // 6. 速度容忍度3.0度/秒
    );

    // --- 阶段二: 切换到“位置模式”以保持位置 ---

    // 5. 停止电机 (这是切换模式前的强制要求) 
    stop_cybergear(Motor, 0); // 第二个参数0代表不清除错误
    HAL_Delay(20);

    // 6. 切换到位置模式
    set_mode_cybergear(Motor, Position_mode); // Position_mode 的枚举值为1
    HAL_Delay(20);

    // 7. 必须重新使能电机
    start_cybergear(Motor);
    HAL_Delay(20);

    // 8. 下达保持指令
    // 设定一个用于位置保持的速度限制
		float hold_speed_dps = 180.0f; // 例如，保持时用180度/秒的速度限制
		
		CyberGear_MoveTo_Sync_PosMode(
			Motor,           // 电机指针
			target_pos_deg,      // 目标角度 (度)
			hold_speed_dps,       // 保持速度 (度/秒)
			0.8,										// 允许误差范围（度）
			200,                    // 稳定持续时间 (200 毫秒)
			1,											// 启用超时
			1000   									// 超时 (毫秒)
			);
			//set_position_target_and_speed(Motor, target_pos_deg, hold_speed_dps); //直接使用位置模式，已由上方更高级的封装代替
}


/**
  * @brief  设置位置模式的目标位置和目标速度
  * @param  target_pos_deg: 目标位置 (单位: 度)
  * @param  target_speed_dps: 目标速度 (单位: 度/秒)
  */
void set_position_target_and_speed(MI_Motor *Motor, float target_pos_deg, float target_speed_dps)
{
    // 在这里将“度”和“度/秒”转换为电机需要的“弧度”和“弧度/秒”
    float target_pos_rad = target_pos_deg * 3.1415926f / 180.0f;
    float target_speed_rad_s = target_speed_dps * 3.1415926f / 180.0f;

    // 调用底层函数发送最终指令
    Set_Motor_Parameter(Motor, Loc_Ref, target_pos_rad, 'f');
    Set_Motor_Parameter(Motor, Limit_Spd, target_speed_rad_s, 'f');
}


/**
  * @brief          (位置模式下)移动到指定角度，等待指定时间后设置保持速度
  * @brief          此函数不依赖电机反馈，严格按照“指令-延时-指令”的流程执行。
  * @param[in]      Motor: 目标电机结构体指针
  * @param[in]      target_pos_deg: 目标角度 (单位: 度)
  * @param[in]      move_speed_dps: 运动过程中的速度限制 (单位: 度/秒,)
  * @param[in]      hold_speed_dps: 等待结束后，用于保持位置的速度限制 (单位: 度/秒,)
  * @param[in]      timeout_ms: 发送初始指令后需要等待的时间 (单位: 毫秒)，由外部传入
  */
void CyberGear_Move_Then_Hold(MI_Motor* Motor, float target_pos_deg, float move_speed_dps, 
                              float hold_speed_dps, uint32_t timeout_ms)
{
    // 1. 发送初始移动指令 (直接使用度/秒)
    set_position_target_and_speed(Motor, target_pos_deg, move_speed_dps);

    // 2. 执行延时
    if (timeout_ms > 0)
    {
        HAL_Delay(timeout_ms);
    }

    // 3. 发送最终保持指令 (直接使用度/秒)
    set_position_target_and_speed(Motor, target_pos_deg, hold_speed_dps);
}


/**
  * @brief          使用位置模式同步移动到指定角度
  * @brief          通过在循环中持续发送位置指令来轮询实时角度，实现阻塞式等待。
  * @param[in]      Motor: 目标电机结构体指针
  * @param[in]      target_pos_deg: 目标角度 (单位: 度)
  * @param[in]      move_speed_rps: 运动过程中的速度限制 (单位: 度/秒)
  * @param[in]      tolerance_deg: 可接受的误差范围 (单位: 度)
  * @param[in]      stable_ms: 判断稳定的持续时间 (单位: 毫秒)
  * @param[in]      timeout_ms: 最大等待时间/超时 (单位: 毫秒)
  * @retval         uint8_t: 1 表示成功到达并稳定, 0 表示超时
  */
uint8_t CyberGear_MoveTo_Sync_PosMode(MI_Motor* Motor, float target_pos_deg, float move_speed_dps, 
                                      float tolerance_deg, uint32_t stable_ms, 
                                      uint8_t enable_timeout, uint32_t timeout_ms)
{
    uint32_t start_time = HAL_GetTick();        // 记录函数开始时间，用于超时判断
    uint32_t stable_start_time = 0;             // 记录进入稳定区间的起始时间，0代表计时器未运行
    uint8_t return_status = 0;                  // 函数返回值，0=超时, 1=成功

    // 进入监控与控制的主循环
    while (1)
    {
				// 1. 超时检查
				if (enable_timeout == 1) // 仅当超时功能开启时，才进行检查
				{
						if (HAL_GetTick() - start_time > timeout_ms)
						{
								return_status = 0; // 标记为超时
								break; // 跳出循环
						}
				}

        // 2. 【核心】持续发送位置模式指令
        // 这一步既是下达运动指令，也是在主动“请求”电机回复其当前状态
        set_position_target_and_speed(Motor, target_pos_deg, move_speed_dps);
        
        // 3. 【关键】短暂延时
        // 给予一个非常短暂的延时，以确保STM32有足够的时间接收并由CAN中断处理电机回复的数据帧
        // 这个延时也决定了我们的轮询频率（例如10ms -> 100Hz）
        HAL_Delay(10); 
        
        // 4. 读取被CAN中断实时更新后的角度，并进行稳定判断
        uint8_t is_in_tolerance = (fabs(target_pos_deg - Motor->Angle) < tolerance_deg);
        if (is_in_tolerance)
        {
            // 当前在误差范围内，如果计时器还未启动，则启动它
            if (stable_start_time == 0)
            {
                stable_start_time = HAL_GetTick();
            }
        }
        else
        {
            // 只要有任何一刻离开了误差范围，就必须立刻重置计时器
            stable_start_time = 0;
        }

        // 检查计时器是否已经“跑完”（即持续稳定时间已满足）
        if (stable_start_time != 0 && (HAL_GetTick() - stable_start_time >= stable_ms))
        {
            return_status = 1; // 标记为成功
            break; // 达到稳定条件，跳出循环
        }
    }
    
    // 5. 跳出循环后，函数结束
    return return_status;
}

// --- MiAngleControl_MultiStage 使用的变量 ---
// 省赛前使用的带载参数，故弃用
/*
// --- 6节点5段动态增益调度参数 ---
// 6个控制节点的误差阈值 (单位: 度)。必须从大到小排列！
const float g_error_nodes_deg[6] = {150.0f, 80.0f, 60.0f, 20.0f, 5.0f, 0.0f};
// 6个节点对应的 Kp
const float g_kp_nodes[6] =        {0.5f, 1.0f, 1.5f, 6.0f, 13.5f, 5.0f};
// 6个节点对应的 Kd
const float g_kd_nodes[6] =        {0.8f, 0.8f, 0.8f, 1.8f, 4.0f, 5.0f};
// 模拟积分项参数,前面已定义
//const float g_ki = 0.1f;
//const float g_integral_active_error = 1.0f / 180.0 * 3.1415926;
//const float g_integral_max_accum = 5.0f;
*/

// --- MiAngleControl_Dynamic 使用的变量 ---
// 省赛前使用的空载参数，故弃用
/*
const float g_kp_min = 2.0f;
const float g_kp_max = 20.0f;
const float g_kd_min = 0.8f;
const float g_kd_max = 4.0f;
const float g_error_for_min_gain = 60.0 / 180.0 * 3.1415926;
const float g_error_for_max_gain = 5.0 / 180.0 * 3.1415926;
const float g_ki_2 = 0.1f;
const float g_integral_active_error_2 = 1.0 / 180.0 * 3.1415926;
const float g_integral_max_accum_2 = 1.0f;
*/

// 以下函数是省赛前用于空载旋转的函数，已由更细致的五段PID代替，弃用
/**
  * @brief  基于偏差的动态增益调度PD角度控制器（带模拟积分）
  * @param  target_angle_deg: 目标角度 (单位: 度)
  * @retval None
  */
/*


void MiAngleControl_Dynamic(float target_angle_deg)
{
    float target_rad = target_angle_deg / 180.0 * 3.1415926;
    float current_rad = 0.0;
    float error = 0.0;
    float error_abs = 0.0;
    
    float dynamic_kp = 0.0;
    float dynamic_kd = 0.0;
    
    static float error_accumulator = 0.0;
    
    error_accumulator = 0.0;

    do {
        current_rad = mi_motor[0].Angle / 180.0 * 3.1415926;
        error = target_rad - current_rad;
        error_abs = fabs(error);

        // 根据误差动态计算 Kp 和 Kd
        if (error_abs > g_error_for_min_gain) {
            dynamic_kp = g_kp_min;
            dynamic_kd = g_kd_min;
        } else if (error_abs < g_error_for_max_gain) {
            dynamic_kp = g_kp_max;
            dynamic_kd = g_kd_max;
        } else {
            float scale = (g_error_for_min_gain - error_abs) / (g_error_for_min_gain - g_error_for_max_gain);
            dynamic_kp = g_kp_min + (g_kp_max - g_kp_min) * scale;
            dynamic_kd = g_kd_min + (g_kd_max - g_kd_min) * scale;
        }

        // 计算模拟积分
        float dynamic_target_rad = target_rad;
        if (error_abs < g_integral_active_error_2) {
            error_accumulator += error;
            if (error_accumulator > g_integral_max_accum) error_accumulator = g_integral_max_accum_2;
            if (error_accumulator < -g_integral_max_accum) error_accumulator = -g_integral_max_accum_2;
            dynamic_target_rad += g_ki_2 * error_accumulator;
        }

        motor_controlmode(&mi_motor[0], 0, dynamic_target_rad, 0, dynamic_kp, dynamic_kd);
        
        HAL_Delay(2);

    } while (error_abs > (2/180.0*3.1415926) || fabs(mi_motor[0].Speed) > 0.05);
}
*/

// 以下函数是省赛前用于带载旋转的函数，已经升级为V3，故弃用
/**
  * @brief  基于6节点5段的精细化动态PD控制器（带模拟积分）
  * @param  target_angle_deg: 目标角度 (单位: 度)
  * @retval None
  */
/*
void MiAngleControl_MultiStage(float target_angle_deg)
{
    float target_rad = target_angle_deg / 180.0 * 3.1415926;
    float current_rad = 0.0;
    float error = 0.0;
    float error_abs = 0.0;
    
    float dynamic_kp = 0.0;
    float dynamic_kd = 0.0;
    
    static float error_accumulator = 0.0;
    static float rad_error_nodes[6];
    static uint8_t is_init = 0;

    // 首次运行时，将角度阈值转换为弧度，提高效率
    if (!is_init) {
        for (int i = 0; i < 6; ++i) {
            rad_error_nodes[i] = g_error_nodes_deg[i] / 180.0 * 3.1415926;
        }
        is_init = 1;
    }
    
    error_accumulator = 0.0;

    do {
        current_rad = mi_motor[0].Angle / 180.0 * 3.1415926;
        error = target_rad - current_rad;
        error_abs = fabs(error);

        // --- 根据误差判断所在区间并计算动态增益 ---
        if (error_abs > rad_error_nodes[0]) { // 大于最大误差阈值 (节点0)
            dynamic_kp = g_kp_nodes[0];
            dynamic_kd = g_kd_nodes[0];
        } else if (error_abs <= rad_error_nodes[4]) { // 小于最小误差阈值 (节点4到5之间)
             // 在最后两个节点之间插值，或者直接使用节点5的值
            if (error_abs < rad_error_nodes[5]) {
                 dynamic_kp = g_kp_nodes[5];
                 dynamic_kd = g_kd_nodes[5];
            } else {
                dynamic_kp = interpolate(error_abs, rad_error_nodes[4], rad_error_nodes[5], g_kp_nodes[4], g_kp_nodes[5]);
                dynamic_kd = interpolate(error_abs, rad_error_nodes[4], rad_error_nodes[5], g_kd_nodes[4], g_kd_nodes[5]);
            }
        }
        else { // 在中间的4个区间内查找并插值
            for (int i = 0; i < 4; ++i) {
                if (error_abs <= rad_error_nodes[i] && error_abs > rad_error_nodes[i+1]) {
                    dynamic_kp = interpolate(error_abs, rad_error_nodes[i], rad_error_nodes[i+1], g_kp_nodes[i], g_kp_nodes[i+1]);
                    dynamic_kd = interpolate(error_abs, rad_error_nodes[i], rad_error_nodes[i+1], g_kd_nodes[i], g_kd_nodes[i+1]);
                    break;
                }
            }
        }

        // --- 计算模拟积分项 (逻辑不变) ---
        float dynamic_target_rad = target_rad;
        if (error_abs < g_integral_active_error) {
            error_accumulator += error;
            if (error_accumulator > g_integral_max_accum) error_accumulator = g_integral_max_accum;
            if (error_accumulator < -g_integral_max_accum) error_accumulator = -g_integral_max_accum;
            dynamic_target_rad += g_ki * error_accumulator;
        }

        // --- 发送控制指令 ---
        motor_controlmode(&mi_motor[0], 0, dynamic_target_rad, 0, dynamic_kp, dynamic_kd);
        
        HAL_Delay(2);

    } while (error_abs > (2/180.0*3.1415926) || fabs(mi_motor[0].Speed) > 0.05);
}
*/



// --- 带梯形速度规划的同步移动函数 ---
// 这个思路在加速后仍然解决不了位置模式结束时的摆动问题，弃用
/**
  * @brief  使用梯形速度规划，同步移动到指定角度
  * @param  Motor: 目标电机结构体指针
  * @param  target_pos_deg: 目标角度 (单位: 度)
  * @param  cruise_speed_dps: 匀速阶段的速度 (单位: 度/秒)
  * @param  hold_speed_dps: 运动结束后用于保持位置的速度 (单位: 度/秒)
  * @param  accel_decel_distance_deg: 加速段和减速段的距离 (单位: 度)
  * @param  start_end_speed_dps: 起始和结束时的最低速度 (单位: 度/秒)
  * @param  tolerance_deg: 可接受的误差范围 (单位: 度)
  * @param  stable_ms: 判断稳定的持续时间 (单位: 毫秒)
  * @param  enable_timeout: 是否启用超时 (1:启用, 0:禁用)
  * @param  timeout_ms: 最大等待时间/超时 (单位: 毫秒)
  * @retval uint8_t: 1 表示成功到达并稳定, 0 表示超时
  */
/*
uint8_t CyberGear_MoveTo_Trapezoidal(MI_Motor* Motor, float target_pos_deg,
                                     float cruise_speed_dps, float hold_speed_dps,
                                     float accel_decel_distance_deg, float start_end_speed_dps,
                                     float tolerance_deg, uint32_t stable_ms,
                                     uint8_t enable_timeout, uint32_t timeout_ms)
{
    uint32_t start_time = HAL_GetTick();
    uint32_t stable_start_time = 0;
    uint8_t return_status = 0;

    // 1. 获取运动的起点
    float start_pos_deg = CyberGear_GetLastAngle();

    // 2. 进入监控与控制的主循环
    while (1)
    {
        // --- 超时与稳定判断逻辑 (与之前版本相同) ---
        if (enable_timeout == 1 && (HAL_GetTick() - start_time > timeout_ms))
        {
            return_status = 0; // 标记为超时
            break; 
        }

        float current_pos_deg = Motor->Angle; // 假设 Motor->Angle 已通过 volatile 保证实时性
        
        if (fabs(target_pos_deg - current_pos_deg) < tolerance_deg)
        {
            if (stable_start_time == 0) stable_start_time = HAL_GetTick();
        }
        else
        {
            stable_start_time = 0;
        }

        if (stable_start_time != 0 && (HAL_GetTick() - stable_start_time >= stable_ms))
        {
            return_status = 1; // 标记为成功
            break;
        }

        // --- 核心：梯形速度规划计算 ---
        float current_speed_dps = 0.0f;
        float dist_remaining = fabs(target_pos_deg - current_pos_deg); // 距目标的剩余距离
        float dist_traveled = fabs(current_pos_deg - start_pos_deg);   // 已走过的距离

        // 确保加减速距离不为负
        accel_decel_distance_deg = fabs(accel_decel_distance_deg);

        // a. 判断是否进入减速段
        if (dist_remaining <= accel_decel_distance_deg)
        {
            // 在减速段，速度根据剩余距离线性降低
            float scale = dist_remaining / accel_decel_distance_deg;
            current_speed_dps = start_end_speed_dps + (cruise_speed_dps - start_end_speed_dps) * scale;
        }
        // b. 判断是否处于加速段
        else if (dist_traveled < accel_decel_distance_deg)
        {
            // 在加速段，速度根据已走距离线性增加
            float scale = dist_traveled / accel_decel_distance_deg;
            current_speed_dps = start_end_speed_dps + (cruise_speed_dps - start_end_speed_dps) * scale;
        }
        // c. 否则，处于匀速段
        else
        {
            current_speed_dps = cruise_speed_dps;
        }
        
        // 安全钳：确保计算出的速度不会低于最低速
        if (current_speed_dps < start_end_speed_dps) {
            current_speed_dps = start_end_speed_dps;
        }

        // 3. 【关键】持续发送指令
        // 目标角度是固定的，但速度是根据规划动态变化的
        set_position_target_and_speed(Motor, target_pos_deg, current_speed_dps);

        HAL_Delay(5); // 设定控制频率为 ~200Hz
    }
    
    // 4. 跳出循环后，无论成功或超时，都发送一次最终的“保持速度”指令
    set_position_target_and_speed(Motor, target_pos_deg, hold_speed_dps);
    
    // 5. 更新内部状态
    if (return_status == 1) // 仅在成功到达后才更新角度记录
    {
        CyberGear_SetLastAngle(target_pos_deg);
    }
    
    return return_status;
}
*/

// --- 带可调加减速距离的梯形速度规划同步移动函数 V2 ---
// 这个思路在加速后仍然解决不了位置模式结束时的摆动问题，弃用
// 现在用于升降的小米电机了
/**
  * @brief  使用梯形速度规划，同步移动到指定角度，加减速距离独立可调
  * @param  accel_distance_deg: 加速段的距离 (单位: 度)
  * @param  decel_distance_deg: 减速段的距离 (单位: 度)
  * @param  (其余参数说明同旧版)
  * @retval uint8_t: 1 表示成功到达并稳定, 0 表示超时
  */
uint8_t CyberGear_MoveTo_Trapezoidal_V2(Motor_ID_t motor_id, MI_Motor* Motor, float target_pos_deg,
                                        float cruise_speed_dps, float hold_speed_dps,
                                        float accel_distance_deg, float decel_distance_deg,
                                        float start_end_speed_dps,
                                        float tolerance_deg, uint32_t stable_ms,
                                        uint8_t enable_timeout, uint32_t timeout_ms)
{
    uint32_t start_time = HAL_GetTick();
    uint32_t stable_start_time = 0;
    uint8_t return_status = 0;

    float start_pos_deg = CyberGear_GetLastAngle(motor_id);

    while (1)
    {
        // --- 超时与稳定判断逻辑 (保持不变) ---
        if (enable_timeout == 1 && (HAL_GetTick() - start_time > timeout_ms))
        {
            return_status = 0;
            break; 
        }

        float current_pos_deg = Motor->Angle;
        
        if (fabs(target_pos_deg - current_pos_deg) < tolerance_deg)
        {
            if (stable_start_time == 0) stable_start_time = HAL_GetTick();
        }
        else
        {
            stable_start_time = 0;
        }

        if (stable_start_time != 0 && (HAL_GetTick() - stable_start_time >= stable_ms))
        {
            return_status = 1;
            break;
        }

        // --- 核心：梯形速度规划计算 (已修改) ---
        float current_speed_dps = 0.0f;
        float dist_remaining = fabs(target_pos_deg - current_pos_deg);
        float dist_traveled = fabs(current_pos_deg - start_pos_deg);

        // 确保加减速距离为正数
        accel_distance_deg = fabs(accel_distance_deg);
        decel_distance_deg = fabs(decel_distance_deg);

        // a. 判断是否进入减速段 (使用新的 decel_distance_deg)
        if (dist_remaining <= decel_distance_deg)
        {
            float scale = (decel_distance_deg > 0) ? (dist_remaining / decel_distance_deg) : 0;
            current_speed_dps = start_end_speed_dps + (cruise_speed_dps - start_end_speed_dps) * scale;
        }
        // b. 判断是否处于加速段 (使用新的 accel_distance_deg)
        else if (dist_traveled < accel_distance_deg)
        {
            float scale = (accel_distance_deg > 0) ? (dist_traveled / accel_distance_deg) : 0;
            current_speed_dps = start_end_speed_dps + (cruise_speed_dps - start_end_speed_dps) * scale;
        }
        // c. 否则，处于匀速段
        else
        {
            current_speed_dps = cruise_speed_dps;
        }
        
        if (current_speed_dps < start_end_speed_dps) {
            current_speed_dps = start_end_speed_dps;
        }

        // 3. 持续发送指令
        set_position_target_and_speed(Motor, target_pos_deg, current_speed_dps);

        HAL_Delay(5);
    }
    
    // 4. 跳出循环后，发送最终的“保持速度”指令
    set_position_target_and_speed(Motor, target_pos_deg, hold_speed_dps);
    
    if (return_status == 1)
    {
        CyberGear_SetLastAngle(motor_id,target_pos_deg);
    }
    
    return return_status;
}
