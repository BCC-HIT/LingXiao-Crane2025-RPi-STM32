/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "can.h"
#include "dma.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stdlib.h"
#include "string.h"
#include "ctype.h"
#include "math.h"

#include "Delay_us.h"
#include "cybergear.h"
#include "CyberGear_Control.h"
#include "Stepper_and_Gripper.h"

#ifndef PI
#define PI 3.14159265358979323846264f
#endif

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
// 定义一个结构体来封装指令，包含指令类型和关联的数值
typedef struct {
    uint8_t type;   // 指令类型, 如 'f', 'g', 'p'
    double  value;  // 指令关联的数值
} Command_t;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
// --- 用于串口指令通信的全局变量 ---
volatile uint8_t receive_flag = '0'; // volatile关键字防止编译器过度优化
double ref = 0.0;                    // 用于存储指令附带的数值
const uint8_t ack = 0xAA;       // 向主控的反馈信息

// --- UART DMA接收相关 ---
// 建议将接收缓冲区大小宏定义，方便管理
#define RX_BUFFER_SIZE 10
uint8_t receiveData[RX_BUFFER_SIZE];

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
	
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_TIM1_Init();
  MX_TIM2_Init();
  MX_USART1_UART_Init();
  MX_CAN_Init();
  MX_TIM3_Init();
  /* USER CODE BEGIN 2 */

    /**************************************************************************
    * CAN总线初始化
    **************************************************************************/
    // CAN过滤器配置，此处配置为接收所有ID的报文
    CAN_FilterTypeDef can_filter_st;
    can_filter_st.FilterActivation = ENABLE;
    can_filter_st.FilterMode = CAN_FILTERMODE_IDMASK;
    can_filter_st.FilterScale = CAN_FILTERSCALE_32BIT;
    can_filter_st.FilterIdHigh = 0x0000;
    can_filter_st.FilterIdLow = 0x0000;
    can_filter_st.FilterMaskIdHigh = 0x0000;
    can_filter_st.FilterMaskIdLow = 0x0000;
    can_filter_st.FilterBank = 0;
    can_filter_st.FilterFIFOAssignment = CAN_RX_FIFO0;
    can_filter_st.SlaveStartFilterBank = 14;
    HAL_CAN_ConfigFilter(&hcan, &can_filter_st); // 应用过滤器配置
    HAL_CAN_Start(&hcan); // 启动CAN总线
    HAL_CAN_ActivateNotification(&hcan, CAN_IT_RX_FIFO0_MSG_PENDING); // 使能CAN接收FIFO0消息挂起中断

    /**************************************************************************
    * 定时器初始化
    **************************************************************************/
    // 启动定时器2，用于微秒级延时函数的基准
    HAL_TIM_Base_Start_IT(&htim2);

    // 启动定时器1和定时器3的基础功能
    HAL_TIM_Base_Start(&htim1); // 注意：此处原为HAL_TIM_Base_Init，Start更符合常规用法
    HAL_TIM_Base_Start(&htim3); // 注意：此处原为HAL_TIM_Base_Init，Start更符合常规用法
    
    // 启动定时器3的PWM功能，用于控制夹爪舵机
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);

    /**************************************************************************
    * CyberGear舵机 (旋转关节) 初始化
    **************************************************************************/
    HAL_Delay(2000); // 等待外设稳定
    
    // 初始化小米CyberGear舵机，使用0x7F作为CAN ID，并设置为位置模式
    //init_cybergear(&mi_motor[0], 0x7F, Position_mode); 
    // 下行为运控模式的初始化，可按需切换
    init_cybergear(&mi_motor[MOTOR_ID_ROTATION], ROTATION_MOTOR_ID, Motion_mode);
    HAL_Delay(10); // 短暂延时以确保设置生效
		init_cybergear(&mi_motor[MOTOR_ID_LIFT], LIFT_MOTOR_ID, Position_mode);
    HAL_Delay(10); // 短暂延时以确保设置生效
    
    // 将电机当前位置设置为机械零点
    set_zeropos_cybergear(&mi_motor[MOTOR_ID_ROTATION]);
	  set_zeropos_cybergear(&mi_motor[MOTOR_ID_LIFT]);
    HAL_Delay(10); // 短暂延时以确保设置生效

    /**************************************************************************
    * 步进电机 (升降关节) 初始化
    **************************************************************************/
    // 使能步进电机驱动器（通常为低电平有效）
//    HAL_GPIO_WritePin(GPIOB, ENABLE_Pin, GPIO_PIN_RESET);

    /**************************************************************************
    * UART串口初始化
    **************************************************************************/
    // 启动UART1的DMA空闲中断接收，用于接收指令
    HAL_UARTEx_ReceiveToIdle_DMA(&huart1, receiveData, 10);
    // 关闭DMA的半传输中断，只在接收完成或空闲时触发一次完整中断
    __HAL_DMA_DISABLE_IT(&hdma_usart1_rx, DMA_IT_HT);

    /**************************************************************************
    * 设备上电后状态设定
    **************************************************************************/
    // 重复发送保持零点指令，确保电机在上电后稳定在零位
    motor_controlmode(&mi_motor[MOTOR_ID_ROTATION], 0, 0, 0, 10.0, 4.5);
		set_position_target_and_speed(&mi_motor[MOTOR_ID_LIFT], 0, 180); 
    HAL_Delay(5);
    motor_controlmode(&mi_motor[MOTOR_ID_ROTATION], 0, 0, 0, 10.0, 4.5);
		set_position_target_and_speed(&mi_motor[MOTOR_ID_LIFT], 0, 180); 
    
    // 设置夹爪舵机初始状态为“闭合”
    __HAL_TIM_SetCompare(&htim3, TIM_CHANNEL_1, GRIPPER_CLOSE_PWM);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
	{
    // 检查是否有新指令需要处理
    if (receive_flag != '0')
    {
        // 使用 switch 结构处理指令
        switch (receive_flag)
        {
            case 's':
                // 调用步进电机的某个动作，例如复位或回到预设点
                // bujin(1, 45, 0); 
								Lift_MoveTo_cm(LIFT_MAX_HEIGHT_CM,0);
                break;

            case 'g':
                // 执行“取物”动作，使用全局变量 ref 作为参数
                get(ref);
                break;
                
            case 'p':
                // 执行“放置”动作，使用全局变量 ref 作为参数
                place(ref);
                break;
            
            case 'n': // 带载移动
                Move_and_Hold_Hybrid(&mi_motor[0], ref, GAIN_SET_LOADED);
//								MiAngleControl_MultiStage_V3(
//										&mi_motor[0],
//										ref,      // 1. 目标角度
//										GAIN_SET_LOADED,             // 2. 传入的增益档案 (GAIN_SET_LOADED 或 GAIN_SET_UNLOADED)
//										1,                   // 3. 启用超时
//										2800,                // 4. 超时时间
//										1.5f,                // 5. 位置容忍度1.5度
//										3.0f                 // 6. 速度容忍度3.0度/秒
//								);
									CyberGear_SetLastAngle(MOTOR_ID_ROTATION, ref);
                break;

            case 'f': // 空载移动
                Move_and_Hold_Hybrid(&mi_motor[0], ref, GAIN_SET_UNLOADED);
//								MiAngleControl_MultiStage_V3(
//										&mi_motor[0],
//										ref,      // 1. 目标角度
//										GAIN_SET_UNLOADED,             // 2. 传入的增益档案 (GAIN_SET_LOADED 或 GAIN_SET_UNLOADED)
//										1,                   // 3. 启用超时
//										1400,                // 4. 超时时间2.8秒
//										1.5f,                // 5. 位置容忍度1.5度
//										3.0f                 // 6. 速度容忍度3.0度/秒
//								);
									CyberGear_SetLastAngle(MOTOR_ID_ROTATION, ref);
                break;

            default:
                // 收到未知指令，不进行任何操作
                break;
        }

        // 发送统一的确认回执
        while(HAL_UART_Transmit(&huart1, (uint8_t*)&ack, 1, HAL_MAX_DELAY) != HAL_OK);
        
        // 在处理完一个指令后，立即将标志位清零，准备接收下一条指令
        receive_flag = '0';
    }
    
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
/**
  * @brief  USART1空闲中断回调函数，用于处理上位机指令
  * @param  huart: UART句柄指针
  * @param  Size: 本次接收到的数据长度
  * @retval None
  */
void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef* huart, uint16_t Size)
{
    // 确保中断来自于我们用于指令通信的USART1
    if (huart->Instance == USART1)
    {
        // 1. 安全处理：在数据末尾添加字符串结束符'\0'
        if (Size < RX_BUFFER_SIZE)
        {
            receiveData[Size] = '\0';
        }
        else
        {
            receiveData[9] = '\0'; // 强制在最后一位添加结束符
        }

        // 2. 解析指令
        char command = receiveData[0]; // 第一个字节是指令类型

        if (command == 's') // 对于不带参数的 's' 指令
        {
            receive_flag = 's';
        }
        else if (command == 'g' || command == 'p' || command == 'f' || command == 'n') // 对于需要参数的指令
        {
            if (Size > 1) // 确保有参数
            {
                ref = atof((const char *)&receiveData[1]); // 解析数值
                receive_flag = command; // 设置标志位
            }
        }
        // 未知指令将被忽略

        // 3. 重新启动DMA空闲中断接收
        HAL_UARTEx_ReceiveToIdle_DMA(&huart1, receiveData, RX_BUFFER_SIZE);
        __HAL_DMA_DISABLE_IT(&hdma_usart1_rx, DMA_IT_HT);
    }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart == &huart1) 
    {
        uint32_t err = huart->ErrorCode;

        // 清除所有接收相关错误标志
        if (err & HAL_UART_ERROR_ORE) __HAL_UART_CLEAR_OREFLAG(huart);
        if (err & HAL_UART_ERROR_FE)  __HAL_UART_CLEAR_FEFLAG(huart);
        if (err & HAL_UART_ERROR_PE)  __HAL_UART_CLEAR_PEFLAG(huart);
        if (err & HAL_UART_ERROR_NE)  __HAL_UART_CLEAR_NEFLAG(huart);

        // 重启接收
        HAL_UART_DMAStop(huart);
        HAL_UARTEx_ReceiveToIdle_DMA(&huart1, receiveData, RX_BUFFER_SIZE);
        __HAL_DMA_DISABLE_IT(&hdma_usart1_rx, DMA_IT_HT);
    }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
