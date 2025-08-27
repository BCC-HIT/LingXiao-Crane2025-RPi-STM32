/*
 * receive_from_pi.h
 *
 *  Created on: Jun 1, 2025
 *  Modified on: Aug 17, 2025
 *  Version: 6
 *      Author: BCC
 */

#ifndef INC_RECEIVE_FROM_PI_H_
#define INC_RECEIVE_FROM_PI_H_

#include "main.h"
#include <stdbool.h>

// 存储识别结果的结构体 (保持不变)
typedef struct {
    int shelf_positions[6];
    int area_positions[6];
} RecognitionResult_t;

// --- 公共接口函数 ---

/**
 * @brief 初始化接收模块 (轮询版)
 * @param huart: 指向您项目中配置的 UART4 句柄
 */
void rpi_init_polling(UART_HandleTypeDef *huart);

/**
 * @brief (简单阻塞轮询) 一直等待并接收一个【完美无误】的数据
 * @note  函数会在此处循环等待，直到接收到不含'x'的完整数据。
 * @param result: 指向一个结构体的指针，函数返回时，有效数据会填充到这里。
 */
void rpi_receive_data_blocking_polling(RecognitionResult_t *result);

/**
 * @brief (可配置的阻塞轮询) 等待接收数据，可选修复和超时机制
 * @param result: 指向一个结构体的指针，函数返回时，数据会填充到这里。
 * @param attempt_repair: 是否启用数据修复逻辑 (true/false)。
 * @param max_attempts_before_repair: 如果启用修复，在尝试修复前，接收和解析结构完整消息的最大次数。
 * (如果 attempt_repair 为 false，此参数被忽略)。
 * @param enable_timeout: 是否启用超时机制 (true/false)。
 * @param timeout_ms: 如果启用超时，整个接收过程的超时时间（毫秒）。
 * (如果 enable_timeout 为 false，此参数被忽略)。
 * @retval bool: 如果成功获取到完美数据（原始或修复后，或在超时前收到），返回true；
 * 如果超时且未获得完美数据，返回false（此时result中可能是最后一次收到的不完整数据或全-1）。
 */
bool rpi_receive_data_configurable_polling(RecognitionResult_t *result,
                                           bool attempt_repair,
                                           uint32_t max_attempts_before_repair,
                                           bool enable_timeout,
                                           uint32_t timeout_ms);
/**
 * @brief (终极版) 周期性发送请求信号，并等待接收数据，同时支持数据修复和超时。
 * @note  此函数是功能最全的接收函数，结合了主动请求、数据修复和超时控制。
 * @param result: (输出) 指向一个结构体的指针，用于存储最终结果。
 * @param request_signal: 您想要周期性发送的请求信号字符串, 例如 "GET_LAST_RESULT\n"。
 * @param request_interval_ms: 重新发送请求信号的时间间隔（毫秒）。
 * @param attempt_repair: 是否启用数据修复逻辑 (true/false)。
 * @param max_attempts_before_repair: 若启用修复，在尝试修复前，接收结构完整消息的最大次数。
 * @param enable_timeout: 是否启用超时机制 (true/false)。
 * @param timeout_ms: 若启用超时，整个接收过程的最大等待时间（毫秒）。
 * @retval bool: 如果在超时前成功获取到完美数据（原始或修复后），返回true；
 * 如果超时，返回false（此时result中可能是最后一次收到的不完整数据或全-1）。
 */
bool rpi_request_and_receive_configurable_polling(
                                           RecognitionResult_t *result,
                                           const char* request_signal,
                                           bool attempt_repair,
                                           uint32_t max_attempts_before_repair,
                                           bool enable_timeout,
                                           uint32_t timeout_ms);

/**
 * @brief (双信号版) 先发送预备信号，延迟后再发送主请求信号，并等待接收数据
 * @note  此函数在开始时先发送一个预备信号（如"PERFECT"），延迟后再发送主请求信号
 * @param result: (输出) 指向一个结构体的指针，用于存储最终结果
 * @param pre_signal: 预备信号字符串，例如 "PERFECT\n"（如为NULL则跳过）
 * @param pre_signal_delay_ms: 发送预备信号后的延迟时间（毫秒）
 * @param request_signal: 主请求信号字符串，例如 "REQUEST\n"
 * @param attempt_repair: 是否启用数据修复逻辑 (true/false)
 * @param max_attempts_before_repair: 若启用修复，在尝试修复前，接收结构完整消息的最大次数
 * @param enable_timeout: 是否启用超时机制 (true/false)
 * @param timeout_ms: 若启用超时，整个接收过程的最大等待时间（毫秒）
 * @retval bool: 如果在超时前成功获取到完美数据，返回true；如果超时，返回false
 */
bool rpi_dual_signal_request_and_receive_polling(
    RecognitionResult_t *result,
    const char* pre_signal,
    uint32_t pre_signal_delay_ms,
    const char* request_signal,
    bool attempt_repair,
    uint32_t max_attempts_before_repair,
    bool enable_timeout,
    uint32_t timeout_ms);

#endif /* INC_RECEIVE_FROM_PI_H_ */
