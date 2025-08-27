/*
 * receive_from_pi.c
 *
 *  Created on: Jun 28, 2025
 *  Modified on: Aug 17, 2025
 *  Version: 6
 *      Author: BCC
 */

#include "receive_from_pi.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* --- 模块内部私有变量 --- */
#define RX_BUFFER_SIZE 64
static UART_HandleTypeDef *uart_handle_polling;

/* --- 内部私有辅助函数 --- */

// 向树莓派发送信号字符串
static void rpi_send_signal(const char* signal_str) {
    if (uart_handle_polling != NULL && signal_str != NULL) {
        HAL_UART_Transmit(uart_handle_polling, (uint8_t*)signal_str, strlen(signal_str), 100);
    }
}

// 检查结果是否“完美”（即所有位置都有有效值，没有-1代表的'x'）
// [MODIFIED] Updated the logic for what constitutes a "perfect" result.
static bool is_result_perfect(const RecognitionResult_t* data) {
    // --- Shelf Positions Check ---
    // 必须包含所有从1-6的数字，且仅出现一次
    bool found_shelf_numbers[7] = {false}; // 索引0未使用，1-6用于数字1-6
    for (int i = 0; i < 6; i++) {
        int val = data->shelf_positions[i];
        if (val < 1 || val > 6) {
            // 必须是1到6之间的数字，同时也处理了-1 ('x') 的情况
            return false;
        }
        if (found_shelf_numbers[val]) {
            // 发现重复数字
            return false;
        }
        found_shelf_numbers[val] = true;
    }

    // --- Area Positions Check (New Logic) ---
    // 必须恰好包含一个'0'和五个不重复的数字(1-6)
    int zero_count = 0;
    int number_count = 0;
    bool found_area_numbers[7] = {false}; // 索引0未使用，1-6用于数字1-6

    for (int i = 0; i < 6; i++) {
        int val = data->area_positions[i];
        if (val == 0) {
            zero_count++;
        } else if (val >= 1 && val <= 6) {
            if (found_area_numbers[val]) {
                // 发现重复数字 (1-6)
                return false;
            }
            found_area_numbers[val] = true;
            number_count++;
        } else {
            // 发现无效值 (例如-1, 或 > 6)
            return false;
        }
    }

    // 检查完所有6个位置后，验证计数
    // 必须只有一个空位 ('0') 和5个物块
    if (zero_count != 1 || number_count != 5) {
        return false;
    }

    // 如果货架区和置物区的检查都通过，则结果是完美的
    return true;
}

/**
 * @brief 强制将数据修复/填充为一个完美的布局
 * @note 这是一个“保底”函数，用于超时发生时。
 * 它会尽可能保留原始数据中的有效部分，然后随机（确定性地）填充剩余部分。
 * @param data 指向要被就地修改的数据结构
 */
static void force_repair_to_perfect(RecognitionResult_t *data) {
    // --- 1. 修复货架区 (shelf_positions) ---
    bool shelf_num_used[7] = {false}; // 记录数字1-6是否已使用
    bool shelf_pos_filled[6] = {false}; // 记录位置0-5是否已填充

    // 第一遍：保留原始数据中有效且不重复的数字
    for (int i = 0; i < 6; i++) {
        int val = data->shelf_positions[i];
        if (val >= 1 && val <= 6 && !shelf_num_used[val]) {
            shelf_num_used[val] = true;
            shelf_pos_filled[i] = true;
        } else {
            // 对于无效或重复的数字，标记为未填充，稍后覆盖
            shelf_pos_filled[i] = false;
        }
    }

    // 第二遍：填充剩余的空位
    for (int i = 0; i < 6; i++) {
        if (!shelf_pos_filled[i]) {
            // 找到一个未使用的数字来填充
            for (int num_to_fill = 1; num_to_fill <= 6; num_to_fill++) {
                if (!shelf_num_used[num_to_fill]) {
                    data->shelf_positions[i] = num_to_fill;
                    shelf_num_used[num_to_fill] = true;
                    break; // 填充后跳出内层循环
                }
            }
        }
    }

    // --- 2. 修复置物区 (area_positions) ---
    bool area_num_used[7] = {false}; // 记录数字1-6是否已使用
    bool area_pos_filled[6] = {false}; // 记录位置0-5是否已填充
    bool zero_placed = false; // 是否已放置了0

    // 第一遍：保留原始数据中有效、不重复的数字和第一个0
    for (int i = 0; i < 6; i++) {
        int val = data->area_positions[i];
        if (val == 0 && !zero_placed) {
            zero_placed = true;
            area_pos_filled[i] = true;
        } else if (val >= 1 && val <= 6 && !area_num_used[val]) {
            area_num_used[val] = true;
            area_pos_filled[i] = true;
        } else {
            // 对于无效、重复的数字或多余的0，标记为未填充
            area_pos_filled[i] = false;
        }
    }

    // 第二遍：填充剩余的空位
    for (int i = 0; i < 6; i++) {
        if (!area_pos_filled[i]) {
            // 首先检查是否需要填充0
            if (!zero_placed) {
                data->area_positions[i] = 0;
                zero_placed = true;
                continue; // 填充了0，继续下一个位置
            }
            // 否则，用一个未使用的数字填充
            for (int num_to_fill = 1; num_to_fill <= 6; num_to_fill++) {
                if (!area_num_used[num_to_fill]) {
                    data->area_positions[i] = num_to_fill;
                    area_num_used[num_to_fill] = true;
                    break;
                }
            }
        }
    }
}

// 解析原始数据串，填充到out_result中
// 如果字符串结构上是12对键值，则返回true，否则false。'x'会被解析为-1。
static bool parse_string_to_result(const uint8_t* buffer, RecognitionResult_t* out_result) {
    memset(out_result, -1, sizeof(RecognitionResult_t));

    char local_buf[RX_BUFFER_SIZE];
    strncpy(local_buf, (char*)buffer, RX_BUFFER_SIZE - 1);
    local_buf[RX_BUFFER_SIZE - 1] = '\0';

    int parsed_pairs = 0;
    char *token = strtok(local_buf, ",");

    while (token != NULL) {
        char key_char, value_char;
        if (sscanf(token, "%c:%c", &key_char, &value_char) == 2) {
            int value = (value_char == 'x') ? -1 : (value_char - '0');
            if (key_char >= '1' && key_char <= '6') {
                out_result->shelf_positions[key_char - '1'] = value;
                parsed_pairs++;
            } else if (key_char >= 'a' && key_char <= 'f') {
                out_result->area_positions[key_char - 'a'] = value;
                parsed_pairs++;
            }
        }
        token = strtok(NULL, ",");
    }
    return (parsed_pairs == 12);
}

// 尝试修复货架区数据 (逻辑不变)
static bool attempt_repair_shelf(RecognitionResult_t *data) {
    int unknown_count = 0;
    int unknown_idx = -1;
    bool found_numbers[7] = {false};

    for (int i = 0; i < 6; i++) {
        if (data->shelf_positions[i] == -1) {
            unknown_count++;
            unknown_idx = i;
        } else if (data->shelf_positions[i] >= 1 && data->shelf_positions[i] <= 6) {
            if (found_numbers[data->shelf_positions[i]]) return false;
            found_numbers[data->shelf_positions[i]] = true;
        } else {
            return false;
        }
    }

    if (unknown_count == 1) {
        for (int num_to_find = 1; num_to_find <= 6; num_to_find++) {
            if (!found_numbers[num_to_find]) {
                data->shelf_positions[unknown_idx] = num_to_find;
                return true;
            }
        }
    }
    return false;
}

// 尝试修复纸垛区数据 (修正后的逻辑)
static bool attempt_repair_area(RecognitionResult_t *data) {
    int unknown_count = 0;
    int unknown_idx = -1;
    int zero_count = 0;
    bool found_numbers_1_to_6[7] = {false}; // 用于记录1-6数字是否出现
    int distinct_1_to_6_count = 0;

    for (int i = 0; i < 6; i++) {
        int val = data->area_positions[i];
        if (val == -1) {
            unknown_count++;
            unknown_idx = i;
        } else if (val == 0) {
            zero_count++;
        } else if (val >= 1 && val <= 6) {
            if (found_numbers_1_to_6[val]) return false; // 1-6数字重复，无法修复
            found_numbers_1_to_6[val] = true;
            distinct_1_to_6_count++;
        } else {
            return false; // 无效数据（非-1, 非0, 非1-6）
        }
    }

    if (zero_count > 1) return false; // 多于一个0，不符合规则

    if (unknown_count == 1) { // 只有一个未知项
        if (zero_count == 0) { // 情况1: 当前没有0，那么未知项必须是0
            if (distinct_1_to_6_count == 5) { // 并且其他5项是1-6中不重复的5个数字
                data->area_positions[unknown_idx] = 0;
                return true;
            }
        }
        // 情况2 (根据您的指正，这种情况无法唯一确定，故不进行修复):
        // 如果已有1个0，且有1个未知，其他4个是1-6中的不同数字。
        // 此时1-6中还有两个数字是候选，无法唯一确定未知项。
        // 因此，当zero_count == 1 时，不进行修复。
    }
    return false; // 其他情况均无法唯一确定修复
}


/* --- 公共接口函数实现 --- */

void rpi_init_polling(UART_HandleTypeDef *huart) {
    uart_handle_polling = huart;
}

// 简单阻塞轮询函数 (等待完美数据)
void rpi_receive_data_blocking_polling(RecognitionResult_t *result) {
    uint8_t rx_char_buffer[1];
    uint8_t line_buffer[RX_BUFFER_SIZE] = {0};
    uint8_t line_index = 0;

    while (1) {
        if (HAL_UART_Receive(uart_handle_polling, rx_char_buffer, 1, HAL_MAX_DELAY) == HAL_OK) {
            if (rx_char_buffer[0] == '\n') {
                line_buffer[line_index] = '\0';
                RecognitionResult_t temp_data;
                if (parse_string_to_result(line_buffer, &temp_data) && is_result_perfect(&temp_data)) {
                    memcpy(result, &temp_data, sizeof(RecognitionResult_t));
                    return;
                }
                line_index = 0;
                memset(line_buffer, 0, RX_BUFFER_SIZE);
            } else if (rx_char_buffer[0] == '\r') {
                // Ignore
            } else {
                if (line_index < RX_BUFFER_SIZE - 1) {
                    line_buffer[line_index++] = rx_char_buffer[0];
                } else {
                    line_index = 0;
                    memset(line_buffer, 0, RX_BUFFER_SIZE);
                }
            }
        }
    }
}

// 可配置健壮版接收函数 (已更新修复逻辑)
bool rpi_receive_data_configurable_polling(RecognitionResult_t *result,
                                           bool attempt_repair,
                                           uint32_t max_attempts_before_repair,
                                           bool enable_timeout,
                                           uint32_t timeout_ms) {
    uint8_t rx_char_buffer[1];
    uint8_t line_buffer[RX_BUFFER_SIZE] = {0};
    uint8_t line_index = 0;

    RecognitionResult_t last_structurally_valid_data;
    memset(&last_structurally_valid_data, -1, sizeof(RecognitionResult_t));
    bool any_structurally_valid_data_received = false;

    uint32_t attempt_count = 0;
    uint32_t start_tick = HAL_GetTick();

    while (1) {
        // 检查超时
        if (enable_timeout && (HAL_GetTick() - start_tick > timeout_ms)) {
            if (any_structurally_valid_data_received) {
                // 如果收到过数据，就以此为基础进行强制修复
                memcpy(result, &last_structurally_valid_data, sizeof(RecognitionResult_t));
                force_repair_to_perfect(result); // 调用新的强制修复函数
            } else {
                // 如果从未收到任何有效数据，则凭空创建一个完美的随机结果
                memset(result, -1, sizeof(RecognitionResult_t)); // 先清空确保状态一致
                force_repair_to_perfect(result); // 然后强制生成
            }
            return false; // 超时。注意：虽然返回false，但result中已是完美数据
        }

        // 尝试接收一个字符 (逻辑不变)
        if (HAL_UART_Receive(uart_handle_polling, rx_char_buffer, 1, 10) == HAL_OK) {
            if (rx_char_buffer[0] == '\n') { // 收到行结束符
                line_buffer[line_index] = '\0';

                RecognitionResult_t current_data;
                if (parse_string_to_result(line_buffer, &current_data)) { // 结构上解析成功
                    any_structurally_valid_data_received = true;
                    memcpy(&last_structurally_valid_data, &current_data, sizeof(RecognitionResult_t));
                    attempt_count++;

                    if (is_result_perfect(&current_data)) {
                        memcpy(result, &current_data, sizeof(RecognitionResult_t));
                        return true; // 收到完美数据
                    }

                    // --- 已更新的修复逻辑 ---
                    if (attempt_repair) {
                        RecognitionResult_t repaired_data;
                        memcpy(&repaired_data, &current_data, sizeof(RecognitionResult_t));
                        bool was_repaired = false;

                        // 1. 立即尝试修复置物区
                        if (attempt_repair_area(&repaired_data)) {
                            was_repaired = true;
                        }

                        // 2. 达到次数后再尝试修复货架区
                        if (attempt_count >= max_attempts_before_repair) {
                            if (attempt_repair_shelf(&repaired_data)) {
                                was_repaired = true;
                            }
                        }

                        // 3. 检查是否有任何修复使得数据变得完美
                        if (was_repaired && is_result_perfect(&repaired_data)) {
                            memcpy(result, &repaired_data, sizeof(RecognitionResult_t));
                            return true; // 修复后数据完美
                        }
                    }
                    // --- 修复逻辑结束 ---
                }

                line_index = 0;
                memset(line_buffer, 0, RX_BUFFER_SIZE);

            } else if (rx_char_buffer[0] != '\r') {
                if (line_index < RX_BUFFER_SIZE - 1) {
                    line_buffer[line_index++] = rx_char_buffer[0];
                } else {
                    line_index = 0;
                    memset(line_buffer, 0, RX_BUFFER_SIZE);
                }
            }
        }
    }
}

// (终极版) 函数开始时请求一次数据，并等待接收，同时支持数据修复和超时
bool rpi_request_and_receive_configurable_polling(
                                           RecognitionResult_t *result,
                                           const char* request_signal,
                                           bool attempt_repair,
                                           uint32_t max_attempts_before_repair,
                                           bool enable_timeout,
                                           uint32_t timeout_ms) {
    // --- 基础变量初始化 (保持不变) ---
    uint8_t rx_char_buffer[1];
    uint8_t line_buffer[RX_BUFFER_SIZE] = {0};
    uint8_t line_index = 0;

    RecognitionResult_t last_structurally_valid_data;
    memset(&last_structurally_valid_data, -1, sizeof(RecognitionResult_t));
    bool any_structurally_valid_data_received = false;

    uint32_t attempt_count = 0;
    uint32_t start_tick = HAL_GetTick();

    // --- 修改点：只在函数开始时发送一次请求 ---
    if (request_signal != NULL) {
        rpi_send_signal(request_signal); // 发送请求
    }
    // --- 移除了 last_request_tick 计时器 ---

    while (1) {
        // --- 移除了周期性重发请求的逻辑 ---

        // --- 以下为数据接收、解析、修复和超时的核心逻辑，保持不变 ---

        // 检查超时
        if (enable_timeout && (HAL_GetTick() - start_tick > timeout_ms)) {
            if (any_structurally_valid_data_received) {
                // 如果收到过数据，就以此为基础进行强制修复
                memcpy(result, &last_structurally_valid_data, sizeof(RecognitionResult_t));
                force_repair_to_perfect(result); // 调用新的强制修复函数
            } else {
                // 如果从未收到任何有效数据，则凭空创建一个完美的随机结果
                memset(result, -1, sizeof(RecognitionResult_t)); // 先清空确保状态一致
                force_repair_to_perfect(result); // 然后强制生成
            }
            return false; // 超时。注意：虽然返回false，但result中已是完美数据
        }

        // 尝试接收一个字符
        if (HAL_UART_Receive(uart_handle_polling, rx_char_buffer, 1, 10) == HAL_OK) {
            if (rx_char_buffer[0] == '\n') {
                line_buffer[line_index] = '\0';

                RecognitionResult_t current_data;
                if (parse_string_to_result(line_buffer, &current_data)) {
                    any_structurally_valid_data_received = true;
                    memcpy(&last_structurally_valid_data, &current_data, sizeof(RecognitionResult_t));
                    attempt_count++;

                    if (is_result_perfect(&current_data)) {
                        memcpy(result, &current_data, sizeof(RecognitionResult_t));
                        return true;
                    }

                    if (attempt_repair) {
                        RecognitionResult_t repaired_data;
                        memcpy(&repaired_data, &current_data, sizeof(RecognitionResult_t));
                        bool was_repaired = false;

                        if (attempt_repair_area(&repaired_data)) {
                            was_repaired = true;
                        }

                        if (attempt_count >= max_attempts_before_repair) {
                            if (attempt_repair_shelf(&repaired_data)) {
                                was_repaired = true;
                            }
                        }

                        if (was_repaired && is_result_perfect(&repaired_data)) {
                            memcpy(result, &repaired_data, sizeof(RecognitionResult_t));
                            return true;
                        }
                    }
                }

                line_index = 0;
                memset(line_buffer, 0, RX_BUFFER_SIZE);

            } else if (rx_char_buffer[0] != '\r') {
                if (line_index < RX_BUFFER_SIZE - 1) {
                    line_buffer[line_index++] = rx_char_buffer[0];
                } else {
                    line_index = 0;
                    memset(line_buffer, 0, RX_BUFFER_SIZE);
                }
            }
        }
    }
}

// 双信号请求版本：先发送预备信号，期间可接收数据，延迟后再发送主请求信号
bool rpi_dual_signal_request_and_receive_polling(
    RecognitionResult_t *result,
    const char* pre_signal,
    uint32_t pre_signal_delay_ms,
    const char* request_signal,
    bool attempt_repair,
    uint32_t max_attempts_before_repair,
    bool enable_timeout,
    uint32_t timeout_ms) {
    
    // --- 基础变量初始化 ---
    uint8_t rx_char_buffer[1];
    uint8_t line_buffer[RX_BUFFER_SIZE] = {0};
    uint8_t line_index = 0;

    RecognitionResult_t last_structurally_valid_data;
    memset(&last_structurally_valid_data, -1, sizeof(RecognitionResult_t));
    bool any_structurally_valid_data_received = false;

    uint32_t attempt_count = 0;
    uint32_t start_tick = HAL_GetTick();
    
    // --- 新增：记录是否已发送第二个信号 ---
    bool second_signal_sent = false;
    uint32_t pre_signal_sent_tick = 0;

    // --- 发送第一个预备信号（如果提供） ---
    if (pre_signal != NULL) {
        rpi_send_signal(pre_signal);
        pre_signal_sent_tick = HAL_GetTick();
    } else {
        // 如果没有预备信号，直接标记为已发送第二个信号
        second_signal_sent = true;
    }
    
    // --- 如果没有预备信号，直接发送主请求信号 ---
    if (second_signal_sent && request_signal != NULL) {
        rpi_send_signal(request_signal);
    }

    // --- 主接收循环：在整个过程中都可以接收和处理数据 ---
    while (1) {
        // --- 检查是否需要发送第二个信号 ---
        if (!second_signal_sent && 
            (HAL_GetTick() - pre_signal_sent_tick >= pre_signal_delay_ms)) {
            second_signal_sent = true;
            if (request_signal != NULL) {
                rpi_send_signal(request_signal);
            }
        }
        
        // --- 检查超时 ---
        if (enable_timeout && (HAL_GetTick() - start_tick > timeout_ms)) {
            if (any_structurally_valid_data_received) {
                memcpy(result, &last_structurally_valid_data, sizeof(RecognitionResult_t));
                force_repair_to_perfect(result);
            } else {
                memset(result, -1, sizeof(RecognitionResult_t));
                force_repair_to_perfect(result);
            }
            return false;
        }

        // --- 尝试接收一个字符（与原函数相同的接收逻辑） ---
        if (HAL_UART_Receive(uart_handle_polling, rx_char_buffer, 1, 10) == HAL_OK) {
            if (rx_char_buffer[0] == '\n') {
                line_buffer[line_index] = '\0';

                RecognitionResult_t current_data;
                if (parse_string_to_result(line_buffer, &current_data)) {
                    any_structurally_valid_data_received = true;
                    memcpy(&last_structurally_valid_data, &current_data, sizeof(RecognitionResult_t));
                    attempt_count++;

                    if (is_result_perfect(&current_data)) {
                        memcpy(result, &current_data, sizeof(RecognitionResult_t));
                        return true;  // 无论何时收到完美数据都立即返回
                    }

                    if (attempt_repair) {
                        RecognitionResult_t repaired_data;
                        memcpy(&repaired_data, &current_data, sizeof(RecognitionResult_t));
                        bool was_repaired = false;

                        if (attempt_repair_area(&repaired_data)) {
                            was_repaired = true;
                        }

                        if (attempt_count >= max_attempts_before_repair) {
                            if (attempt_repair_shelf(&repaired_data)) {
                                was_repaired = true;
                            }
                        }

                        if (was_repaired && is_result_perfect(&repaired_data)) {
                            memcpy(result, &repaired_data, sizeof(RecognitionResult_t));
                            return true;  // 修复成功也立即返回
                        }
                    }
                }

                line_index = 0;
                memset(line_buffer, 0, RX_BUFFER_SIZE);

            } else if (rx_char_buffer[0] != '\r') {
                if (line_index < RX_BUFFER_SIZE - 1) {
                    line_buffer[line_index++] = rx_char_buffer[0];
                } else {
                    line_index = 0;
                    memset(line_buffer, 0, RX_BUFFER_SIZE);
                }
            }
        }
    }
}
