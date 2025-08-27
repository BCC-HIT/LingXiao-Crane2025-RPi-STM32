#!/bin/bash

# --- 请在这里配置你的Webhook地址 ---
WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/4668ae9b-c829-4b3b-ae4e-b803203a9a6d"

# --- 信息收集 ---

# 1. 触发原因 (可选，由调用者传入第一个参数)
TRIGGER_REASON=${1:-"定时报告"}

# 2. 本机时间
CURRENT_TIME=$(date "+%Y-%m-%d %H:%M:%S")

# 3. 运行时长
UPTIME=$(uptime -p | sed 's/up //')

# 4. CPU温度 (适用于树莓派)
# 如果 vcgencmd 不可用，可尝试 cat /sys/class/thermal/thermal_zone0/temp | awk '{printf "%.1f", $1/1000}'
CPU_TEMP=$(vcgencmd measure_temp | sed "s/temp=//")

# 5. CPU 占用率
# 计算方法：100 - CPU空闲率
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{printf "%.2f%%", 100 - $1}')

# 6. 内存占用率
MEM_INFO=$(free | grep Mem)
MEM_TOTAL=$(echo $MEM_INFO | awk '{print $2}')
MEM_USED=$(echo $MEM_INFO | awk '{print $3}')
MEM_USAGE=$(awk -v used="$MEM_USED" -v total="$MEM_TOTAL" 'BEGIN {printf "%.2f%%", (used/total)*100}')
MEM_DETAIL="$(echo $MEM_INFO | awk '{printf "%.2f/%.2f GB", $3/1024/1024, $2/1024/1024}')"

# 7. 网络信息
# 优先使用 nmcli，因为它能同时显示Wi-Fi和热点。如果nmcli不存在，则尝试iwgetid。
if command -v nmcli &> /dev/null; then
    WIFI_SSID=$(nmcli con show --active | grep -E 'wireless|wifi|802-11' | awk '{print $1}')
    HOTSPOT_NAME=$(nmcli con show --active | grep 'hotspot' | awk '{print $1}')
    NETWORK_NAME=${WIFI_SSID:-${HOTSPOT_NAME:-"未连接Wi-Fi或热点"}}
else
    NETWORK_NAME=$(iwgetid -r)
    [ -z "$NETWORK_NAME" ] && NETWORK_NAME="未连接Wi-Fi"
fi

# 8. IP 地址
# 使用 hostname -I 获取所有IPv4地址，取第一个
IPV4_ADDRESS=$(hostname -I | awk '{print $1}')
[ -z "$IPV4_ADDRESS" ] && IPV4_ADDRESS="N/A"
# 获取全局IPv6地址
IPV6_ADDRESS=$(ip -6 addr show scope global | grep 'inet6' | sed -n 's/.*inet6 \([0-9a-fA-F:/]*\).*/\1/p' | head -n 1)
[ -z "$IPV6_ADDRESS" ] && IPV6_ADDRESS="N/A"

# 9. vision.service 服务状态
# 使用 systemctl is-active 获取简短状态，如果不存在则提示
if systemctl list-units --type=service | grep -q "vision.service"; then
    SERVICE_STATUS_RAW=$(systemctl is-active vision.service)
    if [ "$SERVICE_STATUS_RAW" = "active" ]; then
        SERVICE_STATUS="✅ 运行中 (Active)"
    elif [ "$SERVICE_STATUS_RAW" = "inactive" ]; then
        SERVICE_STATUS="❌ 已停止 (Inactive)"
    elif [ "$SERVICE_STATUS_RAW" = "activating" ]; then
        SERVICE_STATUS="🔄 启动中 (Activating)"
    else
        SERVICE_STATUS="⚠️ 异常状态 ($SERVICE_STATUS_RAW)"
    fi
else
    SERVICE_STATUS="🤷‍♂️ 服务不存在 (Not Found)"
fi


# --- 构造发送到飞书的JSON ---
# 使用 "post" 消息类型以获得更丰富的格式
JSON_PAYLOAD=$(cat <<EOF
{
    "msg_type": "post",
    "content": {
        "post": {
            "zh_cn": {
                "title": "树莓派状态报告: ${TRIGGER_REASON}",
                "content": [
                    [
                        {"tag": "text", "text": "📅 本机时间: "},
                        {"tag": "text", "text": "${CURRENT_TIME}"}
                    ],
                    [
                        {"tag": "text", "text": "⏳ 运行时长: "},
                        {"tag": "text", "text": "${UPTIME}"}
                    ],
                    [
                        {"tag": "text", "text": "🌡️ 系统温度: "},
                        {"tag": "text", "text": "${CPU_TEMP}"}
                    ],
                    [
                        {"tag": "text", "text": "⚙️ CPU占用: "},
                        {"tag": "text", "text": "${CPU_USAGE}"}
                    ],
                    [
                        {"tag": "text", "text": "🧠 内存占用: "},
                        {"tag": "text", "text": "${MEM_USAGE} (${MEM_DETAIL})"}
                    ],
                    [
                        {"tag": "text", "text": "📡 网络名称: "},
                        {"tag": "text", "text": "${NETWORK_NAME}"}
                    ],
                    [
                        {"tag": "text", "text": "🌐 IPv4地址: "},
                        {"tag": "text", "text": "${IPV4_ADDRESS}"}
                    ],
                    [
                        {"tag": "text", "text": "🌐 IPv6地址: "},
                        {"tag": "text", "text": "${IPV6_ADDRESS}"}
                    ],
                    [
                        {"tag": "text", "text": "👁️ vision服务: "},
                        {"tag": "text", "text": "${SERVICE_STATUS}"}
                    ]
                ]
            }
        }
    }
}
EOF
)

# --- 发送请求 ---
curl -s -X POST -H "Content-Type: application/json" -d "${JSON_PAYLOAD}" "${WEBHOOK_URL}"

# 在终端打印日志（可选）
echo "报告已于 ${CURRENT_TIME} 发送，触发原因: ${TRIGGER_REASON}"