#!/bin/bash

# 后台训练监控启动脚本
# 在后台运行监控并将结果输出到 recall_acc.md

echo "🚀 启动后台训练监控..."
echo "📄 输出文件: recall_acc_v2.md"
echo "⏰ 开始时间: $(date)"

# 查找最新的训练日志文件
LOG_FILE="/home/an/mmpretrain/train_coat_color_cls.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "⚠️  日志文件不存在: $LOG_FILE"
    echo "请先开始训练或指定正确的日志文件路径"
    exit 1
fi

echo "📋 监控日志文件: $LOG_FILE"
echo "🔍 开始在后台监控训练过程..."
echo "💡 使用 'tail -f recall_acc_v2.md' 查看实时结果"
echo "💡 使用 'ps aux | grep monitor_training' 查看进程状态"
echo "💡 使用 'pkill -f monitor_training' 停止监控"

# 在后台运行监控脚本
nohup python monitor_training.py \
    --log "$LOG_FILE" \
    --output recall_acc_v2.md \
    --interval 60 > monitor.log 2>&1 &

MONITOR_PID=$!
echo "✅ 监控已启动 (PID: $MONITOR_PID)"
echo "📄 结果将实时保存到: recall_acc.md"
echo "📋 监控日志: monitor.log"

# 显示进程信息
sleep 2
if ps -p $MONITOR_PID > /dev/null; then
    echo "✅ 监控进程运行正常"
else
    echo "❌ 监控进程启动失败，请检查 monitor.log"
fi
