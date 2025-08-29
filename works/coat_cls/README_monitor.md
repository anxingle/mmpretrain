# 训练监控工具使用指南

## 🎯 功能概述

这个监控工具可以实时监控训练过程，显示 accuracy、precision、recall 等指标，并将结果保存到 markdown 文件中。

## 🚀 快速开始

### 1. 前台运行（推荐用于测试）
```bash
# 基本用法
./start_monitor.sh

# 或者直接运行
python monitor_training.py --log /home/an/mmpretrain/train_cls.log --output recall_acc.md
```

### 2. 后台运行（推荐用于长期监控）
```bash
# 启动后台监控
./start_monitor_background.sh

# 查看实时结果
tail -f recall_acc.md

# 查看监控进程状态
ps aux | grep monitor_training

# 停止监控
pkill -f monitor_training
```

## 📋 参数说明

```bash
python monitor_training.py [选项]

选项:
  -h, --help           显示帮助信息
  --task {coat_color}  监控的任务类型
  --log LOG            日志文件路径
  --interval INTERVAL  刷新间隔（秒，默认10）
  --debug              启用调试模式
  --clear              清屏模式（默认是追加模式）
  --output OUTPUT      输出文件路径（如：recall_acc.md）
```

## 📊 输出内容

监控脚本会输出以下信息：

1. **基本信息**：
   - 时间戳
   - Epoch 信息
   - Top-1 Accuracy

2. **类别指标**：
   - Precision（精确率）
   - Recall（召回率）
   - F1-Score（F1分数）
   - Macro Average（宏平均）

3. **变化指示**：
   - 📈 准确率上升
   - 📉 准确率下降
   - ⚡ 类别指标变化

## 📄 文件输出

### recall_acc.md 文件格式：
```markdown
# 训练监控记录 - 舌苔颜色分类
开始时间: 2025-08-28 12:09:57
日志文件: /home/an/mmpretrain/train_cls.log

============================================================
🔍 训练监控 - 舌苔颜色分类
============================================================

⏰ [12:09:57] 🚀 首次数据:
🎯 Top-1 Accuracy: 93.19%
📈 类别指标:
------------------------------------------------------------
类别              Precision    Recall       F1-Score    
------------------------------------------------------------
灰色              0.3333       1.0000       0.5000      
白色              0.2750       0.3143       0.2933      
黄色              0.9684       0.9607       0.9645      
------------------------------------------------------------
Macro Average   0.5256       0.7583       0.6209      
------------------------------------------------------------
```

## 🔧 使用技巧

### 1. 实时查看结果
```bash
# 实时查看监控结果
watch -n 1 'tail -20 recall_acc.md'

# 或者使用 tail 命令
tail -f recall_acc.md
```

### 2. 自定义刷新间隔
```bash
# 每5秒刷新一次
python monitor_training.py --interval 5 --output recall_acc.md
```

### 3. 调试模式
```bash
# 启用调试模式
python monitor_training.py --debug --output recall_acc.md
```

### 4. 清屏模式
```bash
# 每次更新都清屏显示
python monitor_training.py --clear --output recall_acc.md
```

## 🛠️ 故障排除

### 1. 日志文件不存在
```bash
# 检查日志文件是否存在
ls -la /home/an/mmpretrain/train_cls.log

# 或者指定正确的日志文件路径
python monitor_training.py --log /path/to/your/log/file.log
```

### 2. 监控进程无法启动
```bash
# 检查进程状态
ps aux | grep monitor_training

# 查看错误日志
cat monitor.log

# 手动停止进程
pkill -f monitor_training
```

### 3. 输出文件权限问题
```bash
# 检查文件权限
ls -la recall_acc.md

# 修改权限
chmod 644 recall_acc.md
```

## 📈 监控指标说明

- **Top-1 Accuracy**: 整体分类准确率
- **Precision**: 精确率，预测为正类中实际为正类的比例
- **Recall**: 召回率，实际正类中被正确预测的比例
- **F1-Score**: 精确率和召回率的调和平均数
- **Macro Average**: 各类别指标的算术平均值

## 🎉 使用示例

完整的使用流程：

1. **启动训练**
```bash
bash train_coat_color_normal.sh
```

2. **启动监控**
```bash
./start_monitor_background.sh
```

3. **查看结果**
```bash
tail -f recall_acc.md
```

4. **停止监控**
```bash
pkill -f monitor_training
```
