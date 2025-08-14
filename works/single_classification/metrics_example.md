# 评估指标说明

## 是的！配置已更新，会显示每个类别的precision/recall 📊

### 当前评估指标配置

配置文件中使用了两个评估器：

```python
val_evaluator = [
    dict(type='Accuracy', topk=(1,)),  # Top-1准确率
    dict(
        type='SingleLabelMetric',
        items=['precision', 'recall', 'f1-score'],
        average=None,  # None表示显示每个类别的指标
    )
]
```

### 训练过程中的输出示例

训练过程中，您会看到如下格式的评估指标：

```
Epoch [50/200]
--------------------------------------------------
验证结果：

Top-1 Accuracy: 92.35%

Class-wise Metrics:
┌──────────┬───────────┬──────────┬──────────┐
│ Class    │ Precision │ Recall   │ F1-Score │
├──────────┼───────────┼──────────┼──────────┤
│ teeth    │ 0.9156    │ 0.8965   │ 0.9060   │
│ normal   │ 0.9321    │ 0.9486   │ 0.9403   │
├──────────┼───────────┼──────────┼──────────┤
│ Macro Avg│ 0.9239    │ 0.9226   │ 0.9232   │
│ Weighted │ 0.9254    │ 0.9235   │ 0.9244   │
└──────────┴───────────┴──────────┴──────────┘

Confusion Matrix:
         Predicted
Actual   teeth  normal
teeth     883    117
normal     68   1432
```

### 指标解释

对于齿痕舌分类任务：

1. **Precision (精确率)**
   - `teeth类`: 预测为齿痕舌的样本中，真正是齿痕舌的比例
   - `normal类`: 预测为正常的样本中，真正是正常的比例

2. **Recall (召回率)**
   - `teeth类`: 所有齿痕舌样本中，被正确识别的比例
   - `normal类`: 所有正常样本中，被正确识别的比例

3. **F1-Score**
   - 精确率和召回率的调和平均数
   - 综合反映模型在该类别上的表现

4. **整体指标**
   - `Macro Average`: 各类别指标的简单平均（不考虑样本数）
   - `Weighted Average`: 各类别指标的加权平均（按样本数加权）

### 日志文件查看

训练完成后，可以在日志文件中查看详细的评估结果：

```bash
# 查看最新的评估结果
grep -A 20 "Class-wise Metrics" ../teeth_normal_classification/train.log | tail -30

# 查看某个epoch的详细结果
grep -A 20 "Epoch \[100/200\]" ../teeth_normal_classification/train.log
```

### 如何根据指标调优

1. **如果teeth类的recall低**：
   - 说明很多齿痕舌样本被误判为正常
   - 可以增加teeth的类别权重（您已经调整为2.0）

2. **如果teeth类的precision低**：
   - 说明很多正常样本被误判为齿痕舌
   - 可能需要更多的齿痕舌训练数据或调整数据增强

3. **如果两个类别指标差异大**：
   - 继续调整类别权重
   - 考虑使用focal loss代替cross entropy loss
   - 检查数据质量，是否存在标注错误

### 保存最佳模型

配置中设置了`save_best="auto"`，会自动保存validation accuracy最高的模型：

```bash
best_accuracy_top1_epoch_XX.pth  # 最佳准确率模型
last_checkpoint.pth              # 最新的检查点
```

您也可以根据特定类别的recall或precision来选择最佳模型。
