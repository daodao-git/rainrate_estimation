# rainrate_estimation

降雨估计论文代码。

## Baseline 实验（新增）

新增 `baseline_experiments.py`，用于在相同训练/测试条件下评估原始 `RainFormerPhys` 与 4 个 baseline：

- `rainformer_phys`（原始 RainFormerPhys 模型）
- `cnn1d`（1D-CNN）
- `vanilla_transformer`（纯自注意力 Transformer）
- `lstm`
- `tcn`（长感受野时序卷积）

### 实验流程

1. 使用合成数据（`dataset_JW_Rreg_*`）在同一训练配置下训练 `RainFormerPhys` 和所有 baseline。
2. 在合成数据测试集上统计 `RMSE/MAE/R2`。
3. 将训练集统计量用于实测数据标准化。
4. 在实测数据上执行：
   - 直接预测（Raw）
   - LOOCV + Linear 校准
   - LOOCV + Ridge 校准
5. 导出各模型、各频率结果与总汇总。

### 运行方式

```bash
python baseline_experiments.py
```

### 输出目录

- 模型权重：`./baseline_models/<freq>/best_<model>.pth`
- 结果文件：`./baseline_results/<freq>/<model>/`
  - `predictions.csv`
  - `loocv_fold_details.csv`
  - `summary.csv`
- 全局汇总：
  - `./baseline_results/all_baselines_summary.csv`
  - `./baseline_results/all_baselines_summary.xlsx`

> 如需加快实验，可在 `baseline_experiments.py` 中调整 `num_epochs`、`FREQ_CONFIGS` 或 `BASELINE_MODEL_NAMES`。
> `RainFormerPhys` 已内置到 baseline 脚本中，复制/重命名脚本到其他目录运行时不再依赖 `from RainFormerPhys import RainFormerPhys`。
