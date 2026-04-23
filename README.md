# AI4S 电力市场交易赛道项目

这个目录用于准备第四届世界科学智能大赛的“电力市场交易赛道（储能电站收益优化）”。

## 当前目标

当前阶段不是直接堆复杂模型，而是先做一套稳健 baseline 骨架，让团队可以快速进入：

1. 数据理解
2. 特征工程
3. 价格预测
4. 储能优化
5. 回测分析
6. 工程提交

## 先看这些文档

- [长期记忆](docs/长期记忆_电力市场交易赛道.md)
- [7天启动计划](docs/7天启动计划_电力市场交易赛道.md)
- [AI接力提示词](docs/AI接力提示词.md)

## 已创建的 baseline 骨架

代码目录：

- [pyproject.toml](pyproject.toml)
- [configs/baseline.toml](configs/baseline.toml)
- [src/ai4s_power_market](src/ai4s_power_market)

功能分层：

- `data.py`：数据读取与演示数据生成
- `features.py`：无泄漏时间序列特征
- `modeling.py`：价格预测 baseline
- `storage.py`：储能调度优化器
- `backtest.py`：滚动回测
- `cli.py`：命令行入口

## 快速开始

建议在项目目录下创建虚拟环境后执行：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
```

生成演示数据：

```powershell
.venv\Scripts\python -m ai4s_power_market generate-demo-data --config configs/baseline.toml
```

运行 baseline 回测：

```powershell
.venv\Scripts\python -m ai4s_power_market backtest --config configs/baseline.toml
```

运行后会在 `outputs/` 下生成结果文件。

## 当前时间基准

- 当前时间基准：2026-04-23
- 当前时区：Asia/Hong_Kong

## 重要提醒

- 公开信息已整理，但评分公式、数据字段、提交格式等细节，仍需报名后以官方文档为准
- 任何依赖具体字段的设计，都必须在官方数据字典确认后再锁定
