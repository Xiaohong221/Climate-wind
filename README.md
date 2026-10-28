# 北大西洋气旋研究成果

本 `results` 分支为独立历史，仅保存研究报告、图表、结果表和复现脚本；原始数据保留在 `main` 分支。本分支不包含原始气象数据或大型 NumPy 缓存。

- [研究报告](results/研究报告.md)
- [概览图](results/overview.png)
- [移动统计](results/movement_summary.csv)
- [形成前后涡度](results/genesis_composite.csv)
- [结果及脚本 ZIP](results/climate-research-results.zip)

## 复现

只浏览报告无需原始数据。如需重新运行，需分别检出 results 与 main。建议先在本地克隆本分支，再在旁边准备数据目录：

```bash
git clone --single-branch --branch results https://github.com/Xiaohong221/Climate-wind.git Climate-wind-results
git clone --single-branch --branch main https://github.com/Xiaohong221/Climate-wind.git Climate-wind-data
git -C Climate-wind-data checkout --detach 9acb13732ebc42b2468062b61156c2f26ee2da56
git -C Climate-wind-data lfs pull
cd Climate-wind-results
python3 -m venv /tmp/climate-research-venv
/tmp/climate-research-venv/bin/pip install numpy==2.3.5 pandas==2.2.3 matplotlib==3.10.8
export CLIMATE_DATA_ROOT="$(cd ../Climate-wind-data && pwd)"
/tmp/climate-research-venv/bin/python results/analyze.py
/tmp/climate-research-venv/bin/python results/diagnostics.py
/tmp/climate-research-venv/bin/python results/write_report.py
```

原始数据版本：`9acb13732ebc42b2468062b61156c2f26ee2da56`。本次使用 Python 3.12.14 与上述依赖完成分析；独立机器上的克隆与安装命令为参考步骤。脚本会在 results 目录创建缓存，`.gitignore` 将其排除。更新数据后应移除旧缓存再运行。

首次分类为 TD/TS/HU 不等于首次达到飓风强度；本研究为回顾分析，不能当作已经验证的预测能力。请阅读报告中的方法与限制。
