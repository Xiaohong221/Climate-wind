# NA_Hourly

仅维护北大西洋小时/三小时涡度及对应研究成果；main 保留完整数据备份。

- 原始数据：NA_5N45N_250E350E_Hourly/
- 辅助路径标签：data_hurricanes/ibtracs.NA.list.v04r01.csv（Git LFS）
- [研究报告](results/研究报告.md)
- [结果概览](results/overview.png)
- [移动统计](results/movement_summary.csv)
- [形成前后合成](results/genesis_composite.csv)

复现：确保 Git LFS 数据完整检出，依次运行 `python3 results/analyze.py`、`python3 results/diagnostics.py`。依赖和限制见报告。本分支不包含 HE 或日尺度气象网格数据。

[下载本分支研究成果 ZIP](results/research-results.zip)（仅结果和脚本，不含原始数据）
