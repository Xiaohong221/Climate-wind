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

## 新增研究：选题 A：时间分辨率与形成判识

[研究报告](results/studies/topic_a/研究报告.md) · [主代码](results/studies/topic_a/run.py) · [统计结果](results/studies/topic_a/metrics.csv) · [完整成果下载](results/studies/topic-a-results.zip)

训练2000—2015、验证2016—2018、测试2019—2022。完整方法、负结果与限制均保留在报告中。复现需 main 的完整固定版本数据，原始数据未修改。
