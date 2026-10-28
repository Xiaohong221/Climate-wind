# HE_2000_2022

原始数据：`HE_2000_2022/`。

[数据质量报告](results/数据质量报告.md) · [矩阵检查统计](results/data_quality.csv) · [闰日核对](results/daily_date_audit.json)

本分支只维护该类数据和专属成果；`main` 保留所有数据，其他研究分支独立维护。

复现：`python3 results/audit_daily.py`。原日尺度质量核查保留；下方新增 HE–NA 协同研究，不等同于仅用 HE 进行预报。

[下载本分支研究成果 ZIP](results/research-results.zip)（仅结果和脚本，不含原始数据）

## 新增研究：选题 C：远程涡度与移动预测

[研究报告](results/studies/topic_c/研究报告.md) · [主代码](results/studies/topic_c/run.py) · [统计结果](results/studies/topic_c/metrics.csv) · [完整成果下载](results/studies/topic-c-results.zip)

训练2000—2015、验证2016—2018、测试2019—2022。完整方法、负结果与限制均保留在报告中。复现需 main 的完整固定版本数据，原始数据未修改。
