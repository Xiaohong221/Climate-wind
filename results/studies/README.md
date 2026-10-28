# 热带气旋选题 A / C：可复现的回顾性实验

数据源固定为 main 的提交 `9acb13732ebc42b2468062b61156c2f26ee2da56`；原始气象文件未修改。主实验均使用 2000—2015 训练、2016—2018 验证、2019—2022 测试，未按测试成绩调整参数。计划保存于各自 analysis_plan.json，未进行外部预注册。

- A：[报告](topic_a/研究报告.md)，[主代码](topic_a/run.py)，[指标](topic_a/metrics.csv)。该部分发布在 NA_Hourly 分支。
- C：[报告](https://github.com/Xiaohong221/Climate-wind/blob/HE_2000_2022/results/studies/topic_c/研究报告.md)，[主代码](https://github.com/Xiaohong221/Climate-wind/blob/HE_2000_2022/results/studies/topic_c/run.py)，[指标](https://github.com/Xiaohong221/Climate-wind/blob/HE_2000_2022/results/studies/topic_c/metrics.csv)。该部分发布在 HE_2000_2022 分支。

各分支和各 ZIP 只包含自己对应的选题；不存在的另一选题文件请到另一分支查看。

## 依赖和输入

本次实际使用 Python 3.12.14。requirements.txt 记录已验证的依赖版本。需要完整 main 数据或具有相同目录结构的只读数据目录，Git LFS 必须完整检出：

- A：NA_5N45N_250E350E_Hourly（1H/3H）、NA 日均涡度、data_hurricanes 的 NA 路径。
- C：HE 日均涡度、NA 日均涡度和风速、NA 三小时涡度、data_hurricanes 的 NA 路径。

用 main 的完整检出最方便。为固定版本，请在数据检出目录执行 `git checkout --detach 9acb13732ebc42b2468062b61156c2f26ee2da56`，再执行 `git lfs pull`。也可按原目录名组合三个研究分支的输入。代码通过 CLIMATE_DATA_ROOT 指向该目录，读取原始数据，但不改写原始文件。

```bash
# 在 results/studies 或 ZIP 解压后的目录中执行
python3 -m venv /tmp/climate-study-venv
/tmp/climate-study-venv/bin/pip install -r requirements.txt
export CLIMATE_DATA_ROOT=/绝对路径/Climate-wind-main
# A 所在分支/下载包
/tmp/climate-study-venv/bin/python topic_a/run.py
/tmp/climate-study-venv/bin/python topic_a/robustness.py
# C 所在分支/下载包
/tmp/climate-study-venv/bin/python topic_c/run.py
# 对当前目录中存在的研究做检查、重新生成文字报告
/tmp/climate-study-venv/bin/python verify_results.py
/tmp/climate-study-venv/bin/python write_reports.py
```

上述命令分别只在含相应选题的目录运行；不要在只有 A 的包中执行 C。common.py 管理只读加载、缓存、球面坐标、日历与训练期异常处理。缓存 fingerprint 包含源文件路径、大小和修改时间；改变源文件会重新加载。cache/、A 的大特征缓存和目录缓存不在发布包中，运行时会重建；A 的 robustness.py 必须在主脚本生成 feature_cache.npz 后运行。

## 核心结果与边界

- A：72h AP 为小时抽样 0.1819、日抽样 0.1888、日均值 0.1818。成对区间未支持小时显著更好。日尺度模型使用同一个三小时候选目录，并不是独立日尺度检测系统。
- C：24h/48h 加远程场后误差约增加 5.24/21.97 km，验证与测试均不支持采用该扩展。只控制了现有涡度和标量风速，不是完整局地环境，也不是因果检验。

时间暂按 UTC 对齐，源变量单位与生产过程仍需来源核实。5°—7.5°块平均限制了小尺度研究。此前测试年份已经查看，后续新模型应与本次实验分开，使用新年份/海盆进行真正独立复核。

几何、闰日日历、训练期处理、分组分区及全部主要指标已通过 verify_results.py。具体漏报、虚警、样本量、敏感性及事后补充的范围请阅读报告，不能只引用点估计。
