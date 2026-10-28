from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/climate-study-mpl")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/climate-study-xdg")
import json,sys,platform
import numpy as np,pandas as pd,scipy,sklearn,matplotlib
B=Path(__file__).resolve().parent

def table(cols,rows):
 return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)

if (B/'topic_a/metrics.csv').exists():
 O=B/'topic_a';m=pd.read_csv(O/'metrics.csv');z=m[m.horizon.eq(72)];boot=pd.read_csv(O/'paired_bootstrap.csv');cat=json.loads((O/'catalog_summary.json').read_text());splits=pd.read_csv(O/'split_counts.csv');unc=pd.read_csv(O/'uncensored_sensitivity.csv');pairs=pd.read_csv(O/'matched_control_pairs.csv');md=pd.read_csv(O/'matched_control_diagnostics.csv');checks=json.loads((O/'cross_resolution_checks.json').read_text())
 tab=table(['特征方案','AP（0—1）','逐次召回率','逐次精确率','测试虚警率','检出风暴/84'],[(r.model,f'{r.average_precision:.4f}',f'{r.test_recall*100:.1f}%',f'{r.test_precision*100:.1f}%',f'{r.test_false_positive_rate*100:.2f}%',f'{r.detected_storms}/84') for r in z.itertuples()])
 stab=table(['划分','候选报次','候选轨迹','72h 正报次','72h 正风暴数'],[(r.split,r.samples,r.candidate_tracks,r.positive_72h,r.matched_event_storms) for r in splits.itertuples()])
 btab=table(['72h 对比','AP 差值','轨迹整组 bootstrap 95% 区间'],[(r.comparison,f'{r.AP_difference:.5f}',f'[{r.ci_low:.5f}, {r.ci_high:.5f}]') for r in boot.itertuples()])
 utab=table(['去除首条记录即为热带气旋的候选后','72h AP'],[(r.model,f'{r.average_precision:.4f}') for r in unc.itertuples()])
 text=f'''# 选题 A：时间分辨率与热带气旋形成判识

分析日期：2026-10-28。这是基于现有 5°块平均数据的回顾性基准实验，不是已验证的业务预报，也不保证可直接发表。主实验计划在运行前保存于 analysis_plan.json，未进行外部预注册。

## 核心结论

在同一个因果生成的三小时候选目录、相同测试样本和固定维度逻辑回归下，没有证据表明小时历史显著优于日尺度历史。72h AP：1h 抽样约 0.1819、24h 抽样约 0.1888、24h 均值约 0.1818。两项预定成对差值区间均包含 0。

这不证明小时数据普遍无用。结论限于当前 5°空间分辨率、候选识别/关联规则、特征摘要和简单模型。复杂结构被摘要成八个历史量，高频信息可能没有被充分利用。

![结果](topic_a_results.png)

## 数据与时间检查

输入为 2000—2022 年 NA 小时、三小时涡度，NA 日均涡度及 IBTrACS NA 主路径。三小时 67208×160，小时 201624×160。全时段逐点核实：小时每三小时抽样与三小时产品最大差为 {checks['1h_vs_3h_max_absolute_difference']:.2e}；小时聚合日均与原日均在有效位置的平均/最大绝对差为 {checks['daily_vs_hourly_derived_mean_absolute_difference']:.2e}/{checks['daily_vs_hourly_derived_max_absolute_difference']:.2e}。

原日均六个闰日整行缺失，但小时源的这些日期存在有限值。实验中的日尺度历史由小时源构造，因而保持完整公历；没有改写原日数据。时间暂按 UTC 解释，涡度单位及原始生产流程仍待来源确认。

## 候选识别、标签和对照

每个三小时场独立找 3×3 邻域正涡度极大值，阈值为文件数值 1e-5；其量级由训练期分布检查支持，并非本轮已确认的物理形成阈值。通过距上一时刻不超过 800km 的一对一最优匹配加幅度惩罚连续赋 ID；不使用未来位置、不桥接漏检。800km 是适应粗格点跳动的关联门限，不是物理移动速度判断。

目录共 {cat['all_catalog_observations']} 次峰值观测、{cat['all_candidate_tracks']} 条轨迹。其中很多只是粗网格背景涡度峰，不能把它们都当作独立确认的真实热带扰动。

每条完整 IBTrACS 主路径首次 TD/TS/HU 作为首次热带分类，先找全记录中的首次时刻，再限制 NA 研究区。标签不是首次达到飓风 HU。共 {cat['study_genesis_events']} 个事件，{cat['matched_to_catalog_at_genesis']} 个在该时刻与 800km 内候选一对一匹配。未来信息只用于结果标签，不进入识别或预测变量。未匹配候选及尚未在给定窗口形成的候选作为负样本；部分未匹配可能源于追踪失败，不能保证它们都是真实未形成扰动。

每天 00Z，对已持续至少 12 小时、具有过去 72 小时场的候选发报；排除当时距已记录热带/副热带气旋中心 800km 内的候选。跨训练/验证/测试边界的轨迹和跨界结果窗口被移除。

{stab}

训练 2000—2015，验证 2016—2018，测试 2019—2022。测试期真实首次分类事件共 84 个；其中 68 个有进入本候选/发报条件的正报次，因此目录与每日发报条件本身限制了全流程召回率。

## 时间特征和模型

所有方案共用三小时候选目录、当前峰值、周边空间统计、位置、季节和候选年龄。基础模型 11 个变量；其他模型统一 19 个变量。历史都取当前已知候选格点，绝不按未来形成位置提取。

抽样方案：过去 72h 至当前，按 1/3/6/12/24h 抽样。均值方案：对发报前完整的 72h 区间按 3/6/12/24h 分块平均，不使用发报后的当天均值。每个历史提取均值、极值、标准差、趋势、净变化、超阈值比例及持续时间。三小时原点处与小时一致。

固定 C=1 的标准化逻辑回归，不对类别重加权；填补与标准化仅在训练集拟合。阈值采用验证负样本概率 99 分位数、严格大于判断；测试虚警率不被人为固定为 1%。这不是各时间尺度分别从头识别候选的端到端比较；日尺度方案也享有共同三小时候选目录和当前状态，必须保留这一限定。

## 独立测试结果：72h

AP 是 average precision，不是准确率，不能解释为“18% 的预报正确”。本测试的正报次比例约 1.45%。

{tab}

{btab}

在 1h 方案对应的验证阈值下，逐次召回约 24.5%、精确率约 28.3%；全测试 84 个形成事件中检出 23 个。被检出事件的首次告警提前量中位数为 42h，不能将其表述为所有风暴都能提前 42h 预测。current_only 检出数甚至略高，不能把新增历史特征当作已验证收益。

24/48h 全部结果及各方案 Brier、ROC-AUC、虚警/漏报详见 metrics.csv。未按测试结果选择所谓最佳分辨率；12h 均值的较高点估计只能视为探索结果。

## 事后补充检查（与预定主结果分开）

首次路径记录已经是热带气旋时，实际形成可能更早。去掉这些关联候选的全部报次后，测试有 91 个正报次、38 个正事件，结果为：

{utab}

结果顺序随标签样本变化，仍不支持宣称小时历史必然优于日均历史。此补充没有用于改变主模型或选择新的参数。

另建立同分区、相距不超过 750km、季节日期差不超过 30天、当前幅度比 0.5—2、候选年龄比约 1/3—3 的对照；每个形成事件选距 48h 最近且提前 24—72h 的一次报次，匹配最多三个不同且不重复使用的候选轨迹。测试成功匹配 {pairs[pairs.split.eq('test')].SID.nunique()} 个事件。

对照必须是未被本目录关联到形成事件的轨迹，但这仍非独立卫星确认的物理未形成扰动。描述性比较中，形成组过去 72h 的净涡度变化较大；若干波动/脉冲指标也有差异。详见 matched_control_diagnostics.csv。六项区间未做多重比较校正，不能当作已确认的新机制，更不能替代独立预测成绩。

## 可复现与限制

运行方式见上级 README。run.py 同时生成用于敏感性检查的 feature_cache.npz；robustness.py 在其后运行。缓存、原始数据和大型中间文件不纳入成果 ZIP。所有原始数据只读。

关键限制：5°空间平均；仅涡度及轨迹信息；首次分类标签不确定；峰值关联可能合并/切换真实系统；每日一次发报；测试只有四年；bootstrap 以候选轨迹为组，未完全处理同一天气系统内不同候选之间的依赖。下一步应增加细网格多层风、湿度、海温、对流信息，验证真实扰动生命周期，并在新年份或独立海盆检验。
'''
 (O/'研究报告.md').write_text(text)

if (B/'topic_c/metrics.csv').exists():
 O=B/'topic_c';m=pd.read_csv(O/'metrics.csv');g=pd.read_csv(O/'incremental_gain.csv');s=pd.read_csv(O/'split_counts.csv');checks=json.loads((O/'checks.json').read_text())
 tab=table(['提前量','测试风暴/报次','持续运动误差 km','仅运动/气候误差 km','加局地场 km','再加远程 HE km'],[(h,f"{int(m[m.horizon.eq(h)].test_storms.iloc[0])}/{int(m[m.horizon.eq(h)].test_samples.iloc[0])}",*[f"{m[m.horizon.eq(h)&m.model.eq(name)].test_storm_mean_km.iloc[0]:.2f}" for name in ['persistence','motion','local','local_plus_remote']]) for h in [24,48]])
 gains=table(['提前量','局地误差−远程扩展误差 km','相对改善','按风暴 bootstrap 95% 区间','远程年度置换单侧 p'],[(r.horizon,f'{r.gain_km:.2f}',f'{r.relative_gain_pct:.2f}%',f'[{r.ci_low:.2f}, {r.ci_high:.2f}]',f'{r.permutation_p:.3f}') for r in g.itertuples()])
 splits=table(['提前量','划分','风暴','报次'],[(r.horizon,r.split,r.storms,r.samples) for r in s.itertuples()])
 text=f'''# 选题 C：远程 HE 涡度对气旋移动是否有增量信息？

分析日期：2026-10-28。问题限定为：在位置、季节、强度、近期运动，以及现有 NA 涡度和标量风速变量之外，加入 HE 远程涡度的低维表示，能否改善未来 24/48h 位置预测？主实验计划在运行前保存，未进行外部预注册。

## 核心结论

当前方案没有验证出正向增量，且添加远程特征后误差增大。验证集上扩展模型已经更差，因此若按验证结果选择是否增加信息，不应采用它。该结论仅适用于本数据、特征表示、参数搜索范围和线性模型，不能推导为“远程环流不影响气旋”。

![结果](topic_c_results.png)

## 数据与样本

原始版本 9acb13732ebc42b2468062b61156c2f26ee2da56。使用 HE 日均涡度、NA 日均涡度和风速、NA 三小时涡度、IBTrACS NA 主路径。文件时间暂按 UTC 对齐，变量单位和生产过程仍需来源说明。

发报时要求位于 5—45°N、250—350°E 且是 TD/TS/HU，过去至少 12h 及目标时刻有位置记录。预测目标允许风暴随后离开研究区或转变性质。每 6h 发报。由于 24/48h 位置资料可用性不同，两种提前量样本不完全相同；同一提前量所有模型使用完全相同的测试报次。

{splits}

训练2000—2015，验证2016—2018，测试2019—2022；跨划分风暴被剔除。最佳路径是事后分析资料，因此这是回顾性分析场实验，不能直接当作业务实时成绩。

## 局地控制与远程信息

运动基线包括位置及二次项、季节正余弦、当前 USA_WIND、过去 6/12h 移动和变化；持续运动使用最近 12h 位移。模型学习相对于持续运动的东/北向残差，最后还原为球面经纬度，使用大圆距离评价。

局地模型再加入：NA 当前及 6/12/24h 历史的 3×3 涡度块统计、NA 前完整日 1/3/5/7天的区域日均涡度 8个 PC、前一完整日的当地标量风速。没有多层 u/v、位势高度、海温或湿度，因此不声称已经控制完整的局地气象环境。标量风速不能代替引导风。

HE 288 个节点先排除扩展 NA 框（纬度 -2.5—52.5°、经度 242.5—357.5°）中的节点，留下 {checks['remote_HE_nodes']} 个节点。每次发报再将距当前中心不足1500km的节点异常置为零，即训练气候均值。此屏蔽基于当时已知位置。

HE 和 NA 日场均减去仅由训练年份拟合的31天平滑日历气候态，按训练异常标准差缩放。所有 PCA 仅拟合训练年份。HE 取前完整日 1/3/5/7天的4或8个 PC，并保留整日缺失标志和可用节点比例；没有使用发报之后的日平均。NA 获得相同的7天历史长度，避免仅靠增加历史窗口制造增量。

各模型使用按风暴等权的岭回归；alpha∈[1,10,100,1000]，HE PC数∈[4,8]。仅按验证集风暴等权平均距离选择参数。实际最优 alpha 均在该搜索范围上界，故不能把参数范围外或其他模型的可能表现视作已排除。

## 独立测试结果

以下主指标先对每个风暴的报次平均，再对风暴等权平均；不是把所有报次当作独立样本。

{tab}

{gains}

改善定义为“局地模型误差减去加入远程信息后的误差”，负数表示变差。24h 配对风暴 bootstrap 区间为负，48h 也为负。单纯位置/近期运动/季节模型甚至优于现有局地涡度扩展，说明不能因为新增数据就默认有可用收益。

四个测试年份逐年描述统计也均未出现正向远程增量，详见 descriptive_year_breakdown.csv。此逐年检查是主结果后的描述性补充，不用于重新选择模型。

## 负对照

对 HE 异常按完整日历年、在各训练/验证/测试分区内部独立置换，保持同月同日及年内天气演变，打乱同年同步关系。每次都重新在验证集选择 PC数和 alpha；每个提前量50次，共100次。模型本体、局地输入和测试标签不变。负对照会人为使用其他测试年份的场，严格只作为离线零假设实验，不是可实时运行的预测方法。

单侧 p 检验的是“真实远程关系的改善是否高于置换远程关系”，数值约0.98和1.00，未支持改善。只有四个测试年，bootstrap也仅按风暴聚类，不能把此统计当作完整的气候因果检验。

## 代码、解释边界与后续

run.py 包含球面目标、训练期气候态/PCA、局地控制、远程掩膜、验证选择、整年负对照和按风暴 bootstrap。preprocessing.npz 保存本次 PCA 参数；validation_search.csv 保留全部验证搜索；test_predictions.csv 保留所有测试预测，便于复核。

这里的“大尺度联系”通过远程场的滞后低维表示检验，没有把相关网络边当作因果传播，也没有识别出可信的遥相关机制。低维 PCA 最大化的是方差，可能忽略对气旋有用的弱信号；线性模型、仅176个远程粗格点及有限年数都限制结论。

下一步优先补充多层引导风和位势高度，建立更强的物理基线；在训练/验证期比较按物理区域划分、正则化监督降维或其他有明确约束的模型，再使用新的未看过年份测试。当前2019—2022成绩已查看，不应反复调整后仍声称它是全新的独立测试。任何新尝试必须与这次主实验分开报告。
'''
 (O/'研究报告.md').write_text(text)

versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__,'analysis_date':'2026-10-28','source_commit':'9acb13732ebc42b2468062b61156c2f26ee2da56'}
(B/'runtime_versions.json').write_text(json.dumps(versions,indent=2))
print('Reports generated for available studies.')
