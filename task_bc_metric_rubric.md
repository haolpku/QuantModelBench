# Task B / Task C 指标与 Rubric 规范

## 0. 文档定位

本文档定义 Task B 和 Task C 的统一评估框架，服务于以下比较：

- numeric-only 与 semantic prompt；
- 不同 LLM、prompt、revision 策略；
- LLM 与固定 baseline；
- 高效果高成本与较低效果高效率的方案。

本规范采用三条原则：

1. **Outcome 与 Process 分开。** 模型的预测/收益结果不能因为成本高而被事后改写；成本、token、重试和代码质量单独评估。
2. **相对 baseline 的提升是独立方向。** 它回答“相对于已有方法增加了多少价值”，不替代绝对表现。
3. **LLM-as-Judge 只评价代码和研究过程。** 它不能覆盖真实 evaluator 的执行结果、数据泄漏审计或 rolling 失败。

主报告不以单一总分排序所有方法，而是同时报告 `Outcome Score`、`Process Score` 和成本-效果 Pareto 前沿。若确实需要一个摘要分数，必须同时公开各分项。

---

## 1. 评估单位与数据边界

### 1.1 评估单位

| 名称 | 定义 | 用途 |
|---|---|---|
| case | 一个固定数据、历史长度、任务配置和评估协议 | 主要 point estimate 单位 |
| origin | 一个 rolling 预测时点 | 计算逐期 Rank IC、收益、coverage |
| family | 同一问题/数据子集下不同历史长度或窗口的 case 集合 | bootstrap 和稳健性分组 |
| candidate | LLM 生成的一个完整代码版本 | 候选合法率、revision 和搜索效率 |
| frozen response | 最终被选择并进入 rolling 的代码版本 | Outcome 评估对象 |
| baseline | 预先冻结的确定性规则或模型 | uplift 参照 |

### 1.2 统计边界

- point estimate 延续当前实验的 **case-macro mean**，以便与已有报告一致。
- 统计推断使用 `family` 作为 block；同一 family 内的不同历史长度不能视为独立样本。
- rolling origin 不作为独立 case 重复计权；它们用于构造 case 指标和时间序列差值。
- 代码失败、不可交易月份和无效 origin 不填充为零收益。
- 所有阈值、baseline、归一化尺度和分数权重必须只用 dev 冻结，test 只做一次最终评估。

---

## 2. 总体结构

### 2.1 三个主轴

```text
Outcome
  ├─ Absolute Quality: 绝对预测/收益质量
  └─ Baseline Uplift: 相对预先冻结 baseline 的增益

Process
  ├─ Reliability: 能否正确完成协议
  ├─ Stability: 结果和行为是否稳定
  ├─ LLM-as-Judge: 生成代码的研究质量
  └─ Efficiency: token、金钱、时间和搜索成本

Decision layer
  └─ Outcome-Process Pareto frontier: 高效果高成本 vs 较低效果高效率
```

### 2.2 推荐权重

权重只用于生成摘要分，不用于替代原始指标。

**Outcome Score**：

\[
O = 0.60 Q_{abs} + 0.40 Q_{uplift}
\]

**Process Score**：

\[
P = 0.30 R + 0.15 S + 0.30 J + 0.25 E
\]

其中：

- `Q_abs`：Absolute Quality；
- `Q_uplift`：Baseline Uplift；
- `R`：Reliability；
- `S`：Stability；
- `J`：LLM-as-Judge 代码质量；
- `E`：Efficiency。

不得把 `O` 和 `P` 强行压成一个主排行榜。若需要平衡分，可在附表报告：

\[
H = \sqrt{O P}
\]

但 `H` 仅作为摘要，不得隐藏“效果高但成本高”和“效果略低但非常高效”的差异。

---

## 3. Outcome 轴

## 3.1 Absolute Quality

Absolute Quality 衡量模型本身是否产生有效预测和可交易结果，不依赖某一个 baseline。

### 3.1.1 AlphaBench 风格的三段式阈值

不直接复制 AlphaBench 的 `IC > 0.03`。对每个任务和指标 (m)，在 dev 阶段冻结：

\[
\tau_{noise,m},\quad
\tau_{effect,m},\quad
\tau_{target,m}
\]

含义如下：

| 阈值 | 含义 | 建议获得方式 |
|---|---|---|
| `noise floor` | 无信号情况下的上界 | block permutation 或 null bootstrap 的 95% 分位数 |
| `meaningful effect` | 最小实际有意义的效果 | 预注册的 MDE，不看 test |
| `target` | 达到 1 分的目标 | dev 冻结的强 baseline、业务目标或固定分位点 |

对“越大越好”的指标：

\[
Q_m =
\operatorname{clip}\left(
\frac{M_m-\tau_{noise,m}}
{\tau_{target,m}-\tau_{noise,m}},
0,1\right)
\]

对“越小越好”的指标，先转换为统一方向：

\[
M^{+}_m=-M_m
\]

或使用预先定义的改善量，再套用同一公式。

`tau_effect` 主要用于报告“是否达到实质有效”，不一定要作为连续分数的端点：

\[
\text{AboveEffectRate}_m
=
\frac{1}{N}\sum_i
\mathbf{1}(M_{im}\geq\tau_{effect,m})
\]

AlphaBench 的阈值是 benchmark-specific 的操作性门槛，不是普适显著性标准；因此本文把它借鉴为“noise floor -> meaningful threshold -> saturation target”的结构。

Task C 的 dev 冻结实现将全部滚动 baseline 与 privileged 的 `jkp_signed_equal_weight` 放入
`FROZEN_BASELINE_SET`；因此 `tau_effect` 取这些 baseline 的 case-macro 中位数，
`tau_target` 取其最大值，Active IR 也使用该冻结 rival 集合校准。由于 JKP 使用了
prompt 不提供的官方因子方向，它不能作为 LLM 的同信息 uplift 配对基线。

### 3.1.2 Task B Absolute Quality

Task B 的主指标：

| 指标 | 方向 | 角色 |
|---|---:|---|
| rolling Rank IC | 高 | 核心预测质量 |
| Top-3 hit rate | 高 | 路由质量 |
| Directional accuracy | 高 | 辅助预测诊断 |
| 净年化收益 | 高 | 经济结果 |
| Sharpe | 高 | 风险调整结果 |
| MDD | 低（绝对回撤越小越好） | 风险约束 |

建议：

\[
Q_{abs}^{B} =
0.35Q_{RankIC}
+0.20Q_{Top3}
+0.10Q_{Direction}
+0.25Q_{Return}
+0.10Q_{Sharpe/MDD}
\]

`Top-3 hit rate` 应与随机选取或等权控制的提升相比，而不是只看绝对值。

### 3.1.3 Task C Absolute Quality

Task C 的主指标：

| 指标 | 方向 | 角色 |
|---|---:|---|
| rolling OOS Rank IC | 高 | 核心预测质量 |
| 净年化收益 | 高 | 经济结果 |
| Sharpe | 高 | 风险调整结果 |
| MDD | 低（绝对回撤越小越好） | 风险约束 |
| coverage | 高 | 可交易性/完整性 |

建议：

\[
Q_{abs}^{C} =
0.40Q_{RankIC}
+0.25Q_{Return}
+0.20Q_{Sharpe}
+0.10Q_{MDD}
+0.05Q_{Coverage}
\]

Task C 的 MSE 只作诊断，不进入主分数。score 可以进行单调变换，MSE 的量纲和横向可比性不足。

### 3.1.4 聚合与不确定性

- 每个 case 先得到完整指标，再做 case-macro mean。
- 作为稳健性分析，报告 median 和 IQM。
- 95% CI 使用按 family 重采样的 block bootstrap。
- 对逐月预测损失差，可补充 Diebold-Mariano 风格的时间相关性稳健检验。
- 只有在 family 数量足够且依赖结构可接受时，才使用 Wilcoxon 或 sign test。

---

## 3.2 Baseline Uplift

Baseline Uplift 衡量“模型相对于一个预先冻结方法增加了多少价值”。

### 3.2.1 Baseline 选择

| 任务 | Primary baseline | Secondary baseline | 特殊说明 |
|---|---|---|---|
| Task B | `momentum_12` | `t_stat_12` | 都使用相同 evaluator 和成本口径 |
| Task C | `rolling_ridge` | `rolling_mlp_ensemble` | uplift 只用这两个同信息 baseline；`jkp_signed_equal_weight` 是 privileged reference，但仍进入 dev 冻结 baseline set，用于 `tau_effect`/`tau_target`（以及 Active IR）校准 |

不允许在 test 上逐 case 选择最优 baseline。若需要展示 oracle baseline，只能作为“regret to toolkit”诊断，不得进入主分数。

### 3.2.2 逐 case 增益

先统一指标方向：

\[
\Delta_{im}=s_m(M_{im}-B_{im})
\]

其中高者越好时 (s_m=1)，低者越好时 (s_m=-1)。

典型指标：

- `Delta Rank IC = RankIC_model - RankIC_baseline`；
- `Delta return = return_model - return_baseline`，单位为百分点；
- `Delta Sharpe = Sharpe_model - Sharpe_baseline`；
- `Delta MDD = abs(MDD_baseline) - abs(MDD_model)`；
- `Delta coverage = coverage_model - coverage_baseline`。

不要把 `(M-B)/abs(B)` 作为主指标。baseline 接近 0 或为负时，这个比例会失真。

### 3.2.3 增益的三个互补指标

**增益幅度：**

\[
U_{magnitude,m}
=operatorname{IQM}_i
\left[
\operatorname{clip}\left(\frac{\Delta_{im}}{a_m},-1,1\right)
\right]
\]

其中 (a_m) 是 dev 冻结的实际意义尺度。

**提升概率：**

\[
ImprovedRate_m
=
\frac{1}{N}\sum_i
\mathbf{1}(\Delta_{im}>\epsilon_m)
\]

其中 (epsilon_m) 是 practical equivalence margin。

**主动收益和 Information Ratio：**

\[
a_t=r^{model}_t-r^{baseline}_t
\]

\[
ActiveIR=\sqrt{12}\frac{\operatorname{mean}(a_t)}
{\operatorname{std}(a_t)}
\]

### 3.2.4 Uplift 分数

\[
Q_{uplift}=
0.50Q_{U,magnitude}
+0.30Q_{ImprovedRate}
+0.20Q_{ActiveIR}
\]

`ImprovedRatio` 可以作为结果表中的补充指标，但不能代替提升概率；概率反映“经常赢”，幅度反映“赢多少”。

### 3.2.5 Semantic treatment effect

semantic 与 numeric-only 是输入处理的 treatment comparison，应单独计算：

\[
\Delta_{semantic}=M_{semantic}-M_{numeric}
\]

它不能与“模型相对传统 baseline 的 uplift”混成一个分数。

---

## 4. Process 轴

## 4.1 Reliability

Reliability 衡量是否严格完成实验协议：

\[
R_{case}=
\frac{\text{严格完成 case 数}}{\text{总 case 数}}
\]

\[
R_{origin}=
\frac{\text{有效 rolling origin 数}}{\text{应完成 origin 数}}
\]

\[
R=0.60R_{case}+0.40R_{origin}
\]

严格完成要求：

- schema 正确；
- 代码可执行；
- 没有 future leakage；
- 所有必需输出存在；
- rolling evaluator 正常结束；
- 预测和交易 coverage 符合协议。

运行失败不得填充成零收益；它应在 `R`、coverage 和 failure taxonomy 中体现。

## 4.2 Stability

建议报告：

- rolling 指标的标准差和 IQR；
- 不同 case family 的性能波动；
- revision 之间的候选选择一致性；
- 多次运行的结果一致性；
- 失败类型是否集中在某一类数据或历史长度。

可定义：

\[
S=1-
\operatorname{clip}\left(
\frac{IQR(\text{origin metric})}{a_S},0,1\right)
\]

其中 (a_S) 在 dev 阶段冻结。

## 4.3 LLM-as-Judge：生成代码质量

### 4.3.1 定位

LLM-as-Judge 评价的是：代码是否体现了可靠的量化研究过程。它不评价最终收益，也不能覆盖真实执行审计。

Judge 输入：

- task contract 和 schema；
- prompt 要求；
- 生成的完整代码；
- 必要的 metadata 和静态检查结果。

Judge 不应看到：

- test outcome；
- baseline 结果；
- 目标模型的排名；
- 任何暗示“这个答案应该更好”的标签。

### 4.3.2 Judge 维度

每项 0-4 分，先逐维度打分，再取平均：

| 维度 | 4 分标准 | 0 分标准 |
|---|---|---|
| Contract adherence | 完全满足输入输出、接口和评估协议 | 无法解析或违反核心接口 |
| Point-in-time / leakage | 训练、验证、预测边界清楚，无未来数据 | 使用未来标签、未来统计量或隐藏信息 |
| Statistical soundness | 预处理、拟合、特征选择均在可见数据内完成 | 明显的数据泄漏或错误拟合边界 |
| Numerical robustness | 正确处理 NaN、空表、常数列、极端值和奇异矩阵 | 易因常见边界条件崩溃 |
| Generalization | 不依赖固定列数、固定 case 或隐藏数据形状 | 大量 hard-code，只能通过单一样例 |
| Code clarity / complexity | 结构清楚、复杂度合理、无无效重复计算 | 代码不可维护或存在明显无效复杂度 |
| Reproducibility | 随机性、排序、状态和副作用处理清楚 | 结果不可复现或有隐式外部状态 |

\[
J_{raw}=\frac{1}{7}\sum_{d=1}^{7}J_d
\]

归一化到 `[0,1]`：

\[
J=J_{raw}/4
\]

### 4.3.3 Judge 的硬性约束

以下问题不能被 LLM-as-Judge 的高分抵消：

- future leakage：Reliability 直接失败；
- evaluator 真实执行失败：该 case 不进入 outcome 有效样本；
- schema/接口失败：candidate 无效；
- 使用禁用外部数据或隐藏标签：case 标记为 protocol violation；
- Judge 认为“代码看起来正确”但实际 audit 失败：以 audit 为准。

### 4.3.4 Judge 可靠性

- 至少使用两次独立 judge pass，或两个独立 judge 配置；
- 报告均值、标准差和 exact agreement / weighted agreement；
- 对高分和低分边界样本做人工抽查；
- Judge prompt、模型版本、temperature、输入快照和输出必须保存；
- Judge 不得使用最终 outcome 反推代码质量；
- Judge 分数只进入 `Process Score`，不得进入 Rank IC、return 或 Sharpe。

## 4.4 Efficiency

Efficiency 专门衡量“达到结果所付出的代价”，不通过扣减 outcome 来表达。

建议记录：

| 指标 | 含义 |
|---|---|
| token cost | prompt、completion 和总 token |
| monetary cost | API 实际费用 |
| wall-clock time | 单 case 和全批次耗时 |
| retry count | 得到合法 candidate 所需尝试数 |
| revision count | 生成和修订次数 |
| valid candidate rate | 合法 candidate / 总 candidate |
| selection fallback rate | host fallback 或无有效选择比例 |
| search diversity | 候选结构和行为多样性 |

对成本使用对数尺度，避免 10 倍成本被线性尺度压扁：

\[
E_{cost}=
\operatorname{clip}\left(
\frac{\log c_{bad}-\log c}
{\log c_{bad}-\log c_{good}},0,1\right)
\]

其中 (c_{good}) 和 (c_{bad}) 在 dev 阶段冻结。

建议：

\[
E=0.45E_{cost}
+0.25E_{valid}
+0.15E_{retry}
+0.15E_{diversity}
\]

---

## 5. 如何区分“效果好但很贵”和“效果略低但很高效”

### 5.1 主决策：二维 Pareto 前沿

每个方法报告一个点：

\[
(O,P)=(Outcome\ Score, Process\ Score)
\]

若方法 A 的 outcome 更高但 process 更低，方法 B 的 outcome 略低但 process 更高，则二者是 Pareto-incomparable，不应人为宣布一个绝对赢家。

### 5.2 三类推荐方案

**Quality champion**

\[
\arg\max O
\quad\text{s.t.}\quad P\geq P_{min}
\]

适合追求最高研究质量的场景。

**Efficiency champion**

\[
\arg\max P
\quad\text{s.t.}\quad O\geq O_{min}
\]

适合大规模 factor search 或预算受限的场景。

**Balanced choice**

\[
\arg\max \sqrt{O P}
\]

只作为明确指定预算和质量下限后的辅助选择。

### 5.3 成本效果表

最终报告至少包含：

| 方法 | Outcome | Process | API cost | valid case rate | Quality champion | Efficiency champion | Pareto |
|---|---:|---:|---:|---:|---|---|---|
| model/prompt A |  |  |  |  |  |  |  |
| model/prompt B |  |  |  |  |  |  |  |

另报：

- `Outcome / $1`；
- `Outcome / 1k tokens`；
- 达到 `O_min` 所需的中位成本；
- 达到 `tau_effect` 的成功率；
- active uplift per dollar。

这样可以清楚回答：

- 高成本方案是否真的带来了额外 outcome；
- 低成本方案是否已经达到可接受质量；
- 额外 10 倍成本换来的提升是否值得。

---

## 6. 最终摘要分与分档

所有一级分数先归一化到 `[0,100]`：

| 分数 | 解释 |
|---:|---|
| 80-100 | Outcome 或 Process 显著强，且在主要稳健性检查下成立 |
| 60-80 | 有明确价值，但存在成本、稳定性或风险短板 |
| 40-60 | 基本可运行，能力接近 baseline 或证据不充分 |
| 0-40 | 大量失败、协议不合规，或没有可重复的有效信号 |

推荐报告格式：

```text
Method: <model / prompt / track>

Outcome
  Absolute Quality:  xx/100
  Baseline Uplift:   xx/100
  Outcome Score:     xx/100

Process
  Reliability:       xx/100
  Stability:         xx/100
  LLM-as-Judge:       xx/100
  Efficiency:         xx/100
  Process Score:      xx/100

Cost
  API cost:           $x
  Tokens:             x
  Wall-clock:         x

Evidence
  Valid cases/origins: x/y
  ImprovedRate:        x%
  Active IR:           x
  95% block-bootstrap CI: [..., ...]
  Pareto status:       quality / efficiency / balanced / dominated
```

---

## 7. 必须冻结的项目

在读取 test outcome 前，必须冻结：

1. primary 和 secondary baseline；
2. `tau_noise`、`tau_effect`、`tau_target`；
3. uplift 的 practical equivalence margin `epsilon`；
4. `a_m`、`c_good`、`c_bad` 等尺度参数；
5. Outcome 和 Process 的权重；
6. case/family 聚合方式；
7. Judge prompt、judge model、评分维度和硬性审计规则；
8. Quality champion、Efficiency champion 的门槛；
9. 是否进行多 revision、多 baseline 或多 prompt 搜索；
10. 多重比较和 data-snooping 处理方案。

不得在 test 结果出来后：

- 选择最有利的 baseline；
- 选择最有利的阈值或归一化区间；
- 只保留成功 case；
- 把 runtime failure 填为零收益；
- 选择最有利的 judge 结果；
- 用多个 revision 的最好结果替代预先定义的 selection policy。

---

## 8. 相关方法依据

- AlphaBench：借鉴 Reliability、Stability、Search Cost、Success Rate、ImprovedRate、ImprovedRatio，以及基于固定有效阈值的饱和式 gain。
- Hyndman & Koehler (2006)：relative error/relative measure 在 baseline 接近 0 时不稳定，支持使用固定尺度或 anchor，而不是简单除以 baseline。
- Agarwal et al. (2021)：支持 IQM、归一化分数、performance profile、probability of improvement 和分层 bootstrap。
- Demšar (2006)：支持 paired case difference、wins/losses/ties、Wilcoxon/Friedman 类比较，但需要注意 case 独立性假设。
- Goodwin (1998)：支持用 active return 和 Information Ratio 衡量相对 benchmark 的主动价值。
- Diebold & Mariano (1995)：支持对逐期预测损失差进行时序相关性稳健比较。
- White (2000) 与 Hansen (2005)：当从多个候选、baseline、revision 或策略中挑选最优结果时，需要处理 data snooping。

---

## 9. 一句话版本

最终 rubric 不问“谁的一个总分最高”，而问四个问题：

1. **Outcome：** 结果本身是否有效？
2. **Uplift：** 相对冻结 baseline 是否增加了价值？
3. **Process：** 代码、协议、稳定性和研究过程是否可靠？
4. **Efficiency：** 这些结果付出了多少 token、金钱、时间和尝试成本？

只有同时报告这四类证据，才能区分“效果好但贵”和“略低但高效”。
