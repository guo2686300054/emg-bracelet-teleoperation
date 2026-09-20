# 单受试者个性化灵巧手采集协议 v2

> 原始方向正确，但用户给的是最低时长，不是上限。为了让 train/validation/sealed-test 都完整覆盖动作因子，并增加 hold 正样本和硬真值门禁，本版有效量提高到 **160.4 分钟**。

## 采集总量

| 数据块 | 数量 | 唯一有效时长 |
|---|---:|---:|
| 结构化连续屈伸 | 3 splits × 132 条件 × 12 秒 = 396 trials | 79.2 分钟 |
| 激活 hold 正样本 | 3 splits × 44 条件 × 8 秒 = 132 trials | 17.6 分钟 |
| rest/no-action | 12 × 20 秒 | 4 分钟 |
| 自由连续动作 | 6 × 100 秒 | 10 分钟 |
| 闭环困难样本 | 独立 train-hard 10 + validation-hard 10 + test 0；每个 30 秒 | 10 分钟 |
| 鲁棒性 | 3 splits × 3 姿态 × 2 疲劳状态 × 11 指组 × 12 秒 | 39.6 分钟 |
| **配额合计** | **764 trials** | **160.4 分钟** |

另采 24 个 transition trials，共 4 分钟，不计最低配额；train/validation/sealed-test 各 8 个，进入 phase 分类和 phase 指标，但严格留在各自 base date/session/donning。全部合计 788 trials、164.4 分钟合格信号。

## 三个切分必须各自完整

| split | 日期/session/佩戴 | 连续屈伸 | hold | 闭环困难样本 | 状态 |
|---|---|---:|---:|---:|---|
| train | 独立 date-train/session/donning-01 | 全部 132 条件 | 全部 44 条件 | 0 | 可训练 |
| validation | 独立 date-validation/session/donning-02 | 全部 132 条件 | 全部 44 条件 | 0 | 只调参 |
| sealed-test | 后续 date-test/session/donning-03 | 全部 132 条件 | 全部 44 条件 | 0 | 首次训练前永久封存 |

初模冻结后，困难样本使用新的 `date-train-hard/session/donning-04` 和 `date-validation-hard/session/donning-05`，绝不复用已结束的 01/02。因此本协议最少需要 5 次独立佩戴。132 条件 = 11 指组 × slow/normal/fast（0.25/0.5/1.0 Hz）× 20/50/80/100% commanded ROM。44 个 hold 条件 = 11 指组 × 4 ROM。必须先按 date/session/donning 分组，再切窗口；重叠窗口不能跨 split。单人结果不能证明跨人泛化。

## commanded 和 measured 绝不混用

`commanded_phase/ROM/speed` 只是提示；`measured_phase/ROM/speed` 只能来自同步且校准过的摄像头、数据手套或编码器。没有真值时 measured 字段必须为 null，不能复制 commanded。

拟合前只能读取 train/validation 真值，sealed-test 的标签和真值不可提前读取。公共门限为同步误差 p95 ≤ 20 ms、最大值 ≤ 50 ms，且每个 session/重新佩戴后校准。ROM 头还要求校准误差 ≤ 5 个百分点、每 trial 真值 ≥ 95%、每个 train/validation×finger×ROM×phase 单元 ≥ 90%。速度头独立要求校准绝对误差 ≤ 0.05 Hz、相对误差 ≤ 10%、每 trial 真值 ≥ 95%、每个 train/validation×finger×speed×phase 单元 ≥ 90%；失败则该模型版本永久禁用速度回归且不得发布速度回归指标。

封存前只允许执行与模型无关的文件、通道、数据包、时间戳、丢包和设备质量 QC，不得解密或汇总 test 标签/真值。最终模型和分析代码冻结后只解封一次，并对 test truth 独立执行相同门禁；失败则相应测试结果无效，且不得把失败、标签、真值、预测或错误反馈给采集、调参或重训练。

## 疲劳安全硬限制

疲劳诱导最多 60 次或 300 秒；连续主动工作最多 120 秒；每 10 次或 60 秒复查 RPE，使用 MVC 时每 20 次复测；之后强制恢复 300 秒。疼痛、麻木、抽筋、眩晕、异常无力、RPE ≥ 8、受试者要求或触及任一硬限制时立即停止。恢复后仅当 RPE ≤ 2 且无症状才继续，否则当天终止。本协议是工程限制，不构成医疗建议。

## 模型路线

现有 `collection_protocol.py` 的 stage 0 以 schema/version/SHA-256、`canonical_session` provenance 和 `session_quality.training_usable=true` 门禁绑定，继续只做 `rest/fist/open_hand` 粗分类链路验证。精细控制在真值门禁通过后采用动作/手指集合分类 + ROM 连续回归，必要时增加速度回归；门禁失败则严格退化为分类模型。
