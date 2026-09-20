# sEMG 肌电手环上位机使用说明

8 通道表面肌电（sEMG）手环上位机采集程序。基于 Python + PyQt5 + Bleak 开发，
负责连接肌电手环、实时接收并显示肌电数据、保存采集数据，并通过共享内存文件
`emg_shared_data_v2.bin` 把最新一帧及其时间、有效性和链路状态提供给外部程序读取。

---

## 当前 BLE 通知协议（实机交付配置）

`config.ini` 通过版本化的 `[Protocol]` 显式选择通知格式。交付内的脱敏摘要
`ble_protocol_evidence_summary.json` 固定记录证据 ID、原始证据 SHA-256 和统计：
511/511 条通知均为 28 字节，且后12字节全部为零。
约 51.1586 Hz 只是主机观测通知率，不代表已经确认的 ADC 采样率。

当前 `wire28_logical16_zero_suffix_v1` 模式严格要求：

- 原始通知恰好 28 字节，前 16 字节为逻辑包，后 12 字节全部为 `0x00`。
- 8 通道仍取逻辑包的 `1,3,5,7,9,11,13,15` 字节。
- 27/29 字节、非零尾部和配置不匹配全部拒绝，不静默截断。
- RawAudit 在解码前保留完整 28 字节，并记录实际原始长度与协议模式。

历史设备必须显式使用 `logical16_odd_bytes_v1`（16/16、padding=`none`）。
两种协议模式不能混用。

## 一、功能

- 蓝牙（BLE）搜索、连接肌电手环设备。
- 开始 / 停止实时采集，采集数据保存为 CSV。
- 单窗口 / 多窗口实时波形显示。
- 录入受试者信息；日志和数据目录只使用随机 `SubjectKey`，不使用姓名。
- 设备地址先经本机持久密钥 HMAC 映射为 `DeviceKey`，原始地址不会写入日志或数据路径。

首次运行会在数据根目录之外创建 `.emg_identity/device_identity.key`。迁移数据时应安全备份该密钥；
删除或更换密钥会让同一物理设备得到新的 `DeviceKey`。
- 历史数据回放（CSV）。
- 通过 `emg_shared_data_v2.bin` 向外部程序实时共享数据和质量状态。

## 二、技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 界面 | PyQt5 | 主窗口、对话框、状态栏 |
| 蓝牙通信 | Bleak | 扫描、连接、notify、GATT 写入 |
| 波形显示 | pyqtgraph | PlotWidget 实时曲线 |
| 数据保存 | csv / pandas / numpy | 每次采集独立的 `samples.csv` 与 `metadata.json` |
| 实时共享 | mmap + struct | `emg_shared_data_v2.bin` 版本化内存映射 |
| 并发调度 | qasync + asyncio + bounded queue | 单一 Qt/asyncio BLE 循环与独立数据消费者 |

## 三、环境配置

### 3.1 安装 Python

- 推荐 **Python 3.10 及以上**（交付的 exe 由 Python 3.10 打包；源码兼容 3.8+，
  在 3.12 下亦可运行）。
- Windows 建议 64 位，安装时勾选 "Add Python to PATH"。

### 3.2 安装依赖库

在项目目录下执行：

```bash
pip install -r requirements.txt
```

各库的作用与版本：

| 库 | 版本 | 用途 |
|----|------|------|
| PyQt5 | 5.15.7 | 图形界面 |
| pyqtgraph | （任意较新版本） | 波形绘制 |
| bleak | 0.11.0 | BLE 蓝牙通信 |
| qasync | 0.28.0 | 让 asyncio 协程运行在 Qt 事件循环中 |
| numpy | 1.26.4 | 数值处理 |
| pandas | 2.2.3 | CSV 回放读取 |
| pyinstaller | 6.14.2 | 打包 exe（仅打包时需要） |

> 注意：若运行时报 `ModuleNotFoundError: No module named 'pyqtgraph'`，
> 说明缺少 pyqtgraph，单独执行 `pip install pyqtgraph` 即可。

## 四、运行

### 4.1 源码运行

```bash
python qt5_bleak.py
```

## Offline EMG training baseline

Training is a separate, engineering-only workflow. Install its pinned dependency with:

```powershell
python -m pip install -r requirements-training.txt
```

Prepare a trusted, subject-grouped manifest from explicit canonical schema-1.8 sessions with confirmed sample-rate evidence:

Training eligibility requires an explicit, versioned `training_provenance` metadata object (`schema: emg.training.provenance`, `version: 1.0`) whose `kind` is one of `canonical_session`, `synthetic_test`, or `external_benchmark`. The normal GUI recording path intentionally does not set this field, so ordinary GUI captures are not training-eligible by default.

Create an eligible recording through the controlled acquisition/fixture code by constructing its `RecordingContext` with an explicit provenance, for example `RecordingContext(..., training_provenance="canonical_session")` for an approved real acquisition protocol or `training_provenance="synthetic_test"` for tests. `external_benchmark` is reserved for a validated benchmark importer. The session must also close cleanly, have zero recorded loss, pass the quality gate, use canonical labels/phases, and carry confirmed sample-rate evidence. Missing, unknown, or unsupported provenance is rejected rather than inferred from `experiment_id` or filenames.

```powershell
python training_dataset.py <session-dir> [<session-dir> ...] `
  --output dataset-manifest.json `
  --window-ms 200 `
  --step-ms 50 `
  --seed 0
```

Train the fixed `StandardScaler` plus shrinkage-LDA baseline from that manifest only:

```powershell
python train_emg_baseline.py `
  --manifest dataset-manifest.json `
  --session-root <canonical-session-root> `
  --output emg-baseline-bundle `
  --seed 0
```

The output directory is published atomically and is never overwritten. It contains bounded, digest-checked JSON (`model.json`, `evaluation.json`, and `digests.json`); it never contains pickle or joblib executable objects. Direct-session loading, guessed sample rates, label overrides, and manifest bypasses are intentionally unsupported.

Measure inference latency on the target upper computer after training:

```powershell
python benchmark_emg_inference.py --bundle emg-baseline-bundle --warmup 100 --iterations 2000 --json latency.json
```

All bundles produced by this baseline are `engineering_only`. The current local recordings are not valid training evidence, and accuracy on synthetic fixtures is only a pipeline check. Neither result demonstrates this handband's real-world recognition accuracy or authorizes robotic-hand control.

程序从项目目录的 `config.ini` 加载并严格校验配置；缺失配置项使用安全默认值。


## 五、数据流

```
肌电手环 ──(BLE notify)──> 有界非阻塞队列 ──> 单消费者解析为 EmgFrame
                                                   ├─> 实时波形显示（始终独立）
                                                   ├─> 会话 CSV（仅录制时）
                                                   └─> emg_shared_data_v2.bin
```

## 六、共享内存格式 `emg_shared_data_v2.bin`

文件固定 **1024 字节**并统一使用小端字节序。v2 头固定为 128 字节，payload 从
偏移 128 开始。既有字段保持原偏移；minor 1 在原保留区偏移 `108..115` 增加小端
`connection_generation uint64`，其有效性由 flags 的 bit 12
`CONNECTION_GENERATION_VALID` 表示。`generation` 表示共享 writer 实例生命周期，
`connection_generation` 表示 BLE 连接生命周期，两者不能互相替代。

CRC32 输入顺序是：先计算头部字节 `[0:96]`，minor 1 再接续计算
`connection_generation` 字节 `[108:116]`，最后接续计算有效 payload
`[128:128+valid_length]`；CRC 字段、commit sequence 和其余保留区不参与 CRC。
minor 0 的历史格式只计算 `[0:96]` 后接 payload，且读取时必须显式设置
`allow_legacy_minor=True`，默认不会猜测或静默降级。读取端必须通过
`SharedMemoryReader` 校验稳定提交序号、版本、长度和 CRC，不要自行按旧偏移读取。

当前设备协议未证明设备包序号、设备采样计数或设备时钟，因此这些字段保持 unknown，
不能用主机接收序号冒充。`DISCONNECTED`、`STALE`、`OVERFLOW` flags 用于诊断链路状态。

读取示例见 `test.py`：

```bash
python test.py emg_shared_data_v2.bin
```

## 七、CSV 数据格式

采集时生成 `<data_path>/<SubjectKey>/<session_id>/samples.csv` 和 `metadata.json`。
本次诊断交付已经取得用户授权，因此交付的 `config.ini` 明确保持
`[RawAudit] enabled=true`，用于定位重复通知、丢帧和包长异常；停止诊断后应改回
`enabled=false`。启用时生成轮转限额的 `raw_packets.jsonl`。sidecar 仅保留主机时间、主机接收序号、BLE 连接代次、包 hex 和解析错误；
启动时只保证完整保存长度恰好等于当前协议 `wire_packet_size` 的通知，包括合法包和同长度但因非零 padding 被拒绝的包。
27/29 字节等非协议长度不享有“必定完整”的协议保证：若 `payload_prefix_bytes` 和单文件上限允许，会按当前策略完整保存；否则只保留
payload 前缀、`original_length` 和 SHA-256。默认 `payload_prefix_bytes=256` 时，27/29 字节通常会完整保留，这是配置效果而非协议不变量。
policy 首行通过 `complete_payload_guarantee` 和 `non_wire_length_policy` 记录这一区别，用于定位设备重复发送、上位机丢弃或包长异常。
Raw packet 属于生物医学敏感数据。除本次已授权诊断外，再次启用前必须取得明确授权，限定访问人员和保留期；
到期或诊断结束后应按组织的安全删除流程处理 sidecar 及其轮转备份，不得将其作为普通调试日志长期保存。
CSV 核心字段为：

```
host_wall_timestamp_ns, host_monotonic_ns, host_receive_index, sample_index,
generation, connection_generation, device_packet_sequence, device_sample_counter, device_time_ticks,
sample_in_packet, action_label, action_phase, sample_rate_hz, device_id,
session_id, quality_flags, channel_1 ... channel_8
```

| 字段 | 含义 |
|------|------|
| host_wall_timestamp_ns | 主机接收通知时的 Unix 纳秒时间 |
| host_monotonic_ns | 主机单调时钟纳秒，用于可靠计算时间间隔 |
| host_receive_index | 主机收到 BLE notify 的递增序号，不代表设备包序号 |
| sample_index | 上位机成功解析的样本递增序号 |
| generation | 共享 writer 实例代次；writer 重启时变化 |
| connection_generation | BLE 连接生命周期代次；每帧来自通知 envelope，不从当前 profile 状态补读 |
| device_packet_sequence / device_sample_counter / device_time_ticks | 当前均留空，等待设备协议证据 |
| action_label / action_phase | Sprint 1 留空，后续标注流程写入 |
| sample_rate_hz | 只有经证据确认的采样时间基准才写值；当前保持空 |
| channel_1 ~ channel_8 | 8 通道肌电值（0–255） |

当前只能确认通知中第 `1,3,5,7,9,11,13,15` 字节被输出为 8 个 `uint8` 通道值。
ADC 位宽、真实采样率、缩放、偏置、滤波、整流、归一化和量化链路尚无设备端证据，
配置与元数据必须保持 `unknown`。历史 CSV 估计的约 51.15 Hz 仅是主机观测通知率。
## 八、时间戳字段说明

共享内存与 CSV 都保存同一个 `EmgFrame` 的 `host_wall_timestamp_ns` 和
`host_monotonic_ns`。展示日历时间时可将墙钟纳秒除以 `1_000_000_000` 后按本地时区格式化；
计算相邻通知间隔应使用单调时钟。采样率未知时不能用 `sample_index / 采样率` 构造时间轴。

## 九、BLE 协议说明

程序通过 GATT service / characteristic 与设备交互，涉及 UUID 如下：

| UUID | 用途 |
|------|------|
| `19b10000-e8f2-537e-4f6c-d104768a1214` | 扫描 / 通知服务 |
| `19b10001-e8f2-537e-4f6c-d104768a1214` | 通知特征（notify，上传数据） |
| `19b10002-e8f2-537e-4f6c-d104768a1214` | 命令特征（write，下发控制） |
| `19b10003 / 19b10004 / 19b10005`（同后缀） | 备用服务组（第二组） |

控制命令（`setDataType`）：

| 命令 | 字节 | 说明 |
|------|------|------|
| 开启通知 | `0x01` | 本手环开启数据上传 |
| 停止 | `0x00` | 停止数据上传 |

数据包解析（`AcquisitionPipeline.parse_notification`）严格服从 `[Protocol]`：历史
`logical16_odd_bytes_v1` 只接受16字节；当前 `wire28_logical16_zero_suffix_v1`
只接受28字节且要求后12字节全零。验证通过后，统一从前16字节逻辑包的
`1,3,5,7,9,11,13,15` 位得到8通道各1字节的肌电值。

> 说明：当前上位机输出的是**每通道 1 字节（0–255）**的值，为原始 ADC 值、
> 滤波后值、整流包络或归一化值，需要结合设备端固件

## 十、文件结构

```
├── qt5_bleak.py               # 主程序入口
├── bleak_ble.py               # BLE 通信封装（扫描/连接/通知/命令）
├── acquisition_pipeline.py    # 有界通知队列、解析、质量状态与帧扇出
├── shared_memory_v2.py        # v2 共享协议读写和一致性校验
├── data_recorder.py           # 会话 CSV、元数据和异常恢复
├── app_config.py              # 严格配置与信号链元数据
├── app_logging.py             # 结构化滚动日志与敏感信息脱敏
├── on_time_show_dialog.py     # 波形显示窗口逻辑
├── user_info.py               # 受试者信息录入对话框
├── Qt5_MainWindow.py          # 主窗口 UI（pyuic5 生成）
├── UI_on_time_show_dialog.py  # 显示窗口 UI
├── UI_user_info.py            # 受试者信息 UI
├── background_rc.py           # Qt 资源（图片等）
├── test.py                    # 共享内存读取示例
├── config.ini                 # 配置文件（串口/蓝牙/采集参数）
├── qt5_bleak.spec             # PyInstaller 打包脚本
├── requirements.txt           # 依赖清单
├── README.md                  # 本文档
├── *.ui / *.qrc               # UI 源文件（可用 pyuic5 重新生成）
├── my_pictures/               # 界面图片资源
└── data/                      # 默认采集数据输出目录（运行时生成）
```

## 十一、常见问题

| 现象 | 排查 |
|------|------|
| `No module named 'pyqtgraph'` | `pip install pyqtgraph` |
| 搜索不到设备 | 确认手环已上电、在蓝牙范围内、未被其他程序占用 |
| 连接后收不到数据 | 确认已点"开始采集"或"单窗口/多窗口显示"以开启 notify |
| 外部程序读不到共享文件 | 显式运行 `python test.py emg_shared_data_v2.bin` |
| CSV 打开乱码 | 文件为 UTF-8-SIG 编码，用 Excel 打开或指定 `encoding='utf-8-sig'` |
# Legacy history audit and conversion

Audit an annotated legacy measurement batch without changing its source files:

```powershell
python audit_legacy_dataset.py <measurement-manifest.json> --output <new-audit-report.json>
```

Convert the full measurement manifest with the strict existing converter, then immediately
validate the published result:

```powershell
python convert_legacy_dataset.py <measurement-manifest.json> --output <new-output-directory>
```

Both tools refuse to overwrite outputs. Converted data is explicitly marked
`training_provenance.kind=legacy_experimental` and `training_usable=false`; it cannot be
presented as a `canonical_session`.
