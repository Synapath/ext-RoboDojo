# 叠碗的原生成功与稳定叠放

原生 `stack_bowls` 要求双臂回到初始位姿附近；因此三碗已经叠放、释放后保持稳定，也可能原生失败。最终 score 的 100 分受整任务 success 门控，失败时可降到 15 分，不能仅从 15 分推断第三碗没叠好。

本扩展提供 `stable-stack-v2` 和原生成功的独立记录。默认关闭；不修改任务 reward、原生几何容差、动作、物理参数或历史结果。它是研究用的额外任务口径，不替代官方 leaderboard。

## 判据合同

只支持 standard `stack_bowls`、`arx_x5`、25 Hz：

- 三碗倾角均 <45°，最低碗 <7°；按高度相邻两碗水平距离 ≤40 mm、高差 ≥5 mm。
- 双爪实测归一化开度 ≥0.8。
- 每个机器人碰撞 link 的 mesh 包围盒与每只碗的有向包围盒，至少存在间隔 ≥10 mm 的分离平面。包含世界轴、盒面轴和边叉积方向；这是保守的无接触代理，不是实际接触传感器。
- 连续 26 个控制帧，即包含两端的 1 秒窗口内，碗相对首帧的位移 ≤5 mm、旋转 ≤5°。
- 首次通过锁存；不要求归位。缺失帧、非法位姿、缺少机器人/碰撞几何时拒绝，不能用零填充补齐。

## 启用方式

在现有 `python src/eval_client/main.py ...` 命令末尾追加以下参数之一。仍需原来的任务、本体、policy host/port、设备等完整参数；本扩展不提供或改变 policy server。

| 参数 | 结束条件与结果 |
|---|---|
| `--stack_bowls_metric off` | 默认原有行为 |
| `--stack_bowls_metric shadow` | 保持原生结束时机，同时记录稳定终点；原生提前结束但未观察到稳定窗口时，稳定结果为 `null`/截断未知，不能记为失败 |
| `--stack_bowls_metric dual` | 原生和稳定终点都通过才提前结束，否则至原 800 步或异常结束；原生指标锁存，不能与历史原生停止轨迹声称逐位一致 |

CLI 将模式写入输出身份；resume manifest 校验模式，不允许跨模式恢复。普通 `success/score/success_rate` 保留原生语义。在 dual 模式中，顶层 `native_scope=shadow-on-dual-trajectory` 明确其观察轨迹不同。新终点在 `details[episode].stack_bowls` 中，不复用原生 success 字段。

每批输出 `stack-bowls/s<seed>-l<layout>.jsonl` 的逐控制步几何，以及 `episode_<index>.metrics.json`。非法/不稳定 episode 沿用原有排除行为；全环境 reset 时重建观察窗口。采样读取现有位姿，不推进仿真、渲染或改变策略 RNG。

## CPU 分析工具

依赖 Python 3.11+、NumPy；不需要 Isaac Sim、GPU 或 policy。已有 X1 的 geometry JSONL 也可读取。仅有机械臂 telemetry 或 RGB 视频不能补出缺失的碗几何真值。

```bash
python scripts/analyze_stack_bowls.py \
  --urdf Assets/Robots/x5/X5A.urdf \
  --output /tmp/stack-bowls-analysis.json \
  /path/to/geometry/s0-l03.jsonl
PYTHONPATH=. python -m pytest -q tests/test_stack_bowls_metric.py tests/test_stack_bowls_integration.py
```

分析命令读取 trace 与对应 collision meshes，写入新的 JSON；已有输出不会被覆盖。结果保留每例时间点、截断状态和源 hash，不把混有截断结果的集合自动汇总成成功率。

## 证据与发布边界

见 [分析包索引](stack-bowls-evidence.json)。索引固定此前 55 个 baseline、4 个局部状态闭环续段的包身份；原始包包含内部运行记录，未上传到公开 Git。具有证据访问权限的贡献者可从其既有证据存储取得归档，核对 hash 后在新目录解包，并执行包内 `sha256sum -c review-manifest.sha256`。

迁移后的 CPU 分析对这 59 条真实轨迹重算，原生及稳定首次成功步均与冻结结果一致；55 条 baseline 保持 39 个原生成功、46 个稳定叠放成功。此差异属于判据定义，不是模型提升。源码级集成测试覆盖原生早停、截断、双终点、归位失败、模式恢复隔离和重复查询。

本次是基于公司 fork main 的实现提交，尚未在共享 worker 上做新版本端到端 smoke。历史隔离运行器的 GPU 结果不替代本提交的部署验证。部署前应把提交移植到实际 worker 版本、构建新镜像并做小规模 shadow/dual canary；不能仅因为本仓 main 较新或已有报告就覆盖现场镜像。服务 API/UI 还需要消费新增的嵌套指标，不会因本模块合并而自动改变工作台展示。

本次验证为 19 项 CPU 测试、59 条已有轨迹重算，以及单任务 smoke 的 dry-run。独立 worktree 的 doctor 已执行：源码/config/inventory 检查通过，但未挂载 Assets，四项资产目录检查失败。未运行本版本 Isaac Sim smoke，也未声明获得新的 `_result.json`。
