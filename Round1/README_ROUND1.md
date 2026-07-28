# Round1 — PlanGrad-UAV 完整梳理与代码/结果/权重总说明

本文件是对 `Round1/` 目录的权威说明。它不是从旧 `CODE_GUIDE.md` 照抄——
**每一条都对照 Lab 上的实际文件逐一验证过**，并在末尾（§9）列出旧
`CODE_GUIDE.md` 中与实际不符、已被本文件纠正的地方。

> 项目定位：PlanGrad-UAV 是一台**可微分测量仪器**，不是"要打败基线的方法"。
> 三条主结论：(1) 安全由硬 CBF 证书治理；(2) 位移精度与运行安全解耦；
> (3) 端到端联合训练相对纯域适应无可测量的安全增量（否定结果）。

---

## 0. 一眼看懂 Round1 目录

```
Round1/
├── README_ROUND1.md              ← 本文件（总说明）
├── RUNBOOK_main.md               主流程操作手册（环境→数据→训练→评估）
├── 01_core_library/              核心库：仿真器+预测器+规划器+训练/评估/诊断 (35 .py)
├── 02_baselines/                 6 个基线臂 + 统一评估 harness + 生图脚本
├── 03_verification_runs/         本轮新增的 3 个一次性核查实验（脚本+结果+说明）
├── 04_weights/                   全部 9 个训练好的 .pt（"最终使用"标注见 §4）
├── 05_results/                   全部结果产物
│   ├── main_and_stats/           论文主表 + 统计检验 + 各基线 result
│   ├── diagnostics/              机制诊断（RQ3/RQ5/失配/探测时域/渗透率）
│   └── robustness/               鲁棒性（校准修复 + 匹配 planner；含"勿用"的旧配置结果）
├── 06_figures/                   论文全部 12 张图（PDF/PNG + TikZ/py 源）
└── 99_archive_logs_and_smoke/    历史日志 + 冒烟测试结果（留档，不参与论文）
```

**核心区分（务必理解）**：
- **代码逻辑**（`.py`）在 `01_core_library/` 和 `02_baselines/`。
- **训练好的权重**（`.pt`）集中在 `04_weights/`。
- **实验结果**（`.txt`/`.json`）在 `05_results/`。
- 原始仓库里代码和权重是**分开两处**的（`code/plangrad_sim/` 放代码、顶层
  `plangrad_sim/` 放权重）；跑代码时代码从 `code/plangrad_sim/` 运行、权重路径
  指向 `../../plangrad_sim/*.pt`。**Round1 只是分类汇总便于阅读；要实际重跑，
  仍建议在原始 `code/plangrad_sim/` 目录下跑**（见 §7）。

---

## 1. 环境与数据（重跑前提）

- **GPU 必需**：可微 QP（cvxpylayers）CPU ~36 s/步、GPU ~0.09 s/步。验证机 RTX 4090。
- **依赖**（机器重置后需重装）：
  `pip install "scipy==1.15.3" "h5py==3.16.0" "cvxpy==1.7.5" "cvxpylayers==0.1.9"`
  （torch 2.11 + CUDA、numpy 2.2.6 通常已在）。
- **GUAM 数据不在仓库里**：
  `git clone --depth 1 https://github.com/nasa/Generic-Urban-Air-Mobility-GUAM.git`
  然后 `export GUAM_MAT=/abs/path/GUAM/Challenge_Problems/Data_Set_1.mat`。
- **复现铁律**：`seed=12345`、`n=200`、`batch=8`（n//batch 恰好 200；batch=16 只跑
  192 丢尾巴，ADE 从 4.32 漂到 4.07）、部署 planner `α=0.1, Hp=15, a_max=20,
  d_sep=30`。训练用 GUAM 轨迹 0–2499、评估用 2500–2999（不重叠）。

---

## 2. `01_core_library/` — 每个 .py 的用途（已验证）

### 2.1 基础设施（仿真器与数据）
| 文件 | 用途 | 关键实现 |
|---|---|---|
| `params.py` | eVTOL 物理参数 | NASA SACD Lift+Cruise，英制→SI；质量 2653 kg。`DEFAULT_PARAMS`。|
| `dynamics.py` | 可微 6-DOF 动力学 `x_{t+1}=g(x,u,w)` | 12 维状态/4 维控制，RK4，纯 torch。`EVTOLDynamics`。|
| `wind.py` | 城市风场 | 平均风+随机傅里叶特征高斯随机场，乘子 `eta_w`。`UrbanWindField`。|
| `guam_data.py` | 加载 GUAM `.mat`（h5py） | 3000 条 Bernstein 轨迹；`resample_by_arclength` 赋物理时间尺度（关键）。|
| `dataset.py` | Stage-1 滑窗数据集 | (历史 L=25,未来 T=30) 窗口，recenter+/100，可选 yaw 增强；`add_kinematic_features` [B,L,3]→[B,L,6]。|
| `guam_encounters.py` | 遭遇场景生成 | ego + 减速/旋转/平移的 neighbour，medium 难度。`GUAMEncounters`。|
| `simulator.py` | 闭环 rollout 封装 | 仅被 `test_smoke.py` 使用。|
| `seeding.py` | 复现性 | `set_seed(12345)` + cudnn.deterministic。|
| `config.py` | GUAM 路径解析 | 环境变量 `GUAM_MAT` 优先。|

### 2.2 模型组件
| 文件 | 用途 | 关键实现（已核实） |
|---|---|---|
| `predictor.py` | 轨迹预测器 f_θ | `GMMTrajectoryPredictor`：GRU(6→128,2层)→2层MLP→K=5 高斯混合头 {α,μ,logσ},T=30。**唯一输入是单邻机历史位置**（无地图/风）。|
| `cbf_mpc.py` | **可微**安全 MPC（训练用） | `CBFMPCLayer`（**非 nn.Module,无可训练参数**）。**QP 是双积分器代理**（决策=指令加速度,`p_{k+1}=p_k+dt·v_k`,`‖a‖∞≤a_max`），**非**完整非线性动力学；barrier 用预测**均值**线性化（无 Σ/κ 进约束）。α 是编译期常数 float（DPP 禁 α^k 自乘积）。cvxpylayers，梯度经 KKT。|
| `fast_cbf_mpc.py` | **非可微**快速 CBF-MPC（评估用） | `FastCBFMPC`，OSQP，比 cvxpylayers 快 ~10-50×。|
| `safe_policy.py` | 端到端策略 | `SafePolicy`：预测均值喂 QP→指令加速度→`accel_to_control`（带饱和的前馈+姿态PD内环）→推力/力矩。**只喂均值,Σ 只返回 info 不进约束**。|
| `orca_baseline.py` | ORCA（渗透率未装备机） | van den Berg 2011 互惠避让。|

### 2.3 训练脚本（→ 生成权重）
| 文件 | 训练哪个权重 | 说明（已核实） |
|---|---|---|
| `train_stage1.py` | `stage1_full.pt` | 损失=`w_nll·GMM-NLL+w_ade·minADE-K`（默认1.0/1.0）。**必须 `--yaw_augment`**。`--n_traj 2500 --epochs 30 --batch 512 --lr 8e-4`。|
| `train_stage2.py` | `stage2_final.pt`(=PlanGrad) | TASL 微调,穿 CBF-MPC 闭环。**优化器只含 pred.parameters(),规划器 φ 冻结**。**此为 patched 版（含 `--a_max` CLI,默认 10.0 不改原行为）**。⚠️ TASL 的 lead 项见 §8。|
| `train_stage1b.py` | `stage1b_domainadapt.pt` | 域适应对照：同 Stage-2 数据/协议,损失=**纯 ADE-anchor**（mixture-mean 位移,不过规划器）。⚠️ 与 Stage-1 的 L_ADE 不同型,见 §8。|

### 2.4 评估脚本（→ 生成结果）
| 文件 | 用途 | 权重 | 结果→ |
|---|---|---|---|
| `final_best.py` | headline Stage-1 vs Stage-2 | stage1_full, stage2_final | `BEST.txt` |
| `eval_stage1_vs_stage2.py` | 核心 RQ2 | stage1_full, stage2_final | 打印 |
| `eval_mismatch.py` | RQ4 模型失配 | stage1_full, stage2_final | `MISMATCH*.txt` |
| `eval_leadtime.py` | 探测时域扫描 | +oracle | `LEADTIME*.txt` |
| `run_penetration.py`+`penetration_sim.py` | 渗透率扫描 | stage2_final+ORCA | `PENETRATION*.txt` |
| `final_compare.py` | ⚠️**旧 planner 配置**大规模对比 | stage1_full, stage2_final | `FINAL2.txt`（**勿用于最终表**） |

### 2.5 诊断脚本
| 文件 | 用途 | 权重 | 结果→ |
|---|---|---|---|
| `diagnose_conflicts.py` | RQ3 冲突归因 | stage1,stage2 | 打印 |
| `rq5_error_profile.py` | RQ5 critical/inert 误差剖面 | stage1_full,stage1b,stage2_final | `RQ5_PROFILE.txt` |
| `diag_error_direction.py` | RQ5 误差方向分解 | 同上 | `ERRDIR.txt` |
| `diag_leadtime_6s.py` | 6s 峰机制诊断 | stage1_full | 打印 |
| `scan_planner.py` | 规划器参数扫描 | stage2_final | `PLANNER_SCAN.txt`（⚠️见 §8,已由 03 独立流验证） |
| `scan_scale.py` | 训练池规模扫描 | stage1_full | `SCAN_RESULT.txt` |
| `diagnose_decoupling.py` | 旧版方差重分配 | stage1,stage2 | 打印（已被误差版取代,留档） |

### 2.6 冒烟测试
`test_smoke.py`（悬停/闭环/梯度穿闭环/GUAM 加载，**已在 Round1 验证全过**）、
`test_cvxlayer.py`、`test_cbf_avoid.py`、`test_safe_policy.py`。

---

## 3. `02_baselines/` — 6 个实验臂 + 统一评估

**唯一评分真理源** `common/eval_common.py`：held-out 2500-3000,风种子 7,
best planner α=0.1/Hp=15/a_max=20,d_sep=30,n=200,seed 12345,batch=8。
返回 `{CR_%, minSep_m, ADE_m, LeadT_s, Energy}`。
`common/stats_tests.py`→配对 McNemar+Wilson CI→`STATS.txt`。

| 目录 | 臂名 | 预测器 | 规划器 | 隔离什么 |
|---|---|---|---|---|
| `00_plangrad_reference/` | PlanGrad (ours) | **stage2_final.pt** | CBF-MPC | 参照行 |
| `01_constant_velocity/` | Constant-Velocity | 匀速外推(免训练) | CBF-MPC | 学习型预测器的价值 |
| `02_vanilla_mpc/` | Vanilla-MPC | stage2_final.pt | **无 CBF** | CBF 证书的贡献 |
| `03_fixed_predictor/` | Fixed-Predictor | **stage1_full.pt** | CBF-MPC | Stage-2 买到什么 |
| `04_soft_ipp/` | Soft-IPP | soft_joint.pt | 软规划器 | 硬证书 vs 软惩罚 |
| `05_conformal_mpc/` | Conformal-MPC | stage1_full.pt+保形校准 | CBF-MPC(d_sep+r_conf) | 不确定性感知安全规划 |
| `06_sim_ood/` | RQ4 风扫描 | 所有臂 | — | OOD 风鲁棒性 |

> `04_soft_ipp/train_soft.py` 训 `soft_joint.pt`（同 Stage-1 初始化+TASL,软规划器无 CBF）。
> ⚠️ `05_conformal_mpc/conformal.py` 已是**修复后**版本（校准集 2500-2999,独立种子 777；
> 见 §5 与 `05_results/robustness/calibration_fix/README.md`）。

---

## 4. `04_weights/` — 每个 .pt 是什么、是否最终使用（★=论文最终使用）

| 权重 | 是什么 | 论文用途 | 最终使用？ | 由谁产生 |
|---|---|---|---|---|
| **`stage1_full.pt`** ★ | Stage-1 预测器(位移,yaw增强) | Fixed-Predictor;Stage-2/1b 初始化;Conformal 冻结预测器 | ✅ 是 | `train_stage1.py` |
| **`stage2_final.pt`** ★ | Stage-2 任务对齐 = **PlanGrad** | 主参照臂;Vanilla 预测器;渗透率装备机 | ✅ 是(**论文主模型**) | `train_stage2.py`(训练 planner α0.4/Hp8/amax10) |
| **`stage1b_domainadapt.pt`** ★ | 域适应对照(纯 ADE-anchor) | RQ5 决定性对照 | ✅ 是 | `train_stage1b.py` |
| **`stage2_seed1/2/3.pt`** ★ | Stage-2 三 seed 重训 | 训练随机性统计(CR 11.4±0.4%) | ✅ 是(附录多seed) | `train_stage2.py` 换 seed |
| `soft_joint.pt` ★ | Soft-IPP 预测器 | Soft-IPP 臂 | ✅ 是 | `04_soft_ipp/train_soft.py` |
| `stage2_matched.pt` ★ | 匹配部署配置重训 Stage-2 | RQ5 保险跑(训练=评估 planner) | ✅ 是(附录 robustness) | `03_verification_runs/`(patched trainer,`--Hp 15 --alpha 0.1 --a_max 20`) |
| `stage1_conv_check.pt` | Stage-1 收敛性检查 | 证明已收敛非欠拟合 | ⚠️ 辅助(非表数据) | `train_stage1.py` 收敛监控 |

**权重关系图**：
```
GUAM 0-2499 ──train_stage1.py──> stage1_full.pt (位移训练)
                                     │
     ┌──────────────┬───────────────┼──────────────┬────────────────┐
     │(纯ADE-anchor) │(TASL,CBF α0.4) │(TASL,软规划器)│(TASL,CBF α0.1) │(换seed×3)
     ▼              ▼                ▼              ▼                ▼
stage1b_       stage2_final.pt   soft_joint.pt  stage2_matched.pt  stage2_seed1/2/3
domainadapt    (=PlanGrad,主模型) (Soft-IPP)     (RQ5保险跑)        (多seed统计)
(RQ5对照)
```

---

## 5. `05_results/` — 结果产物地图

### 5.1 `main_and_stats/`（论文主表数据源）
- `BEST.txt` — 最优 planner 下 Stage-1 vs Stage-2（Table 1/2/3 headline）。
- `00..05_*_result.{txt,json}` — 6 个基线臂各自结果（Table 1 各行）。
- `ood_results.{txt,json}` — RQ4 风扫描。
- `STATS.txt` — 配对 McNemar + Wilson CI（统计小节）。

### 5.2 `diagnostics/`（机制分析）
`RQ5_PROFILE.txt`、`ERRDIR.txt`、`MISMATCH*.txt`、`LEADTIME*.txt`、`PENETRATION*.txt`。

### 5.3 `robustness/`（鲁棒性 + 数据完整性修复）
- `calibration_fix/` — **共形校准集数据完整性修复**（校准集原为 2000-2499,与训练集
  0-2499 重叠→违反 split-conformal；改为 2500-2999+独立种子 777）。含 CLEAN vs
  OLD(污染) 对比、修复后 `conformal_FIXED.py`、详细 README。结论：r_conf
  18.57→18.93m,操作指标基本不变。
- `FINAL2_OLD_config_do_not_use.txt` — ⚠️**旧 planner 配置,勿用于任何最终表**（仅留档）。

---

## 6. `03_verification_runs/` — 本轮新增的 3 个核查实验

| 文件 | 是什么 | 结论 |
|---|---|---|
| `scan_planner_indep.py` + `PLANNER_SCAN_INDEP_seed999.txt` | 独立流(seed 999)重跑 6 配置扫描 | 排名复现,`α0.1/Hp15/amax20` 仍唯一最优(12% CR)→配置选择经独立流验证 |
| `zero_slack_feasibility.py` + `ZERO_SLACK_FEAS.txt` | 冲突 episode 强制 ε=0 重解判可行性 | 22/22 冲突 ε=0 不可行→"actuation-limited" 成立(强证据) |
| `eval_matched.py` | 匹配部署配置 Stage-2 评估助手 | 支撑 `stage2_matched.pt` 的 RQ5 保险跑 |
| `README_verification.md` | 三实验动机/方法/结果详述 | — |

---

## 7. 如何重跑（在原始仓库目录,非 Round1）

Round1 是**只读整理归档**。要实际重跑,在原始 `code/plangrad_sim/` 下：
```bash
export GUAM_MAT=/abs/path/GUAM/Challenge_Problems/Data_Set_1.mat
python3 test_smoke.py                        # 冒烟（应全过）
python3 final_best.py                         # headline → BEST.txt
python3 rq5_error_profile.py                  # RQ5 剖面 → RQ5_PROFILE.txt
cd ../baselines/05_conformal_mpc && python3 run.py --n 200 --delta 0.1
cd ../.. && python3 baselines/common/stats_tests.py
```
详细分步见 `RUNBOOK_main.md`。

---

## 8. ⚠️ 已知的"文本 vs 代码"裂缝（写论文/回审稿必读）

核查中发现、已在代码侧确认的不一致处：

1. **TASL 的 lead 项是碰撞权重的重参数化,不是 lead-time**。
   `lead=mean_t σ(+β·margin)`、`soft_coll=mean_t σ(−β·margin)`,共用同一 margin 与 β。
   σ(x)+σ(−x)=1 ⟹ `lead≡1−soft_coll`,故 `w_coll·soft_coll−w_lead·lead=
   (w_coll+w_lead)·soft_coll−常数`。→ lead 项梯度上 ≡ 碰撞权重设为 w_coll+w_lead,
   **不含时间信息**,soft-argmin lead 公式从未实现。处置：按恒等式改写,删 soft-argmin 段。
2. **Φ_delay 是走廊横向偏移**（`√(y²+z²)/100`）,非到达时间延误;评估侧 `eval_common.py`
   **不报 delay 列**。处置：§03 重定义为 Φ_dev(走廊偏移)。
3. **CBF 用预测均值+固定 d_sep,Σ/κ 从不进约束**(训练与评估皆然)。Conformal-MPC 只把
   d_sep 加大 r_conf。处置：4.2/4.3 降为均值基+共形对照臂;Prop 1 删概率覆盖从句,
   Prop 2/3 梯度陈述 (μ,Σ,κ)→仅 μ。
4. **QP 是双积分器代理**,非非线性动力学嵌入+SCP;**风不进 QP**(只作用真实 plant);
   无 warm-start;a₀→u 有未在正文出现的姿态 PD 内环;Hp(8/15)≠预测 T(30)。
5. **规划器"最优配置"原在评估集上选**(测试集调参)→已由 `03_verification_runs/`
   独立流(seed 999)验证排名复现,§05 应写"选择经独立流验证+所有臂共享配置"。
6. **Stage-1 与 Stage-1b 的 L_ADE 不同型**：Stage-1=NLL+minADE;Stage-1b=mixture-mean 位移。
   描述 1b 干净对照时须写明(否则 1b vs Stage-1 差了数据分布+L_ADE 形式两处)。

---

## 9. 对旧 `CODE_GUIDE.md` 的勘误（本次验证发现）

| 旧 CODE_GUIDE 说法 | 实际（已核实） |
|---|---|
| 提到 `diag_errdir_stats.py` 生成 `ERRDIR_STATS.txt` | 该脚本与该结果**均不存在**;方向分解只有 `diag_error_direction.py`→`ERRDIR.txt` |
| 列 `SCAN_RESULT.txt`、`PLANNER_SCAN.txt` 为现成产物 | 两者当前**不在** Lab（需重跑才生成） |
| 各基线 `result.json` "当前不在仓库" | 实际**已存在**于 `baselines/*/result.{txt,json}` |
| `train_stage2.py` a_max 硬编码、无 CLI | 原始版如此;**Round1 收录 patched 版**（含 `--a_max`） |
| Conformal 校准 `range(2000,2500)` disjoint | **错**:与训练 0-2499 重叠;已修复为 2500-2999+独立种子（见 §5） |

> 旧 CODE_GUIDE 的**主体结构与权重/脚本用途描述准确**,上述 5 点是它超前于代码或
> 未及更新之处,本文件已全部纠正,以 Lab 实际为准。

---

*本 README 基于 Lab 实际文件逐一核实生成（含冒烟测试通过验证）。若后续跑出新
权重或结果,请在对应子目录追加,并更新 §4/§5 表格与本文件勘误节。*
