# PlanGrad-UAV 详细操作手册 (RUNBOOK)

手把手在自己服务器从零复现全部结果。固定随机种子 **12345**；训练用
GUAM 轨迹 **0–2500**，评估用 **2500–3000**（不重叠，无数据泄漏）。
**阶段 2 起所有命令都在 `code_all/plangrad_sim/` 目录下运行。**

流程：0 环境 → 1 数据 → 2 自检 → 3 Stage-1 → 4 Stage-2 → 5 评估诊断

---

## 阶段 0：环境准备

**硬件**：必须有 NVIDIA GPU（验证机 RTX 4090）。CBF-MPC 可微 QP 在 CPU
约 36 s/步、GPU 约 0.09 s/步（差 ~400 倍），没 GPU 跑不动 Stage-2/评估。
显存 ≥ 6 GB（实测峰值 <1 GB），磁盘 ≥ 1 GB。

```bash
conda create -n plangrad python=3.10 -y
conda activate plangrad
cd code_all
pip install -r requirements.txt
```
锁定版本：torch 2.11.0 / numpy 2.2.6 / scipy 1.15.3 / h5py 3.16.0 /
cvxpy 1.7.5 / cvxpylayers 0.1.9。若 torch 装成 CPU 版，去 pytorch.org
按 CUDA 版本重装 GPU 版。

验证 GPU：
```bash
python -c "import torch;print('CUDA:',torch.cuda.is_available());print(torch.cuda.get_device_name(0))"
```
预期 `CUDA: True` + 显卡型号。

---

## 阶段 1：拿数据（NASA GUAM，无需 MATLAB）

3000 条真实 Lift+Cruise eVTOL 轨迹，v7.3 .mat，用 h5py 纯 Python 读。
```bash
cd code_all
git clone --depth 1 https://github.com/nasa/Generic-Urban-Air-Mobility-GUAM.git GUAM
```
克隆后应有 `code_all/GUAM/Challenge_Problems/Data_Set_1.mat`（约 23 MB）。
若放别处：`export GUAM_MAT=/abs/path/Data_Set_1.mat`（config.py 优先读它）。

---

## 阶段 2：自检

```bash
cd code_all/plangrad_sim          # 之后所有命令都在这里
python params.py                   # 打印 Lift+Cruise SI 参数(mass≈2653kg)
python guam_data.py                # 打印 #trajectories = 3000 ...
python test_smoke.py               # 预期末行: All smoke tests passed.
python test_cbf_avoid.py           # (可选) 预期 no-CBF≈2m vs CBF≈33m
```
`test_smoke.py` 过了说明整套管线可用（动力学/闭环/梯度/数据全 OK）。
报 `No such file` → GUAM 路径不对，回阶段 1。

---

## 阶段 3：Stage-1 训练（预测器预训练）

**目的**：训练预测器 f_theta，只追求"预测准"。全部 2500 条训练轨迹
+ yaw 增强（对任意朝向鲁棒）。

```bash
python train_stage1.py --cuda --seed 12345 --yaw_augment \
    --n_traj 2500 --epochs 30 --batch 512 --lr 8e-4 \
    --out stage1_full.pt
```
- 耗时：4090 上约 5 分钟。
- 预期：每 epoch 打印 `train_loss | val_minADE(m) | val_meanADE(m)`，
  末行应出现 `saved BEST predictor weights -> stage1_full.pt`。
- `--yaw_augment` 必须加：Stage-2 遭遇场景邻居朝向任意，不增强则
  预测器对跨朝向输入不鲁棒。
- 最终 val minADE ≈ 2–4 m，meanADE ≈ 5–8 m（随数据量波动）。

---

## 阶段 4：Stage-2 训练（任务对齐微调）

**目的**：在 Stage-1 权重基础上，把预测器嵌入 SafePolicy（预测器→CBF-MPC→6-DOF
动力学），跑可微闭环 rollout，用 TASL 损失微调 θ，使预测器把不确定性
分配到规划器能消化的地方。

```bash
python train_stage2.py --cuda --seed 12345 \
    --stage1 stage1_full.pt \
    --iters 70 --batch 16 --T 20 --Hp 8 --lr 1e-4 \
    --w_coll 3.0 --w_lead 0.5 --w_ade 0.12 \
    --out stage2_full.pt
```
- 耗时：4090 上约 15–25 分钟（CBF-MPC 可微 QP 每步有开销）。
- 预期：每 5 步打印 `loss | mean_min_sep(m) | soft_coll | energy | lead | ade`，
  loss 应逐步下降，`mean_min_sep` 应逐步增大。
- 偶尔出现 `SKIP (solver issue: ...)` 属正常（QP 不可行时跳过，不影响收敛）。
- 输出 `stage2_full.pt`。

**重要参数说明**：
- `--w_coll 3.0`：碰撞惩罚权重（主驱动项）。
- `--w_lead 0.5`：冲突预警提前量奖励（鼓励提前规避）。
- `--w_ade 0.12`：ADE 锚定项（防预测器漂移太远，保持基本预测精度）。
- `--Hp 8`：MPC 预测 horizon = 8 步 × 0.2 s = 1.6 s。
- 若 OOM 减 `--batch` 至 8；若 QP 报错频繁可降 `--iters` 至 50。

---

## 阶段 5：评估与诊断

### 5.1 核心 Stage-1 vs Stage-2 对比

```bash
python eval_stage1_vs_stage2.py --cuda --seed 12345 \
    --stage1 stage1_full.pt --stage2 stage2_full.pt \
    --n_eval 96
```
- 预期输出表格：
  ```
  metric              Stage-1     Stage-2      change
  ADE_m                  x.xxx      x.xxx     +x.xxx
  minADE_m               x.xxx      x.xxx     +x.xxx
  conflict_rate_%        xx.xxx      xx.xxx    -xx.xxx
  mean_min_sep_m         xx.xxx      xx.xxx    +xx.xxx
  ```
- **关键观察**：Stage-2 的 ADE 可能与 Stage-1 持平甚至略差，但
  `conflict_rate_%` 应明显降低（约减半）、`mean_min_sep_m` 应增大。
  这正是论文核心论点——位移误差不是安全的充分代理指标。

> 冲突率是 0/1 指标，方差大，n=96 只是快速检验。正式数字用下面 5.2。

### 5.2 大规模最终对比（n≥200）

```bash
python final_compare.py 200
```
- 自动加载 `stage1_full.pt` 和 `stage2_full.pt`，评估 200 条 held-out
  遭遇场景，结果写入 `FINAL2.txt`。
- 预期：Stage-2 冲突率比 Stage-1 低约 15–25 个百分点，minSep 多 8–10 m。

### 5.3 最优规划器配置下的对比

```bash
python final_best.py
```
- 使用调优后的规划器参数（`alpha=0.1, Hp=15, a_max=20`），
  200 条 held-out，结果写入 `BEST.txt`。
- 预期：规划器更强后，两者冲突率都下降，但 Stage-2 仍有优势。

### 5.4 冲突归因诊断

```bash
python diagnose_conflicts.py
```
- 分别统计 Stage-1 和 Stage-2 冲突场景中 QP slack 激活 vs
  预测误差 >20 m 的比例，结果写入 `DIAG.txt`。
- 预期：绝大多数冲突是 slack-active + 小预测误差 → 瓶颈在规划器
  规避权限，不在预测器精度。这是做规划器扫描（5.5）的依据。

### 5.5 预测方差重分配机制验证

```bash
python diagnose_decoupling.py --stage1 stage1_full.pt \
    --stage2 stage2_full.pt --n_eval 96 --seed 12345
```
- 拆分预测轨迹为 CRITICAL 段（最近接近邻机 ±3 步）和 INERT 段，
  报告各段的平均预测方差（trace Σ）。
- 预期：Stage-2 的 critical/inert variance ratio < Stage-1 →
  机制生效，预测器在冲突区域分配了更多不确定性。
- 输出末行应显示 `=> mechanism present`。

### 5.6 规划器参数扫描（可选）

```bash
python scan_planner.py
```
- 固定 `stage2_final.pt`，扫描 6 组 CBF-MPC 参数（alpha / Hp / a_max），
  每组 n=200，结果写入 `PLANNER_SCAN.txt`。
- 耗时较长（4090 上约 30–60 分钟）。
- 预期最优：`alpha0.1 Hp15 amax20` → CR≈12%, minSep≈48 m。

### 5.7 训练池规模扫描（可选）

```bash
python scan_scale.py
```
- 从同一 `stage1_full.pt` 分别用 300/800/1500/2500 条训练池做 Stage-2
  微调，每组 50 iters，快速评估 n=64，结果写入 `SCAN_RESULT.txt`。
- 耗时较长。用于验证 Stage-2 收益非纯规模效应。

### 5.8 训练/评估模型失配（model mismatch）鲁棒性评估 —— 新增

**目的（回应审稿人）**：现有所有评估脚本（`final_best.py` 等）在闭环
rollout 里用的都是与规划器内部模型 **完全相同** 的
`EVTOLDynamics(DEFAULT_PARAMS)`，即"控制器模型 == 被控 plant"，CBF 的前向
不变性保证按构造成立，安全数字是在"模型完美"假设下拿到的。这是本文实证
链条最大的一个洞：从未检验规划器所用模型（其内部 `g`）与真实被控对象
不一致时结论是否保持。

`eval_mismatch.py` **只做评估、不重训**（性价比高）：控制器保持名义
`DEFAULT_PARAMS`（SafePolicy + CBF/Soft-MPC + 内环反馈律都用名义参数），而
rollout 的 plant 用被扰动的动力学：
- 质量误差 `mass_factor`（±15–20%）
- 惯量误差 `inertia_factor`（±30%）
- 推力效率 `thrust_eff`（真实推力 = 指令 × 0.85）
- 执行器延迟 `act_delay`（1–2 拍传输时延）
- 风场偏移（更大 `eta_w`/`gust_std` + 不同 seed）
以及一个"combined"最坏情形。每个 regime 都在同一批 held-out GUAM 遭遇
（2500–3000，seed 12345）上重跑三条对照：Stage-1 +CBF、Stage-2 +CBF、
Stage-2 no-CBF（**= 主表那个 Vanilla-MPC**，见下）。

```bash
python eval_mismatch.py --n 200 --seed 12345 \
    --stage1 stage1_full.pt --stage2 stage2_final.pt
```
- 耗时：4090 上约 40–70 min（QP 主要在 CPU 端，纯评估）。
- 结果写入 `MISMATCH.txt`。
- **两个核心结论**：
  - **Q1 解耦稳健**：所有 regime 下 Stage-2 的 ADE 恒定远低于 Stage-1
    （4.32 m vs 20.90 m），而两者 CR 始终接近 → 位移精度与安全的解耦在
    失配下完全保持。
  - **Q2 CBF 优势稳健且更关键**：nominal 下 no-CBF = **41.0%**（精确 == 主表）
    vs CBF 11–12%；combined 失配下 CBF 30% 而 no-CBF 崩到 **94.5%** → 失配下
    无证书规划器彻底失效，硬 CBF 证书仍把冲突压住,且**优势随失配扩大**。
- **重要修正（回应审稿人第一条,跨表一致性）**：no-CBF comparator **原来**用
  `eval_mismatch.py` 内定义的 `SoftMPCLayer`（二次 shortfall 软惩罚,`w_rep=8`）,
  与主表 Vanilla-MPC（`baselines/02_vanilla_mpc/VanillaMPCLayer`,hinge 软斥力,
  `w_rep=50`）**是两个不同的规划器**,故 nominal 曾报 50.5% ≠ 主表 41.0%。
  已改为**直接 import 主表 VanillaMPCLayer** 作为 no-CBF comparator,nominal 行
  现精确复现主表 **41.0%**,所有 regime 都对同一个 comparator。
  受影响的仅 Vanilla 列:`50.5→41.0 / 72.5→75.0 / 67.5→68.5 / 67.0→87.5 / 92.5→94.5`;
  CBF 两列(S1/S2)不变。`SoftMPCLayer` 类保留在文件中(即主表的 Soft-IPP 规划器形式)。

### 5.9 多智能体渗透率（market-penetration）系统级实验 —— 新增

**目的（回应审稿人 scope/fit 质疑）**：现框架是单 ego 的 MPC 控制问题，
TRC 审稿人会问"transportation science 贡献在哪"。该实验把方法抬到系统级：
一条双向走廊、泊松到达、固定总需求，装备 PlanGrad-UAV 的飞行器比例 p 从
0% 扫到 100%，其余用 **ORCA**（有出处的可信多机避让基线，`orca_baseline.py`，
保证 p=0% 端点不是稻草人）。装备/未装备飞行器共用同一 6-DOF 动力学与执行
包线，只有决策层不同。

关键实现（`penetration_sim.py` + `run_penetration.py`）：
- 每架维护 25 步历史喂预测器；每架把最近 K=3 架当邻居；
- 装备机用 **`fast_cbf_mpc.py`（FastCBFMPC / OSQP）** 解与训练完全相同的
  CBF-MPC QP —— 纯评估不需要梯度，OSQP 比 cvxpylayers 快 ~30–70×，闭环最小
  分离与可微层一致（差 <0.01 m，已验证），这是让 MC 扫描可行的关键；
- 分组统计装备组 / 未装备组的冲突率、吞吐、延误（外部性问题的核心）。

```bash
# 需求先标定到 ORCA 基线 ~50%（arrival 0.16）
python3 run_penetration.py --reps 6 --horizon 400 --warmup 100 \
    --arrival 0.16 --ps 0,25,50,75,100 --stage2 stage2_final.pt --seed 12345
```
- 耗时：4090 上约 1.5–2.5 h（p 越高越慢，全装备点最重）。结果写入 `PENETRATION.txt`。
- **诚实结论（跟数据走，非预设）**：
  - **效率外部性为正且单调**：吞吐 44.8→47.7/min，未装备机延误 6.5→4.4s，
    装备机全程低延误（1.3–2.6s）→ 装备飞行器释放走廊容量、惠及未装备邻居。
  - **安全（冲突率）非单调**：~48–57%，中段（p=25）因 **混合车队 off-distribution
    协调成本**（CBF 机 vs ORCA 机两种避让哲学 + 预测器面对分布外的 ORCA 邻居）
    略升，p=100 全装备时降到最低 47.7%。这印证论文核心论点——冲突率由控制
    权限/协调主导，不由位移预测精度主导（呼应 §6.2–6.3）。
- 说明：`fast_cbf_mpc.py` 仅用于评估；训练仍用可微 `cbf_mpc.py`。
- **两档需求（回应审稿人"50% 不是工况"）**：校准发现单双向走廊里 ORCA
  互惠假设失效 → 基线 CR 有高底(~18-40%)，**不存在既有流量又 5-10% 的工况**
  (arrival<0.04 走廊几乎空、n<60、std±15-19%)。故报**两档**：低载 0.06
  (ORCA~34%) 与高载 0.16 (ORCA~50%)。两档**同一非单调形状**(中段凸起)、
  装备机在各混合 p 都比未装备机安全 → 混合车队协调成本非高载伪影。
  低载结果 `PENETRATION_LOW.txt`。
  ```bash
  python3 run_penetration.py --reps 6 --horizon 400 --warmup 100 \
      --arrival 0.06 --ps 0,25,50,75,100 --stage2 stage2_final.pt --seed 12345 \
      --out PENETRATION_LOW.txt
  ```

---

### 5.10 探测时域（lead-time）扫描 —— 新增

**目的（回应审稿人"actuation-limited 是性质还是伪影"）**：审稿人质疑
"残余冲突全是机动受限、预测无杠杆"可能只是 encounter 把冲突设计得太晚
（CPA 前约 1s）的产物——那个时域下任何预测器都无力回天。为区分"性质"与
"伪影"，扫描**探测时域**：CPA 前多少秒冲突进入规划窗口。

关键设计（`eval_leadtime.py`，`LeadTimeEncounters` 子类）：
- 用参数 `t_cpa` 控制 CPA 在 rollout 中的步位，**几何其余部分完全固定**——
  未规避 miss 距离在各时域恒为 mean 18.3m（97% <30m），即冲突同等严重，
  只有"可探测提前量"在变，这是干净的受控实验；
- 每档时域跑三种预测条件：Stage-1、Stage-2、**Oracle（喂真值邻居未来）**。
  Oracle 是关键对照——它是任何预测器能达到的 CR 下界，Oracle 仍冲突处即
  预测无杠杆；
- 冲突归因：Oracle 也冲突=机动受限；Oracle 避开而 S2 没避=预测受限。
- 用 `fast_cbf_mpc.py`（OSQP，eval-only）；tuned planner。

```bash
python3 eval_leadtime.py --n 200 --seed 12345 \
    --stage1 stage1_full.pt --stage2 stage2_final.pt --horizons 1,2,3,6,10,20
# 结果写入 LEADTIME.txt；4090 上约 30–50 min（长时域 episode 更长更慢）
```
- **诚实结论**：
  - 1–2s 战术末端：**Oracle CR ≈ S2 CR**（82.0 vs 82.5；11.0 vs 11.0），
    95–98% 机动受限 → **审稿人对：该体制确是纯机动竞赛,非伪影,但体制特定**；
  - 时域 ≥3s：所有带 CBF 方法 + Oracle 的 CR 都塌到 ~0 → **CR 主导变量是
    探测提前量,不是预测精度**（强化"权限治理"论点）；
  - **6s 峰值机制（回应审稿人第二条,已细扫核实）**：Stage-1 CR 是一个
    **宽峰**,不是单调上升,也不是精确 6s 尖峰。细扫（`--horizons 4,5,6,7,8`
    → `LEADTIME_FINE.txt`）:
    `3s=1.5 → 4s=7.0 → 5s=10.0 → 6s=12.0(峰) → 7s=8.5 → 10s=0.5 → 20s=0.0`。
    这**否证了"误差随时域累积"**（那会预测 10s/20s 最差,与数据相反）。
    真实机制:Oracle 与 Stage-2 在 ≥3s 全 0%,故此峰不是几何冲突,而是
    Stage-1 的 OOD 预测误差**诱发的虚假规避机动**;它在 5–6s 最大——此时
    坏预测已可被规划器执行,却还没被足够多轮 receding-horizon 重规划纠正。
    ≥10s 时探测过早,多轮重规划把坏预测洗掉;3s 时 encounter 太短,机动
    来不及破坏间隔。任务对齐(Stage-2)全程 0%,彻底消除该峰。

### 5.11 统计严谨性:配对检验 + 多 seed —— 新增

**目的（回应审稿人第五条）**：论文原来单 seed、无置信区间,还给 1-episode
的 CR 差异加粗排名。补两件事:

**(a) 配对 McNemar + Wilson CI**（`baselines/common/stats_tests.py`）:
CR 是同批 encounter 上的 per-episode 0/1,是配对设计。脚本收集 4 个带 CBF
方法的逐 episode 冲突向量,报每个 CR 的 Wilson 95% CI + PlanGrad 对其余
方法的 McNemar 精确检验。
```bash
python3 baselines/common/stats_tests.py --n 200   # -> STATS.txt
```
结果:CI 全重叠(PlanGrad [7.4,16.1] vs Conformal [7.8,16.7] 等),McNemar
p=1.0/0.25/0.50 全不显著 → **CBF 方法 CR 统计不可区分,是"检验过"而非断言**;
也说明 11.0 vs 11.5 这种差异不该加粗为排名。

**(b) Stage-2 多 seed**（`train_seeds.sh` + tuned-planner eval）:
用同超参、seed 1/2/3 重训 Stage-2,tuned planner 评估。
```bash
cd plangrad_sim && ./train_seeds.sh    # -> stage2_seed{1,2,3}.pt
```
结果:**CR = 11.4 ± 0.4%**(对训练随机性极稳),**ADE = 6.3 ± 1.4m**(有方差,
但最差 seed 7.93m 仍 << Stage-1 的 20.90m → ADE 大幅下降结论对 seed 稳健)。

### 5.12 域适应对照（Stage-1b）+ RQ5 方向分解 —— 新增（决定性一条）

**目的（回应审稿人最关键一条,已接受）**："任务对齐把 ADE 砍 80%" 的卖点
被**域适应**混淆了:Stage-2 见过闭环 encounter 数据,而 Stage-1 从没见过。
必须做一个**只差损失函数**的干净对照。

**Stage-1b 对照**:在与 Stage-2 **完全相同**的闭环 encounter 流上微调
Stage-1（同 pool / 同 seed / iters=50 / lr=1e-4 / batch=16 / T=20）,但损失
是**纯 `L_ADE`**（不经过 planner）。两个训练脚本都用同一 `set_seed` +
`GUAMEncounters(range(2500),seed).sample()` → 同一批 encounter,唯一差别是
损失 → 干净归因。
```bash
cd plangrad_sim
python3 train_stage1b.py            # -> stage1b_domainadapt.pt（纯 L_ADE 微调）
python3 rq5_error_profile.py        # -> RQ5_PROFILE.txt（临界/惰性区 ADE 剖面）
python3 diag_error_direction.py     # -> ERRDIR.txt（沿 ego→neighbour 轴分解）
```

**结论 = Outcome B（任务对齐在每条轴上打平或输给域适应）**:
- 全维打平/更差:Stage-1b ADE=**1.84m**(< Stage-2 的 4.32m),CR=11.5%(vs
  11.0%),MinSep=47.8,LeadT=0.037,Energy=52.9 —— 全部打平;
- 5 个失配体制全打平(S1b/S2):nominal 11.5/11.0,mass+20 17.0/17.5,
  thrust 16.0/16.5,actuator2 30.0/28.0,combined 31.0/30.0;
- 6s 峰在 Stage-1b 也消失(3/4/5s = 0.0/0.0/0.5)→ 消峰是"见过 encounter
  分布",不是任务对齐特有。

**RQ5 升级（把"假设被否证"升级为"否证 + 我们指出梯度去哪了"）**:
方向分解 —— 把误差投影到 ego→neighbour 轴。
- 权威剖面（n=200 exact）:Stage-1 13.53/23.14(ratio 0.585),
  Stage-1b 1.04/2.08(0.499),Stage-2 4.07/4.40(**0.925**);
  **注意:旧论文 RQ5 表里 Stage-2 是 STALE 的 7.02/10.89,已作废刷新**;
- Stage-2 临界 e_par=**−0.34m(t 检验 p=0.023 显著),54.6% 指向 ego
  (binomial p=5.6e-4,N=1400)** = 弱保守偏置(学到的安全裕度,比 Conformal
  的 18.6m 小约 50×);Stage-1b −0.18m(p=0.14 不显著),45.4%;
- 离轴 |e_perp|:Stage-2=2.91m vs Stage-1b=0.68m;
- **方向性地证实 gradient-support Prop**:梯度只沿约束法向(along-axis)到达
  绑定约束处;certificate 把偏置和离轴误差都吸收 → 净操作增益为零。

> ⚠️ **关键采样 bug（必读,否则所有表对不上）**:eval **必须用 batch=8**,
> 使 `n//batch` = 恰好 200(batch=16 → 只有 192,丢尾巴 → ADE 4.07 vs 真值
> 4.32)。`baselines/common/eval_common.py:134` 已加 `assert n%batch==0`
> 硬拦截。任何新 eval 脚本都要遵守:n 必须是 batch 的整数倍,且所有上报数
> 一律 n=200 / seed=12345,保证跨表一致。

---

## 文件产出清单

| 文件 | 来源 | 说明 |
|------|------|------|
| `stage1_full.pt` | 阶段 3 | Stage-1 预测器权重 |
| `stage2_full.pt` | 阶段 4 | Stage-2 TASL 微调后权重 |
| `FINAL2.txt` | 5.2 | 大规模 Stage-1 vs Stage-2 对比 |
| `BEST.txt` | 5.3 | 最优规划器配置下对比 |
| `DIAG.txt` | 5.4 | 冲突归因诊断 |
| `PLANNER_SCAN.txt` | 5.6 | 规划器参数扫描结果 |
| `SCAN_RESULT.txt` | 5.7 | 训练池规模扫描结果 |
| `MISMATCH.txt` | 5.8 | 训练/评估模型失配鲁棒性评估结果 |
| `PENETRATION.txt` | 5.9 | 多智能体渗透率(高载 0.16)结果 |
| `PENETRATION_LOW.txt` | 5.9 | 多智能体渗透率(低载 0.06)结果 |
| `LEADTIME.txt` | 5.10 | 探测时域扫描结果（机动 vs 预测受限归因） |
| `LEADTIME_FINE.txt` | 5.10 | 细扫 4/5/6/7s,确认 6s 宽峰(否证"误差累积") |
| `STATS.txt` | 5.11a | 配对 McNemar 检验 + Wilson CI |
| `stage2_seed{1,2,3}.pt` | 5.11b | 多 seed 重训的 Stage-2 权重 |
| `stage1_conv.log` | 5.6 | Stage-1 收敛曲线(证明已收敛,非欠拟合) |
| `stage1b_domainadapt.pt` | 5.12 | 域适应对照权重(纯 L_ADE,同 encounter 流) |
| `RQ5_PROFILE.txt` | 5.12 | 临界/惰性区 ADE 剖面(S1/S1b/S2,ratio 0.585→0.925) |
| `ERRDIR.txt` | 5.12 | 沿 ego→neighbour 轴的方向分解(e_par/e_perp) |
| `MISMATCH_S1B_LEAN.txt` | 5.12 | Stage-1b 的 5 体制失配对照(全打平) |
| `LEADTIME_S1B.txt` | 5.12 | Stage-1b 的 6s 峰检查(也无峰) |

## 常见问题

- **QP solver 报错频繁**：正常，Stage-2 训练中偶发。若 >50% 步被 SKIP，
  检查 `--beta` 是否过大（默认 0.3），或降低 `--w_coll`。
- **Stage-2 冲突率反而变高**：`--w_ade` 太小导致预测器崩溃。
  调回 0.12–0.3 区间。
- **CUDA out of memory**：减少 `--batch`（Stage-1 512→256，Stage-2 16→8）。
- **`No such file: Data_Set_1.mat`**：回到阶段 1，确认 GUAM 仓库克隆成功，
  或 `export GUAM_MAT=/abs/path/Data_Set_1.mat`。
- **eval 结果波动大**：冲突率方差高，增大 `--n_eval`（建议 ≥200）。
- **ADE/CR 和论文表对不上（如 ADE=4.07 而非 4.32）**：几乎一定是 batch
  不是 8 导致 `n//batch` 丢尾（batch=16,n=200 → 只跑 192）。用 batch=8,
  n=200；`eval_common.py:134` 的 `assert n%batch==0` 会拦住非整除的 n。
- **想复现 RQ5 方向分解**：先 `train_stage1b.py` 出 `stage1b_domainadapt.pt`，
  再 `rq5_error_profile.py` / `diag_error_direction.py`；三者都用 batch=8。