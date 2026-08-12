# 模型清单与复现手册 — PlanGrad-UAV (TRC 投稿)

本文件回答两个问题:**论文用了哪些训练好的模型、每个拿来干什么**,
以及**如何从零把它们重新训练出来并复现论文数字**。

编写方式与 Round1 一致:**每一条都在 lab 上对实际文件核对过**,
包括逐权重的 MD5、`state_dict` 形状、以及权重间的 L2 距离(用于确认血缘)。
凡是文档之间互相矛盾的地方,本文件给出**以代码为准的判定**并标注出来。

- 权重权威副本:`Round1/04_weights/`(9 个 `.pt`)
- 训练脚本:`code/plangrad_sim/train_stage1.py` / `train_stage1b.py` / `train_stage2.py`
- 唯一评分真理源:`code/baselines/common/eval_common.py`
- 补充实验:`Round2/01_new_experiments/`

> **Round1 已有权重用途表**(`Round1/README_ROUND1.md` §4)。本文件不重复它,
> 而是补上 Round1 缺的东西:**逐权重 MD5、精确复现命令、
> 训练/评估配置的区分、以及三处文档不一致的判定**。

---

## 0. 一页速览

| 权重 | 是什么 | 论文角色 | 训练配置(γ, H_p, a_max) |
|---|---|---|---|
| `stage1_full.pt` ★ | Stage-1 位移预训练 | Fixed-Predictor 臂;所有后续初始化 | 无规划器(纯监督) |
| `stage1b_domainadapt.pt` ★ | 域适应对照(仅 ADE) | **RQ5 决定性对照** | 无规划器(纯监督) |
| `stage2_final.pt` ★ | Stage-2 任务对齐 = **PlanGrad 主模型** | 主参照臂 | **(0.4, 8, 10)** |
| `stage2_seed1/2/3.pt` ★ | 换种子重训 ×3 | 训练随机性统计 | (0.4, 8, 10) |
| `stage2_matched.pt` ★ | 匹配部署配置重训 | 跨规划器迁移/RQ5 保险跑 | **(0.1, 15, 20)** |
| `soft_joint.pt` ★ | Soft-IPP 预测器 | Soft-IPP 基线臂 | 软惩罚(无硬 CBF) |
| `stage1_conv_check.pt` | Stage-1 收敛性检查 | 辅助,**不进表** | 无 |

★ = 论文最终使用。共 **6 个进论文**,1 个仅作收敛证据。

**符号对照(极易踩)**:稿件里的 **`γ`** 就是代码里的 **`alpha`**
(`CBFMPCLayer(alpha=...)` / CLI `--alpha`),
见 `Round2/01_new_experiments/results/SEED_ROBUSTNESS.txt:4`。
稿件另有一个 `α^{i,m}` 表示 GMM 混合权重,**与 CBF 的 γ 无关**。
本文件一律用代码名 `--alpha`,并在括号里注明稿件的 γ。

**最关键的一条区分**(整篇论文的核心设计):

- **训练时规划器** `(γ, H_p, a_max) = (0.4, 8, 10)` —— 弱规划器
- **部署/评估规划器** `(γ, H_p, a_max) = (0.1, 15, 20)` —— 强规划器

`stage2_final.pt` 训练于**弱**规划器,评估于**强**规划器。
`stage2_matched.pt` 是唯一训练=评估的那个。这不是笔误,是实验设计。

---

## 1. 权重完整性校验(MD5)

`Round1/04_weights/` 与原始位置(`plangrad_sim/`、`updated_code/`、
`code/baselines/04_soft_ipp/`)**逐字节一致**,已核对:

| 权重 | MD5 | 字节 | 原始位置 |
|---|---|---|---|
| `stage1_full.pt` | `101d7044a76f88bd59b0277bcf539b68` | 1284294 | `plangrad_sim/` |
| `stage1_conv_check.pt` | `813591434d457fc3b4296e9fb7c700c0` | 1284396 | `plangrad_sim/` |
| `stage1b_domainadapt.pt` | `4e3f9462d726f9c1736532042efbd2f0` | 2563534 | `plangrad_sim/` |
| `stage2_final.pt` | `360b187e825cd03a16fcdaa111faa187` | 2563351 | `plangrad_sim/` |
| `stage2_seed1.pt` | `f44b69a718b175c7218730ebef378ed0` | 2563351 | `plangrad_sim/` |
| `stage2_seed2.pt` | `4215fda073f092bfc23cc096a1c63f27` | 2563351 | `plangrad_sim/` |
| `stage2_seed3.pt` | `2aca6b734fec7610fd6656cc16f93b55` | 2563351 | `plangrad_sim/` |
| `stage2_matched.pt` | `183f870e251fae7e94c757309d2e855c` | 2563385 | `updated_code/` |
| `soft_joint.pt` | `0c1937f9ee0f6e59a6f2488ca9d3777a` | 2563317 | `code/baselines/04_soft_ipp/` |

另有 `Round1/05_results/robustness/p0_referee/loose_minsep.pt`
(`ff2a6253c88ae188720d05bb9c39b911`),是**审稿人问询的宽松 d_sep 跑**,
不属预测器权重体系。

校验命令:

```bash
cd /data/lab/TRC_UAV_WEIGHTS
md5sum Round1/04_weights/*.pt
```

**已验证:9 个权重两两均不相同**(无重复文件冒充不同实验)。

---

## 2. 模型架构(从 state_dict 反推,已核对代码)

所有 9 个 `.pt` 都是**裸 `state_dict`**,18 个键,**不含**优化器状态、
epoch 计数或任何元数据。这意味着:**无法从权重本身反查它用什么超参训练**,
只能依赖脚本 + RUNBOOK。这是本项目的一个可复现性弱点,已在 §7 提出改进建议。

`GMMTrajectoryPredictor`(`code/plangrad_sim/predictor.py:14`):

```
in_dim=6, hidden=128, T=30, K=5, n_layers=2, out_dim=3
logsig clamp = [-6.0, 3.0]
参数量 = 319,770
```

| 层 | 形状 | 说明 |
|---|---|---|
| `encoder.*_l0/l1` | (384,6)/(384,128)+… | 2 层 GRU,width 128(384 = 3×128,GRU 三门) |
| `ctx.0` / `ctx.2` | (128,128) ×2 | 两层 MLP + ReLU |
| `head_alpha` | (150,128) | 混合权重,150 = T(30) × K(5) |
| `head_mu` | (450,128) | 均值,450 = 30 × 5 × 3 |
| `head_logsig` | (450,128) | 对角 log 标准差,同上 |

**与稿件的一致性核对**:
- `Draft_v10.tex` 写 "two-layer GRU with hidden width 128" → ✅ 与 `(384,128)`、`n_layers=2` 一致
- 输入 "L×6 sequence"(位置 + 有限差分速度) → ✅ `in_dim=6`
- Eq. (5) 混合分布 `α^{i,m}`、`μ^{i,m}`、对角 `Σ` → ✅ 三头结构一致

⚠️ **注意 T=30 vs T=20**:网络头按 `T=30` 构建(150/450),
但训练与评估都用 `--T 20`。即**预测头输出 30 步,只用前 20 步**。
`Draft_v10.tex` 的 `T` 指 rollout 长度(20),与网络头的 `T=30` **不是同一个量**。
复现时**不要**把 `predictor.py` 的 `T` 改成 20,否则权重形状不匹配、无法加载。

---

## 3. 血缘关系(用权重距离验证,非凭注释)

`train_stage2.py:101` 与 `train_stage1b.py:72` 都执行
`pred.load_state_dict(torch.load(args.stage1))`,即**全部从 Stage-1 分叉**。

用全参数 L2 距离验证(单位:参数空间):

| checkpoint | → `stage1_full` | → `stage1b` |
|---|---|---|
| `stage2_final` | 0.9847 | 0.6097 |
| `stage2_matched` | 0.9951 | 0.7839 |
| `stage2_seed1` | 1.1495 | 0.7915 |
| `stage2_seed2` | 1.0744 | 0.7642 |
| `stage2_seed3` | 1.1319 | 0.8376 |
| `soft_joint` | 0.9092 | 0.4601 |
| `stage1_conv_check` | **40.8010** | 40.7327 |
| `stage1b` | 0.7207 | — |

**判读**:所有微调产物到 `stage1_full` 的距离都在 ~1.0 量级,
而 `stage1_conv_check` 是 **40.8** —— 后者是独立的另一次 Stage-1 运行,
**不是**任何模型的父节点。微调确实只是小幅扰动初始权重。

> **一处我先前的错误判断,在此更正**:我曾只看 `encoder.weight_ih_l0`
> 单层就推断 "Stage-2 从 stage1b 分叉"。看全参数距离 + 读代码
> (`--stage1 stage1_full.pt`)后可确认:**Stage-2 与 Stage-1b 是
> stage1_full 的两个并列分支**,彼此无父子关系。距离表里
> "→stage1b 更近" 只是因为两者都朝相似方向小步移动,不代表血缘。

```
GUAM 轨迹 0–2499
      │
      └─ train_stage1.py ──> stage1_full.pt        (纯位移监督)
                                  │
         ┌────────────┬───────────┼────────────┬──────────────┐
         │ 仅 ADE     │ TASL      │ TASL 软惩罚 │ TASL         │ TASL ×3 seed
         │            │ (0.4,8,10)│            │ (0.1,15,20)  │ (0.4,8,10)
         ▼            ▼           ▼            ▼              ▼
   stage1b_      stage2_final  soft_joint  stage2_matched  stage2_seed1/2/3
   domainadapt   =PlanGrad     =Soft-IPP   (训练=评估)      (随机性统计)
   (RQ5 对照)     (主模型)
```

---

## 4. 环境与数据(复现前提)

已在验证机上实测的版本组合(`python3 -c "import torch,cvxpy…"` 核对):

```
torch 2.11.0+cu128   CUDA 12.8   NVIDIA GeForce RTX 4090
numpy 2.2.6   scipy 1.15.3   h5py 3.16.0
cvxpy 1.7.5   cvxpylayers 0.1.9
```

```bash
# 1) 依赖(机器重置后需重装)
pip install "scipy==1.15.3" "h5py==3.16.0" "cvxpy==1.7.5" "cvxpylayers==0.1.9"

# 2) GUAM 数据(不在仓库里)
git clone --depth 1 https://github.com/nasa/Generic-Urban-Air-Mobility-GUAM.git
export GUAM_MAT=/abs/path/GUAM/Challenge_Problems/Data_Set_1.mat

# 3) 运行路径
cd /data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim
export PYTHONPATH=/data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim:/data/lab/TRC_UAV_WEIGHTS/code/baselines/common
```

**GPU 必需**(`RUNBOOK_main.md:13-14`):可微 QP(cvxpylayers)
CPU 约 36 s/步、GPU 约 0.09 s/步(**约 400×**)。验证机 RTX 4090。
CPU 上跑 Stage-2 或全量评估不现实。
注意纯评估阶段 QP 主要在 CPU 端,4090 上仍需约 40–70 min
(`RUNBOOK_main.md:215`)。

**数据划分(不可重叠)**:
- 训练:GUAM 轨迹 `range(0, 2500)`
- 评估:GUAM 轨迹 `range(2500, 3000)` —— `eval_common.py:65 EVAL_RANGE`
- Conformal 校准:`2500–2999` + **独立种子 777**
  (⚠️ 早期版本误用 `range(2000,2500)`,与训练重叠、违反 split-conformal,**已修复**)

---

## 5. 从零复现:三阶段训练

### 5.1 Stage-1(位移预训练)→ `stage1_full.pt`

```bash
python train_stage1.py --cuda --seed 12345 --yaw_augment \
    --n_traj 2500 --epochs 30 --batch 512 --lr 8e-4 \
    --out stage1_full.pt
```

- 耗时:4090 上约 5 分钟
- 优化器 Adam,`weight_decay=1e-5`,`CosineAnnealingLR(T_max=epochs)`
- 保存策略:**按 `0.5*minADE + 0.5*meanADE` 选最优 epoch**(不是最后一个 epoch)
- 预期 val minADE ≈ 2–4 m,meanADE ≈ 5–8 m
- `--yaw_augment` **必须加**:Stage-2 遭遇场景邻居朝向任意,不增强则跨朝向不鲁棒

### 5.2 Stage-1b(域适应对照)→ `stage1b_domainadapt.pt`

这是**整篇论文最重要的对照臂**:它与 Stage-2 用
**完全相同的遭遇数据、迭代数、学习率、batch、种子**,
唯一区别是**损失里没有任何运行项,只有 ADE**。
因此 Stage-2 相对它的任何增益,都不能归因于"多看了闭环数据"。

```bash
python train_stage1b.py --cuda --seed 12345 \
    --stage1 stage1_full.pt \
    --iters 70 --batch 16 --T 20 --lr 1e-4 \
    --out stage1b_domainadapt.pt
```

代码中三处注释明确标注 `IDENTICAL`(`train_stage1b.py:69,77,83`):
同一 `GUAMEncounters(range(2500), seed=12345)`、同一 Adam/lr、同一采样调用。

### 5.3 Stage-2(任务对齐微调)→ `stage2_final.pt`

```bash
# 注意:此处用 code/plangrad_sim/train_stage2.py(a_max 硬编码 10.0,不要传 --a_max)
python train_stage2.py --cuda --seed 12345 \
    --stage1 stage1_full.pt \
    --iters 70 --batch 16 --T 20 --Hp 8 --lr 1e-4 \
    --alpha 0.4 \
    --w_coll 3.0 --w_lead 0.5 --w_ade 0.12 \
    --out stage2_final.pt
```

> ⚠️ **两个版本的 trainer,差一行,后果不同**(已 diff 核对):
> - `code/plangrad_sim/train_stage2.py:105` → `a_max=10.0` **硬编码**,
>   **无 `--a_max` 参数**。传 `--a_max` 会直接报
>   `unrecognized arguments` 而退出。
> - `updated_code/train_stage2.py:105` → `a_max=args.a_max`,
>   新增 `--a_max`(默认 10.0,**不改变原行为**)。这就是 Round1 README 所说的
>   "patched 版",`stage2_matched.pt` 由它产生。
>
> 复现 `stage2_final` / `seed1-3`(a_max=10)用**任一**版本均可
> (只要不传 `--a_max`);复现 `stage2_matched`(a_max=20)
> **必须**用 `updated_code/` 那份。

- 耗时:4090 上约 15–25 分钟
- 优化器 **只含 `pred.parameters()`** → 规划器配置 φ **全程冻结**(论文的核心设定)
- 梯度裁剪 `clip_grad_norm_(..., 5.0)`
- 偶发 `SKIP (solver issue: ...)` 属正常(QP 不可行时跳过该步)
- 期望 `loss` 下降、`mean_min_sep` 上升

**TASL 损失项与稿件符号对应**(`train_stage2.py` docstring):

| CLI | 值 | 稿件符号 | 含义 |
|---|---|---|---|
| `--w_coll` | 3.0 | `λ_s` | 平滑分离违规(主驱动项) |
| `--w_delay` | 0.05 | `λ_d` | `Φ_dev` 终端横向偏差 |
| `--w_energy` | 0.01 | `λ_u` | `Φ_effort` 控制活动(归一化) |
| `--w_lead` | 0.5 | — | 冲突预警提前量**奖励**(损失中为负号) |
| `--w_ade` | **0.12** | `λ_a` | ADE 锚定,防预测器漂离数据 |
| `--beta` | 0.3 | `β` | logistic 锐度 |

> ⚠️ **三处文档不一致,以下为判定**:
> 1. `train_stage2.py` argparse 默认 `--w_ade 0.3`,但 **docstring 与
>    `Round1/RUNBOOK_main.md:90` 都写 `0.12`**。
>    → **复现请显式传 `--w_ade 0.12`**,不要依赖默认值。
> 2. argparse 默认 `--iters 60`,docstring/RUNBOOK 写 `70`。
>    → **用 70**。
> 3. RUNBOOK 里输出名为 `stage2_full.pt`,而仓库中论文主模型叫
>    `stage2_final.pt`。→ 是同一角色的**不同文件名**,以 `stage2_final.pt` 为准。
>
> `--w_lead` 这一项在 `Round1/README_ROUND1.md` §8 有已知问题说明,
> 复现时请一并阅读。

### 5.4 三个种子(训练随机性)→ `stage2_seed1/2/3.pt`

同 §5.3,仅换 `--seed`。**其余全部参数必须保持一致**,
否则种子离散度会混入超参差异:

```bash
for S in 1 2 3; do
  python train_stage2.py --cuda --seed $S \
      --stage1 stage1_full.pt \
      --iters 70 --batch 16 --T 20 --Hp 8 --lr 1e-4 \
      --alpha 0.4 \
      --w_coll 3.0 --w_lead 0.5 --w_ade 0.12 \
      --out stage2_seed$S.pt
done
```

⚠️ 种子值本身未在 RUNBOOK 中记载,`--seed 1/2/3` 是依文件名的
**合理推断**(argparse 默认为 0,主模型用 12345)。
若复现出的权重 MD5 与 §1 不符,种子编号可能不同,
但**离散度结论**(CR 11.5 ± 0.5)不依赖具体种子值。

> ⚠️ `stage2_final.pt` 是**独立的第四次运行**(种子 12345),
> 与 seed1/2/3 权重互不相同(已 MD5 + L2 距离验证)。
> 它的 ADE(4.32 m)相对三种子(6.96 ± 1.09 m)是**离群值**,
> 但 CR(11.0%)**不离群**(三种子 11.0/11.0/11.5%)。
> 报告时**必须**说明主模型 ADE 偏低是单次运行的波动,
> **不可**据此声称任务对齐训练能改善 ADE。

### 5.5 匹配部署配置 → `stage2_matched.pt`

唯一"训练规划器 = 评估规划器"的检查点,用于跨规划器迁移分析:

```bash
cd /data/lab/TRC_UAV_WEIGHTS/updated_code      # ← 必须用 patched 版
python train_stage2.py --cuda --seed 12345 \
    --stage1 stage1_full.pt \
    --iters 70 --batch 16 --T 20 \
    --Hp 15 --alpha 0.1 --a_max 20.0 \
    --lr 1e-4 --w_coll 3.0 --w_lead 0.5 --w_ade 0.12 \
    --out stage2_matched.pt
```

`--a_max 20.0` 只有 patched trainer 认(见 §5.3 警告)。
配套评估脚本 `updated_code/eval_matched.py`
(另有副本 `Round1/03_verification_runs/eval_matched.py`)。

### 5.6 Soft-IPP 基线 → `soft_joint.pt`

```bash
cd /data/lab/TRC_UAV_WEIGHTS/code/baselines/04_soft_ipp
python train_soft.py   # 软惩罚替代硬 CBF 约束
```

用途:隔离**硬 CBF 证书**的贡献。这是一个**受控消融**——
`train_soft.py:6-17` 明确记载:同一初始化(`stage1_full.pt`)、
同一可微闭环 rollout、同一 `iters/batch/T/Hp/lr/数据池/seed 12345`,
**唯一差别**是回路里的规划器换成软惩罚 Vanilla-MPC(`w_rep` 斥力项)
而非硬 CBF 安全层。损失结构与 TASL 完全同形
(`train_soft.py:90-94`,同样的 `w_coll/w_delay/w_energy/-w_lead/w_ade`)。

### 5.7 零松弛可行性检查(不产生权重)

`ZERO_SLACK_FEAS.txt` 由 `Round1/03_verification_runs/zero_slack_feasibility.py`
生成,回答"冲突是执行器受限还是松弛变量造成的":

```bash
cd /data/lab/TRC_UAV_WEIGHTS/Round1/03_verification_runs
python zero_slack_feasibility.py     # 对 stage2_final.pt 重解 eps=0 QP
```

结论(已核对文件):22 个冲突 episode **全部**含至少一个
`eps=0` 不可行步 → **100.0% 执行器受限**,0 个纯松弛导致。
每冲突不可行步数 mean/max = **18.82 / 20**。
稿件第 1967 行写 18.8,与此一致,**无需改动**。

---

## 6. 复现论文数字

### 6.1 唯一评分真理源

**所有**表格数字必须经 `code/baselines/common/eval_common.py`,
其固定配置(`eval_common.py:65-101`):

```
EVAL_RANGE   = range(2500, 3000)      # held-out,与训练 0-2499 无重叠
BEST_PLANNER = alpha=0.1, horizon=15, a_max=20.0   # 部署规划器
D_SEP = 30 m,  dt = 0.2,  T = 20,  n = 200
GLOBAL_SEED = 12345,  WIND_SEED = 7,  eta_w * gust_std = 0.3 * 3.0
```

**铁律 `n % batch == 0`**:`eval_common.py:140` 硬编码 `batch = 8`,
`n=200` → 恰好 25 个 batch、200 条。`eval_common.py:145` 有 `assert`
强制这一点,注释原文说明理由:`n//batch` 截断会**静默丢掉尾部遭遇**,
从而改变 ADE/CR 并破坏跨表一致性。复现时不要改 `batch`。

冲突定义:**per-episode** —— 一条 episode 内**任一时刻**最小间隔
< 30 m 即计为冲突(不是逐步计数)。

### 6.2 主对比

```bash
cd /data/lab/TRC_UAV_WEIGHTS/code/plangrad_sim
python eval_stage1_vs_stage2.py --cuda --seed 12345 \
    --stage1 stage1_full.pt --stage2 stage2_final.pt --n_eval 200
```

部署规划器下的预期值(已在 Round2 补充实验中精确复现):

| 检查点 | CR | ADE | MinSep | Effort |
|---|---|---|---|---|
| `stage2_final` | **11.0%** (22/200) | 4.321 m | 47.74 m | 52.349 |
| `stage1b_domainadapt` | **11.5%** (23/200) | 1.839 m | 47.79 m | 52.898 |
| `stage1_full` | 12.5% | 20.90 m | — | — |

三种子(`SEED_ROBUSTNESS.txt`,n=3, ddof=1):

| seed | CR | ADE | MinSep |
|---|---|---|---|
| seed1 | 12.0% (24) | 7.930 m | 47.92 m |
| seed2 | 11.0% (22) | 5.774 m | 47.36 m |
| seed3 | 11.5% (23) | 7.159 m | 48.29 m |
| **mean ± SD** | **11.5 ± 0.5** | **6.955 ± 1.092** | 47.854 ± 0.466 |

⚠️ `RUNBOOK_main.md:336` 写的 "CR 11.4 ± 0.4 / ADE 6.3 ± 1.4"
用的是 **SD 而非 SEM**,且均值与实测(11.500 / 6.955)略有出入。
**以本表为准**。三个 McNemar 检验 p 全为 1.0000,
且**都对同一个 Stage-1b 向量**做配对 → **三者不独立**,
只能回答"单个种子是否可与对照区分",不能回答"种子之间是否有差异"。

**这就是论文的否定结果**:任务对齐训练相对纯域适应,
在部署规划器下**无可测量的安全增量**(11.0% vs 11.5%,
McNemar p = 1.0)。ADE 相差 2.3 倍而 CR 几乎相同 —— 即
**开环精度与闭环安全解耦**。

### 6.3 Round2 四项补充实验

脚本在 `Round2/01_new_experiments/scripts/`,结果在 `../results/`。
每个脚本都把**断言与控制项校验写进输出文件**,
因此图/表不可能与数据反向发布。

脚本在 `code/baselines/figures_gen/`,产出 5 个结果文件:

| 结果文件 | 生成脚本 |
|---|---|
| `SEED_ROBUSTNESS.txt` | `export_seed_robustness.py` |
| `LEADTIME_HP_SENS.txt` | `export_leadtime_hp.py` |
| `WEAK_2X2_MATCH.txt` | `export_weak_2x2.py` |
| `PLANNER_TRANSFER.txt` | `export_planner_transfer.py` |
| `ZERO_SLACK_FEAS.txt` | 零松弛可行性检查(见 §8 注) |

```bash
export FIG_DATA_DIR=/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data
cd /data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen

nohup python export_seed_robustness.py  > /tmp/seed.log  2>&1 &
nohup python export_leadtime_hp.py      > /tmp/lt.log    2>&1 &   # 最慢
nohup python export_weak_2x2.py         > /tmp/weak.log  2>&1 &
nohup python export_planner_transfer.py > /tmp/tr.log    2>&1 &
```

⚠️ **必须后台运行并轮询**,前台调用会超时(~120 s 上限)。
`export_leadtime_hp.py` 全表 18 格(3 个 H_p × 6 个 lead time)
**累计约 146 分钟**(4090)。结果文件方括号里的数字是**累计耗时**,
不是单格耗时 —— 单格从 1.2 min(h=1, H_p=8)递增到约 26 min
(h=7, H_p=25),因为 `T = t_cpa + H_p + 2` 随两者同时增长。

复现判据(控制项必须精确命中,否则环境有问题):

| 实验 | 控制项 | 必须等于 |
|---|---|---|
| 种子稳健性 | Stage-2 / Stage-1b 部署 CR | 11.0% / 11.5% |
| 规划时域 | H_p=15 一列 | 已发表 lead-time 曲线 |
| 弱 2×2 | Stage-1b / primary 弱配置 CR | 56.0% / 28.5% |
| 跨规划器 | 训练时 CR 56.5/56.0/28.5;ADE 20.90/1.84/4.32 | 全部命中 |

---

## 7. 复现性弱点(如实列出)

1. **权重不含元数据**。9 个 `.pt` 都是裸 `state_dict`,无优化器状态、
   无 epoch、无超参、无 git commit。**无法从权重反查其训练配置**,
   只能信脚本 + RUNBOOK,而后两者已发现 3 处不一致(§5.3)。
   → 建议后续训练改存 `{'state_dict':…, 'args':vars(args), 'commit':…}`。
2. **`--w_ade` 默认值与实际用值不同**(0.3 vs 0.12)。
   照默认值跑会得到**另一个模型**。已在 §5.3 显式标注。
   同理 `--iters` 默认 60 但实用 70。**两者都必须显式传参**。
3. **同名 trainer 有两个版本**(`code/plangrad_sim/` vs `updated_code/`),
   仅差 `a_max` 是否可配。`git` 里两份都在、无版本标记,
   靠路径区分极易搞错。→ 建议合并为一份并保留 `--a_max`。
 4. **三个 seed 的具体数值未记载**(§5.4),只能从文件名推断。
 5. **`stage2_final.pt` 无法严格重现**:它是第四次独立运行,
    虽然种子 12345 已固定,但 cvxpylayers 的 QP 求解在 GPU 上
    存在非确定性,且 `SKIP (solver issue)` 的发生位置会随之变化。
    `eval_common.py:34` 只承诺"同一 GPU + 同一库版本下"可复现。
 6. **`T=30` 头 vs `T=20` rollout** 的双重含义(§2)容易误改。
 7. **GUAM 数据不在仓库**,需外部 clone;若 NASA 上游改动数据集,
    数值可能漂移。建议记录所用 commit。

---

## 8. 一条最容易踩的表述陷阱

`Round2/01_new_experiments/results/WEAK_2X2_MATCH.txt` 与
`PLANNER_TRANSFER.txt` 共同给出:

- 同配置族(训练=评估=弱规划器)27.9 ± 0.7%(n=4),**全部严格嵌套**
- `stage2_matched.pt`(训练=评估=**强**规划器)在弱规划器下 **69.0%**,
  **比 Stage-1b(56.0%)更差**,+13.0 pp,p = 1.06e-4,**非嵌套**
- 检查点间离散度:部署规划器下 **2.5 pp** vs 训练规划器下 **42.0 pp**(17×)

**因此**:
- ❌ 不可写"弱规划恢复了任务对齐的杠杆"
- ❌ 不可画/暗示"各检查点在自己训练的规划器下最好"的对角结构
  —— `stage2_final` 在它**未**受训的强规划器下反而绝对更好(11.0% vs 28.5%)
- ✅ 只能写**非对称交互**:部署规划器**压缩**预测器差异,
  训练规划器**暴露**它们
- ✅ ADE 不能解释该分歧(该区间精度跨 11× 而结果几乎不变),
  只能写 "not explained by ADE alone and consistent with
  planner-specific adaptation",**不得**当作证明

---

## 9. 自检清单(复现后逐项核对)

```bash
cd /data/lab/TRC_UAV_WEIGHTS

# 1) 权重完整性
md5sum Round1/04_weights/*.pt

# 2) 架构与参数量(应为 319,770)
python3 -c "
import torch; d=torch.load('Round1/04_weights/stage2_final.pt',map_location='cpu')
print('keys',len(d),'params',sum(v.numel() for v in d.values()))"

# 3) 数据划分不重叠(应为 range(2500,3000))
grep -n 'EVAL_RANGE' code/baselines/common/eval_common.py

# 4) 规划器配置(应为 alpha=0.1 horizon=15 a_max=20)
grep -n 'BEST_PLANNER' code/baselines/common/eval_common.py

# 5) 补充实验的控制项
grep -n 'control' Round2/01_new_experiments/results/*.txt
```

第 5 项应命中 4 个文件(`LEADTIME_HP_SENS` 1 处、`PLANNER_TRANSFER` 2 处、
`SEED_ROBUSTNESS` 4 处、`WEAK_2X2_MATCH` 1 处)。
`ZERO_SLACK_FEAS.txt` **没有** `control` 行 —— 它不是对照实验,
而是对已发表数字的直接重算,自检看它最后一行是否为
`=> EVERY conflict has an eps=0-infeasible step`。

上述 5 项已在 lab 上逐条实测通过(§1 MD5、参数量 319770、
`range(2500,3000)`、`alpha=0.1 horizon=15 a_max=20.0`)。
若 `control` 行与已发表值不符,**先排查环境**
(GPU 型号、cvxpy/cvxpylayers 版本、是否误改 `batch`),
再怀疑数据。
