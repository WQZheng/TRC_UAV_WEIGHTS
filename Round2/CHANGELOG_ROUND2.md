# Round2 变更与验证记录 — PlanGrad-UAV (TRC 投稿)

本文件逐项记录 Round1 之后所做的**全部改动与验证**，并对每一条给出
**可复核的证据来源**（文件路径、行号、DOI、或 lab 结果文件）。

编写原则与 Round1 一致：**只写数据支持的结论**；凡是无法验证的，明确
标注为"未验证"或"已删除"，不做修饰。凡是我发现自己或工具出错的地方，
一律写入本文件，不隐去。

- 基线版本：`Draft_v9.tex`
- 本轮产出：`Draft_v10.tex`（3415 行，46 个被引 key）
- 参考文献：`refs.bib`（65 条）
- v9 → v10 差异：66 行（`diff` 计数）

---

## 0. 一页速览：本轮到底改了什么

| # | 事项 | 结果 | 严重度 |
|---|---|---|---|
| 1 | 未解析引用 | **59 → 0** | 阻断编译 |
| 2 | 键名漂移（引用了不存在的 key，但论文在库中另有其名） | 修正 **12** 处 | 阻断编译 |
| 3 | 新增并核验参考文献 | **17** 条，全部经 CrossRef 逐字段核对 | 高 |
| 4 | **发现 4 条编造的参考文献** | 已删除并替换为真实文献 | **最高（学术诚信）** |
| 5 | **发现 `ref_verify` 工具把一条引用"修正"成了另一篇论文** | 已人工纠正 DOI | **最高** |
| 6 | 补充指定期刊（CTR × 2、JICV × 2） | 已加入并改写正文描述 | 中 |
| 7 | Figure 1 框架图 | 修正 4 处与正文不符的标注 | 高 |
| 8 | Round1 遗留的 5 项未提交实验产物 | 已归档进 Round2 并提交 | 中 |

---

## 1. 参考文献体系（本轮最大的问题）

### 1.1 初始状态

`Draft_v9.tex` 引用 44 个 key，编译日志报
`Latex failed to resolve 59 citation(s)`。经比对 `refs.bib`（48 条），
**29 个被引 key 在库中不存在**。

> **纠正一处此前的错误判断**：先前的排查曾把这 29 个 key 全部归为
> "库中彻底缺失"。这是错的。实际逐一比对后发现，其中相当一部分论文
> **确实在库里，只是 key 写错了**。下面 §1.2 就是这部分。

### 1.2 键名漂移（12 处，逐一核对同一篇论文后改名）

改的是**正文里的 key**，指向 `refs.bib` 中已存在且经核对为同一篇论文的条目：

| 正文原 key（不存在） | 改为（库中真实条目） | 核对依据 |
|---|---|---|
| `coppola2024urban` | `coppola2024uam` | Coppola 等，*Transport Policy*，UAM airport shuttles or city-taxis |
| `mandi2024decision` | `mandi2024decisionfocused` | Mandi 等，*JAIR*，Decision-Focused Learning 综述 |
| `brunke2021safe` | `brunke2022safe` | Brunke 等，*Annual Review of Control*，Safe Learning in Robotics |
| `emam2022safe` | `emam2022sablas` | Emam 等，*RA-L*，Robust CBF + RL |
| `rezaee2024comprehensive` | `rezaee2024collision` | Rezaee 等，*T-ITS*，无人机避撞综述 |
| `zhou2024enhancing` | `zhou2024platoon` | Zhou/Yan/Yang，*T-IV*，mixed-autonomy platoon 安全 RL |
| `pak2025can` | `pak2024uam` | Pak 等，*CEAS Aero. J.*，Can UAM become reality |
| `wang2023quasi` | `wang2023quasidynamic` | Wang 等，*TR-C*，quasi-dynamic 空中交通分配 |
| `wang2023urban` | `wang2023uam` | Wang/Li/Qu，*The Innovation*，urban aerial mobility |
| `acheson2025generic` | `acheson2024guam` | NASA GUAM 仿真器 v1.1 |
| `campbell2025systems` | `campbell2024pyguam` | Campbell 等，AIAA SciTech，GUAM AI 集成 |
| `tang2023incorporating` | `tang2023strategic` | Tang/Xu，*T-ITS*，UTM 战略冲突解脱优化 |

后 4 条（`wang2023urban`、`acheson2025generic`、`campbell2025systems`、
`tang2023incorporating`）是此前排查**遗漏**的漂移项，本轮补上。

### 1.3 新增并核验的 17 条文献

全部**先用 CrossRef API 按标题检索确认存在**，再逐字段（标题/作者全名/
期刊/年/卷/期/页/DOI）核对后写入 `refs.bib`。**不使用记忆撰写条目。**

`kuchar2000review`、`amos2017optnet`、`elmachtoub2022smart`、
`talebpour2016influence`、`stern2018dissipation`、`loquercio2021learning`、
`wang2017safety`、`bauranov2021designing`、`pang2021data`、
`yang2021autonomous`、`reiche2021initial`、`chen2024integrated`、
`yang2020scalable`、`zhao2024clustering`、`mohammadian2023continuum`、
`sheng2024kinematics`、`an2025longterm`

抽样核对记录（CrossRef 返回值）：

- `kuchar2000review` → DOI `10.1109/6979.898217`，*T-ITS* 1(4):179–189，
  作者 J.K. Kuchar; L.C. Yang ✓
- `stern2018dissipation` → DOI `10.1016/j.trc.2018.02.005`，*TR-C* 89:205–221；
  **作者名修正**：第 8 作者应为 `R'mani Haulcy`（我初稿误写 `R'aphael Haulcy`）✓
- `elmachtoub2022smart` → DOI `10.1287/mnsc.2020.3922`，*Management Science*
  68(1):9–26 ✓
- `talebpour2016influence` → DOI `10.1016/j.trc.2016.07.007`，*TR-C* 71:143–163 ✓

---

## 2. 两个必须记录的严重问题

### 2.1 发现 4 条编造的参考文献（已删除）

以下 4 个 key 在 `Draft_v9.tex` 中被引用，但**在 CrossRef / OpenAlex 中
以任何合理检索式都无法找到对应论文**。判断为前序环节凭记忆编造：

| 编造的 key | 原文中承担的说法 | 处置 |
|---|---|---|
| `xi2025accurate` | "面向 UAM 交通管理架构的学习型预测" | **删除**，替换为 `kim2025dlcollision` |
| `cho2026toward` | 同上（并列引用） | **删除**，替换为 `wen2025bidgcnllm` |
| `inbaraj2026physics` | 同上（并列引用） | **删除** |
| `al2022experimental` | "城市气流与天气扰动随建成环境变化" | **删除**，替换为 `nithya2024wind` |

替换所用文献均为**已在 `refs.bib` 中且经 CrossRef 核实**者：

- `kim2025dlcollision` → *Drones* 9(7):460，DOI `10.3390/drones9070460` ✓
- `wen2025bidgcnllm` → *Drones* 9(7):508，DOI `10.3390/drones9070508` ✓
- `nithya2024wind` → *Drones* 8(4):147，DOI `10.3390/drones8040147` ✓
- `schweiger2023wind` → *Drones* 7(7):464，DOI `10.3390/drones7070464` ✓

**并且**：正文句子是**按替换后论文的真实内容重写的**，不是把新 key 塞进
旧句子。详见 §3.1。

另有 `wu2022safety` 亦无法核实，同样删除，改引 `yang2020scalable`
（Yang & Wei，*JGCD* 43(8):1473–1486，DOI `10.2514/1.G005000`，已核实），
并把正文说法从"环境不确定性下的显式间隔保证"改为该文实际支持的
"可扩展到多机、并带显式间隔保证"。

### 2.2 `ref_verify` 工具把一条引用"修正"成了另一篇论文

这一条尤其需要留档，因为它说明**不能盲信自动核验工具的 `correctedBib`**。

- 我为 `yang2021autonomous` 初填 DOI `10.1109/TITS.2021.3052229`。
- `ref_verify` 报告该条 "Corrected from CrossRef — fixed: `title`, `author`,
  `pages`, `publisher`"，即它**接受了这个 DOI 并据此改写了标题和作者**。
- 我直接向 CrossRef 查询该 DOI，实际返回：
  **"A Traffic Demand Analysis Method for Urban Air Mobility",
  Bulusu; Onat; Sengupta; Yedavalli; Macfarlane** — 与目标论文
  （Yang & Wei，*Autonomous Free Flight Operations...*）**完全是两篇不同的论文**。
- 正确 DOI 经标题检索确认为 **`10.1109/TITS.2020.3048360`**，
  *T-ITS* 22(9):5962–5975，作者 Xuxi Yang; Peng Wei。

**结论**：若当时把 `metadata.correctedBib` 直接写回，就会发表一条
指向错误论文的引用。本轮所有 DOI 均**另行独立向 CrossRef 复核**过。

`ref_verify` 汇总为 "11 entries: 0 verified, 11 corrected, 0 UNVERIFIED"，
即它**没有**把任何一条标为不可验证——这正是风险所在。

---

## 3. 正文改动（Related Work 及相关段落）

### 3.1 低空预测段（约 line 209–214）

**改前**（引用 3 条编造文献）：

> Recent UAM-specific studies have begun to narrow the gap, including
> learning-based prediction designed for future UAM traffic-management
> architectures~[xi2025accurate, cho2026toward, inbaraj2026physics], but an
> interaction-aware, multi-agent prediction literature ... has yet to form.

**改后**（按 4 篇真实论文各自的实际内容重写）：

> ... including deep-learning frameworks that pair trajectory prediction with
> downstream collision prediction~[kim2025dlcollision] and graph-based models
> that forecast drone states from Remote ID data to support separation
> monitoring~[wen2025bidgcnllm]. Road traffic, by contrast, already supports a
> mature interaction-aware multi-agent prediction literature, in which
> heterogeneous road users are modelled jointly through graph attention over
> kinematically consistent representations~[sheng2024kinematics] and
> long-horizon forecasts are grounded in recurring vehicle-following behaviour
> patterns~[an2025longterm]. An equivalent body of work has yet to form for
> low-altitude traffic.

这样处理同时达成三件事：去掉编造引用、补入 JICV 两篇、
并让原本**无引用支撑**的"道路交通已有成熟文献"这一对比句获得实证依据。

### 3.2 战术层段（line ~270）

`wu2022safety`（无法核实）→ `yang2020scalable`，说法同步改为
"scale to many agents while carrying explicit separation assurance"。

### 3.3 eVTOL 扰动段（line ~309–311）

`al2022experimental` → `nithya2024wind`，并补一句由
`schweiger2023wind` 支撑的、可核查的具体表述（风况已被证明会约束
eVTOL 运行与 vertiport 交通流）。

### 3.4 混合自主段（line ~338，CTR 两篇的落点）

在 `talebpour2016influence` / `stern2018dissipation` 之后新增两句：

- `zhao2024clustering`（CTR 4:100151）：混合车流的结果不仅取决于自动
  车**比例**，还取决于其**空间聚集**方式，聚集倾向会改变给定渗透率下
  可达到的通行能力。
- `mohammadian2023continuum`（CTR 3:100107）：连续介质交通流模型综述
  强调，当新的受控车辆类别进入车流时，宏观模型**必须重新检验而非直接沿用**。

这两句与本文"不预设地面交通结论可以外推到低空"的立场一致，
并为 §5.6 的混合装备走廊实验提供了文献依据。

### 3.5 Introduction（line 120）

`al2022experimental` → `nithya2024wind`（同 §3.3 原因）。

---

## 4. Figure 1 框架图

依据 `Draft_v10.tex` 的 §4 正文与 lab 代码核对用户手绘版本，
发现 **4 处与正文不一致的标注**，已在新图中修正：

| 手绘图中 | 正确 | 依据 |
|---|---|---|
| `d_safe` | `d_sep` | 全文 `d_sep` 出现 19 次，`d_safe` 0 次 |
| `π^i_{m,k}`（混合权重） | `α^{i,m}_{t+k\|t}` | Eq. (5) `eq:gmm-decoder` |
| "route deviation (RMS / cross-track)" | **terminal lateral deviation** | `eq:phi-dev`：`P_⊥` 作用于 `T_end` 终端状态，非时间平均 |
| "effort (energy / control)" | **control activity** | 正文原话："a proxy for control activity rather than a physical energy measurement" |

> 第 3 条尤其重要：这正是 v9 中**已经修正过一次**的错误说法，
> 手绘图把它又带回来了。若随图发表会与正文自相矛盾。

图面设计上的调整（用户要求"少公式、少文字、不要像 AI 画的"）：

- 删除 *Operational quantities* 中的图标条（警示三角/双人像/仪表盘/箭头爆炸）
  ——改为直接排版 `Φ_los`、`d_min`、`Φ_effort`、`Φ_dev`，
  既去掉 AI 观感，也让图与公式可交叉对照。
- 删除 3-D 商务喷气机剪贴画（且**机型与本文 eVTOL 设定不符**），
  改为 `x_b/y_b/z_b` 三轴。
- 预测器由"3 行 × 5 格 GRU + 3 条高斯曲线 + 柱状图 + 协方差椭圆"
  简化为"3 格 GRU 图元 + 一行混合分布公式"。
- QP 框内展开的双积分递推式改为 `see Eq. (12)` 指针。
- 5 个彩色列首胶囊 → 全图仅 2 个强调色。
- **修正一处逻辑错误**：原图梯度虚线穿过 split-conformal 校准框，
  而该框自身标注为 "analysis only; not in control path"，自相矛盾。
  新图中梯度路径从校准框**下方绕过**，且全图只有一条虚线梯度路径。

产物：`Round2/02_figures/figure1_architecture_v3.pdf`
（文件名沿用 `Draft_v10.tex` line 654 已引用的名字，无需改 tex）。

**已知局限（如实说明）**：该 PDF 为位图（6336×2688，双栏页宽约 600 dpi），
送审足够，但 TRC 生产环节通常要求矢量图。仓库中已存在
`Round1/06_figures/figure1_architecture.tex`（498 行 TikZ 源），
后续应在其基础上做矢量版，并把 `see Eq. (12)` 改为 `\eqref{eq:planning-qp}`
以免公式编号变动后失同步。

---

## 5. 本轮归档的 5 项实验产物（Round1 遗留未提交）

这 5 项共约 27 分钟 GPU 计算，此前一直处于 git 未跟踪状态，本轮归入
`Round2/01_new_experiments/` 并提交。**全部为对已发表数值的复核，
且控制项均精确复现**，故可安全引用。

### 5.1 种子稳健性 `SEED_ROBUSTNESS.txt`

- 控制项复现：Stage-2 11.0%(22/200)、Stage-1b 11.5%(23/200) —— 与已发表值一致
- 三种子：CR 12.0 / 11.0 / 11.5%，ADE 7.93 / 5.77 / 7.16 m
- 均值 CR 11.5 ± 0.5(SD)，ADE 6.96 ± 1.09(SD)
- **RUNBOOK 中 "11.4 ± 0.4"、"6.3 ± 1.4" 用的是 SD 而非 SEM**（脚本已判定）
- 三种子对固定 Stage-1b 的精确 McNemar 全部 p = 1.0000 → 与匹配对照不可区分
- 注意：三次检验共用同一 Stage-1b 向量，**彼此不独立**

### 5.2 规划时域敏感性 `LEADTIME_HP_SENS.txt`

- H_p ∈ {8, 15, 25}（时域 1.6 / 3.0 / 5.0 s）下，Stage-2 **一律在 3 s 崩到 0%**
- h=2 s 时三者均为 11.0%
- → **3 s 阈值由遭遇几何与接近动力学决定，不是规划时域的产物**，
  故"监视/探测提前量"的解读成立
- H_p=15 一列精确复现已发表的 lead-time 曲线

### 5.3 弱规划器 2×2 第四格 `WEAK_2X2_MATCH.txt`

- 第四格（deployment-trained 在弱规划器下）= **69.0%**
- 控制项：Stage-1b 56.0%、primary 28.5% 精确复现
- 关键否定结果：**deployment-trained 检查点比 Stage-1b 更差**
  （+13.0 pp，精确 McNemar p = 1.06e-4，且**非嵌套**：9/35）
- → 已发表的 28.5% **不能**读作"弱规划恢复了任务对齐的杠杆"，
  它是**训练/评估规划器相匹配**的效应。§5.4 的正面论断须据此重述。

### 5.4 跨规划器迁移 `PLANNER_TRANSFER.txt`

| 检查点 | 训练于 | ADE | deploy CR | train-time CR |
|---|---|---|---|---|
| Stage-1 | — | 20.90 | 12.5 | 56.5 |
| Stage-1b | — | 1.84 | 11.5 | 56.0 |
| Stage-2 primary | training-time | 4.32 | 11.0 | 28.5 |
| seed1 | training-time | 7.93 | 12.0 | 27.5 |
| seed2 | training-time | 5.77 | 11.0 | 28.5 |
| seed3 | training-time | 7.16 | 11.5 | 27.0 |
| Stage-2 deploy | deployment | 9.05 | 13.5 | **69.0** |

- 同配置族 27.9 ± 0.7%（n=4），**全部严格嵌套**（introduced = 0）
- 检查点间离散度：部署规划器下 **2.5 pp** vs 训练规划器下 **42.0 pp**（17×）
- **不存在"各自在自己规划器下最好"的对角结构**：primary 在它**未**受训的
  规划器下反而绝对更好
- 可辩护表述：**非对称交互** —— 部署规划器**压缩**预测器差异，
  训练规划器**暴露**它们。图不得暗示 matched-is-better 对角线
- ADE 不能解释该分歧（该区间内精度跨 11× 而结果几乎不变），
  只能写作 "not explained by ADE alone and consistent with
  planner-specific adaptation"，**不得**当作证明

### 5.5 `ZERO_SLACK_FEAS.txt`（修改项）

- `infeasible steps per conflict (mean/max)`：**17.77 → 18.82**（新增 raster 转储）
- 已核对：`Draft_v10.tex` line 1967 写的是 **18.8**，
  与更新后的值一致 → **正文无需修改**
- 22/22 conflict 全部含 ε=0 不可行步 → "actuation-limited" 判定不变（100.0%）

---

## 6. 编译状态

- 未解析引用：**59 → 0**
- 未定义交叉引用：**0**
- 参考文献表实际渲染条目：**46**
- 页数：56 页（含 Figure 1 后为 54 页；缺图占位框会影响页数）

**遗留问题（未解决，需后续处理）**

1. **仍缺 14 张图的 PDF**：`fig01_ade_cr_decoupling`、`fig02_minsep_ecdf`、
   `fig03_effort_cr`、`fig05b_raster` 等，`figures/` 中不存在，当前以占位框编译。
2. **3 处 Float too large 警告**：line 676 超 6.3 pt；line 1954 超 356 pt；
   line 2405 超 354 pt。后两处溢出量大，排版会明显异常。
   （注：新增正文导致浮动体重排，警告由 2 处增至 3 处。）
3. **命名不一致**：`\method` 宏 = `PlanGrad`，但 Figure 1 标题写
   `PlanGrad-UAV`，caption 会显示 "Overview of the PlanGrad framework"。需统一。
4. **本地目录漂移未解决**：用户机上为 `code/figure_plotting/`，
   仓库中只有 `figure_plot/` 与 `figure_plotting_v1/`；
   `check_cell_labels` 仅存在于 `figure_plotting_v1/figstyle.py`。
5. **Round2 未 push 到 GitHub**：lab 上无 GitHub 凭证
   （`could not read Username for 'https://github.com'`），
   提交停在本地 `main` 分支，需用户自行 push。

---

## 7. 复核方式（供审阅者验证本文件）

```bash
cd /data/lab/TRC_UAV_WEIGHTS

# 引用完整性：应输出空
grep -o '\\cite[tp]*{[^}]*}' Round2/03_manuscript/Draft_v10.tex \
  | sed 's/\\cite[tp]*{//;s/}//' | tr ',' '\n' | tr -d ' ' | sort -u > /tmp/c.txt
grep -o '^@[a-zA-Z]*{[^,]*' Round2/04_bibliography/refs.bib \
  | sed 's/^@[a-zA-Z]*{//' | sort -u > /tmp/b.txt
comm -23 /tmp/c.txt /tmp/b.txt

# 任一 DOI 的独立复核（示例）
curl -s https://api.crossref.org/works/10.1109/TITS.2020.3048360 \
  | python3 -c "import json,sys; m=json.load(sys.stdin)['message']; print(m['title'][0])"

# 4 项新实验的断言均写在结果文件内，可直接读
cat Round2/01_new_experiments/results/PLANNER_TRANSFER.txt
```

编造引用与错误 DOI 均已在 §2 留档。若后续发现本文件任何一条与实际不符，
应直接在此追加更正，而非改写原条目。
