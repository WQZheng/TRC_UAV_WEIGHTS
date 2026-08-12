# Round2 — 投稿稿件、参考文献、框架图与补充实验

本目录是 **Round1 之后**这一轮工作的权威归档。定位与 Round1 不同：

- `Round1/` = **方法与主体实验**的完整梳理（代码、权重、结果、12 张图）
- `Round2/` = **投稿定稿相关**的产物：稿件、参考文献体系、Figure 1、
  以及 Round1 遗留未提交的 5 项补充实验

> **Round2 不复制 Round1 的内容。** 代码、权重、主体结果仍以 `Round1/`
> 与原始 `code/` 目录为准。要重跑实验请按 `Round1/RUNBOOK_main.md`。

逐条改动与验证记录见 **[`CHANGELOG_ROUND2.md`](CHANGELOG_ROUND2.md)**。

---

## 0. 目录结构

```
Round2/
├── README_ROUND2.md               ← 本文件
├── CHANGELOG_ROUND2.md            ← 全部改动 + 验证证据（含 2 起严重问题留档）
├── 01_new_experiments/            Round1 遗留未提交的 5 项补充实验
│   ├── scripts/                   4 个 export 脚本
│   ├── results/                   5 个结果 txt（断言已写入文件内）
│   └── fig_data/                  6 个 npz/json 数据产物
├── 02_figures/
│   └── figure1_architecture_v3.pdf  ← Draft_v10.tex 引用的就是这个名字
├── 03_manuscript/
│   └── Draft_v10.tex              ← 当前投稿稿（3415 行）
└── 04_bibliography/
    └── refs.bib                   ← 65 条，全部经 CrossRef 核对
```

---

## 1. 一句话结论

本轮把稿件从**无法正确编译参考文献**（59 条未解析引用）修到 **0 条未解析**，
并在过程中**发现并清除了 4 条编造的参考文献**、**纠正了 1 条被自动工具
错误指向另一篇论文的引用**，同时补入指定的 CTR / JICV 各 2 篇，
重制了 Figure 1（修正 4 处与正文不符的标注）。

---

## 2. 稿件状态（`03_manuscript/Draft_v10.tex`）

| 项目 | 值 |
|---|---|
| 行数 | 3415 |
| 被引 key | 46 |
| 未解析引用 | **0** |
| 未定义交叉引用 | **0** |
| 参考文献表渲染条目 | 46 |
| 页数 | 56 |

`Draft_v10.tex` 相对 `Draft_v9.tex` 的改动集中在 Related Work（§2）
与 Introduction 的 1 处引用，共 66 行 diff。**方法与结果章节未改动。**

编译方式（bibliography 为 `plainnat`，author-year）：

```bash
pdflatex Draft_v10.tex && bibtex Draft_v10 && pdflatex Draft_v10.tex && pdflatex Draft_v10.tex
```

需要 `refs.bib` 与 `figures/` 在同级目录。

---

## 3. 参考文献（`04_bibliography/refs.bib`）

65 条，46 条被引用。本轮：

- 修正 **12** 处键名漂移（正文 key 指向库中已存在的同一篇论文）
- 新增 **17** 条，每条都先经 CrossRef 按标题确认存在，再逐字段核对
- 删除 **5** 条无法核实的引用（其中 4 条判定为编造）

**铁律**：本轮所有条目均**独立向 CrossRef 复核过 DOI**，
不依赖 `ref_verify` 的 `correctedBib` 直接写回。
原因见 `CHANGELOG_ROUND2.md` §2.2 —— 该工具曾把一条引用
"修正"成了完全不同的另一篇论文，且**未**标记为不可验证。

### 指定期刊补充

| 期刊 | 条目 | 落点 |
|---|---|---|
| Communications in Transportation Research | `zhao2024clustering`（4:100151） | 混合自主段 §2.3 |
| Communications in Transportation Research | `mohammadian2023continuum`（3:100107） | 同上 |
| Journal of Intelligent and Connected Vehicles | `sheng2024kinematics`（7(2):138–150） | 低空预测对比段 §2.1 |
| Journal of Intelligent and Connected Vehicles | `an2025longterm`（8(1):9210045） | 同上 |

4 篇均按其**实际内容**改写正文句子，不是把新 key 塞进旧句子。

---

## 4. Figure 1（`02_figures/`）

修正了手绘版中 4 处与正文不符的标注：`d_safe`→`d_sep`；
混合权重 `π`→`α^{i,m}`；`Φ_dev` 的"RMS/cross-track 路径偏差"→
**terminal lateral deviation**；`Φ_effort` 的"energy"→**control activity**。

> 其中 `Φ_dev` 那条是 v9 中**已修正过一次**的说法，手绘图又带了回来。

同时修正一处逻辑矛盾：原图梯度虚线穿过 split-conformal 校准框，
而该框标注为 "analysis only; not in control path"。新图梯度路径从其下方绕过。

源图 `figure1_architecture_v4.png`（6336×2688，10.4 MB）**未纳入 git**：
仓库未启用 git-lfs，`.git` 仅 24 MB，收录会使其膨胀约 40%，
而它只是 PDF 的中间产物。该文件仍留在 lab 的 `figures/` 下（未跟踪），
需要时可取。

**局限**：位图 PDF（约 600 dpi @ 双栏页宽），送审够用，
**但 TRC 生产环节通常要求矢量图**。仓库已有
`Round1/06_figures/figure1_architecture.tex`（498 行 TikZ），
建议在其基础上出矢量版，并把图内 `see Eq. (12)` 换成
`\eqref{eq:planning-qp}` 以免编号漂移。

---

## 5. 补充实验（`01_new_experiments/`）

5 项，约 27 分钟 GPU 计算，Round1 时期已跑但一直未提交 git。
**控制项全部精确复现已发表数值**，故可安全引用。
每个 txt 内部都写入了断言与判读，**图/表不可能与结论反向发布**。

| 结果文件 | 回答的问题 | 关键结论 |
|---|---|---|
| `SEED_ROBUSTNESS.txt` | Stage-2 结果是否只是幸运种子？ | CR 11.5±0.5(SD)；三种子对 Stage-1b 的 McNemar 全 p=1.0 |
| `LEADTIME_HP_SENS.txt` | 3 s 崩塌是规划时域造成的吗？ | **不是**。H_p∈{8,15,25} 一律 3 s 崩到 0% |
| `WEAK_2X2_MATCH.txt` | 弱规划器 2×2 的第四格 | **69.0%**，deployment-trained 比 Stage-1b 更差（p=1.06e-4，非嵌套） |
| `PLANNER_TRANSFER.txt` | 任务对齐目标能跨规划器迁移吗？ | 非对称交互：离散度 2.5 pp vs 42.0 pp（17×） |
| `ZERO_SLACK_FEAS.txt` | actuation-limited 判定 | 22/22 成立；均值 17.77→**18.82**（正文 line 1967 已是 18.8，无需改） |

### 两条必须遵守的表述纪律

1. **§5.4 的正面论断须重述**：已发表的 28.5% 不能读作"弱规划恢复了任务
   对齐的杠杆"。deployment-trained 检查点在弱规划器下**更差**（69.0%），
   所以那是**训练/评估规划器相匹配**的效应。
2. **图不得暗示 matched-is-better 对角线**：primary 检查点在它**未**受训的
   规划器下反而绝对更好。可辩护的说法只有"非对称交互"——
   部署规划器压缩预测器差异，训练规划器暴露它们。
   ADE 亦**不能**解释该分歧，只能写 "not explained by ADE alone and
   consistent with planner-specific adaptation"。

---

## 6. 与 Round1 的关系

| 内容 | 位置 | 本轮是否改动 |
|---|---|---|
| 核心库 / 基线代码 | `Round1/01_*`、`02_*`；原始 `code/` | 未改 |
| 权重 9 个 `.pt` | `Round1/04_weights/` | 未改 |
| 主表 / 统计检验 | `Round1/05_results/main_and_stats/` | 未改 |
| 12 张主图 | `Round1/06_figures/` | 未改（Figure 1 另出新版于 Round2） |
| 稿件 | — | **Round2 新增 `Draft_v10.tex`** |
| 参考文献 | — | **Round2 新增（大幅修订）** |
| 5 项补充实验 | 原散落在 `code/` 未跟踪 | **Round2 归档并提交** |

`ZERO_SLACK_FEAS.txt` 在原位（`code/plangrad_sim/`）有一处修改，
Round2 保存的是**更新后**的版本。

---

## 7. 遗留问题（按优先级）

1. **仍缺 14 张图的 PDF** —— `figures/` 中不存在
   （`fig01_ade_cr_decoupling`、`fig02_minsep_ecdf`、`fig03_effort_cr`、
   `fig05b_raster` 等），当前以占位框编译。
2. **3 处 Float too large** —— line 676 超 6.3 pt、line 1954 超 356 pt、
   line 2405 超 354 pt。后两处溢出量大，排版明显异常。
3. **命名不一致** —— `\method` = `PlanGrad`，Figure 1 标题为 `PlanGrad-UAV`。
4. **Figure 1 需矢量化**（见 §4）。
5. **本地目录漂移** —— 用户机 `code/figure_plotting/` 与仓库
   `figure_plot/` / `figure_plotting_v1/` 不一致；
   `check_cell_labels` 仅在 `figure_plotting_v1/figstyle.py` 中。
6. **未 push 到 GitHub** —— lab 无凭证，提交停在本地 `main`。
   需在 lab 上执行 `git push origin main`。
