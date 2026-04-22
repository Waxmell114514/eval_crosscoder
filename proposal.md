# Proposal

## 题目

**可控 LoRA 差分下的跨模型特征恢复：评估 Crosscoder、DFC 与 Delta-Crosscoder 对窄行为微调的可恢复性与因果有效性**

**英文题目：**
**Recovering Narrow Fine-Tuning Deltas under Controlled LoRA: A Causal Evaluation of Crosscoders, Dedicated Feature Crosscoders, and Delta-Crosscoder**

---

## 摘要

近期 model diffing 工作表明，比较 base / finetuned 模型的内部表征，能够发现后训练引入的行为差异；其中 crosscoder 学习两个模型的共享特征空间，已成为 feature-level diff 的重要路线。与此同时，后续工作也指出，**标准 crosscoder 容易把实际上两边都存在的概念误判为 finetune-exclusive**；为此，研究者提出了 **Latent Scaling + BatchTopK** 来缓解 L1 诱导的 exclusivity artifact，并进一步提出 **Dedicated Feature Crosscoder (DFC)** 与 **Delta-Crosscoder**，分别通过结构性独占分区、shared-feature masking、dual-latent allocation 与 contrastive pairing 来提高对模型差异、尤其是**窄微调差分**的恢复能力。([arXiv][1])

然而，当前文献仍缺一个关键证据：**在训练目标完全已知、行为差分可控且细微的 LoRA 微调里，这些方法究竟能否恢复“真正新增或被强化”的行为特征，并在 held-out prompt 上展示因果有效性？** 你的项目设想正是抓住这个缺口：在窄 LoRA 设置中比较 standard crosscoder、BatchTopK crosscoder、DFC、Delta-Crosscoder 与 raw activation diff 等方法，考察它们的 exclusivity、behavior-delta predictiveness 与 steering / ablation causal precision。

本研究拟构造两个层次的“可控行为差分”：一个是**低语义、高可测**的格式型 delta（JSON-only output），用于快速打通管线；另一个是**高语义、可发表**的证据约束型 delta（unsupported-citation abstention / 不足证据时拒绝编造引用），作为主实验。研究将回答：在窄 LoRA 下，哪类 diff 方法最能恢复与行为变化因果相关的 latent；这些 latent 是否优于 raw diff / PCA / 线性 probe；以及“恢复成功”究竟对应的是可预测性、可操纵性，还是两者兼有。这个问题直接落在你文档所说的 “what diff methods actually recover under known training deltas” 上。

---

## 1. 研究背景与问题提出

model diffing 的核心价值，不是重新解释整个模型，而是像代码 diff 一样，直接定位“**这次训练到底改了什么**”。这一路线之所以越来越重要，是因为它天然适合发现 evaluation suite 之外的 unknown unknowns，而且比纯行为评测更接近内部机制。近期工作已经把 crosscoder-based diff 从 base-vs-finetune 推进到 chat-tuning、cross-architecture diff，甚至用于识别版权拒答、政治对齐等行为差异。([arXiv][2])

但这条线目前有两个未被解决的痛点。第一，**标准 crosscoder 的 exclusivity 可能是伪的**：Minder 等指出，L1 训练目标会把本来共享的概念误判成 finetuned-only，Latent Scaling 和 BatchTopK 能显著缓解这个问题。第二，即使有了 DFC 与 Delta-Crosscoder，文献中的成功案例仍大多建立在自然出现的训练差异或 model organism 上，**缺乏一个“训练目标已知、行为差分细而干净”的 controlled benchmark**，来检验这些方法究竟恢复了什么。([arXiv][1])

你的选题报告把这一缺口概括得很准确：在窄 LoRA 微调里，若训练目标足够明确，则 feature-level diff 理应比 raw activation diff 更能解释 held-out 行为差异，而 delta/dedicated 变体理应更抗 exclusivity artifact；但这仍缺少系统验证。报告同时强调，这个项目的优势在于 **ground truth 相对可控、适合单人做干净实验**。

---

## 2. 研究目标

本项目的总目标是：

**在可控 LoRA 差分下，系统评估 crosscoder-family 方法是否能恢复与窄微调目标真正相关的 latent changes，并区分“看起来 exclusive”与“对行为变化有因果作用”的差别。**

具体分成三个子目标：

### 目标 1

比较不同 diff 方法在**恢复窄微调行为差分**上的表现：

* Standard Crosscoder
* BatchTopK Crosscoder
* Dedicated Feature Crosscoder (DFC)
* Delta-Crosscoder
* Raw activation difference / PCA / mean-diff / linear probe
* 仅在 finetuned 模型上训练 SAE 的单模型 baseline

### 目标 2

构建一套比“top activating examples”更硬的评估协议，重点衡量：

* **Exclusivity**：特征是否真只对应 finetune delta
* **Held-out predictiveness**：特征是否在未见模板/未见任务上仍预测行为差分
* **Causal precision**：对目标行为有效操纵时，是否低副作用
* **Robustness**：跨 seed、跨 prompt 模板、跨层是否稳定

### 目标 3

给出一个更强的 mechanistic 结论：

* 不是“某方法找到了几个有趣 feature”
* 而是“**在已知训练 delta 下，哪些方法恢复的是 causally useful latent，而哪些方法只是重命名了 activation drift**”

---

## 3. 核心研究问题与假设

### RQ1

在窄且已知的 LoRA 微调目标下，feature-level diff 是否比 raw activation diff 更能解释 held-out prompt 上的行为变化？

**H1：** 在训练目标足够窄时，crosscoder-family 方法恢复的 latent 对行为差分的预测性将显著优于 raw residual difference、PCA 和 mean-diff。这个假设直接来自你报告对项目三的基本设定。

### RQ2

在 narrow finetuning regime 中，Delta-Crosscoder 与 DFC 是否优于标准 crosscoder？

**H2：** DFC 和 Delta-Crosscoder 会优于标准 crosscoder，尤其体现在：

* 更高的 exclusivity
* 更高的 causal precision
* 更少的 shared-feature contamination

这个假设的依据是：DFC 通过结构上把 feature 空间分成 model-A exclusive、model-B exclusive 与 shared 三部分，显式去掉标准 crosscoder 偏向 shared features 的结构先验；而 Delta-Crosscoder 进一步通过 contrastive pairing、shared-feature masking 和 dual latent allocation，专门增强对**细小、局部、窄微调**差分的敏感度。([arXiv][2])

### RQ3

“恢复成功”是否必然意味着这些特征具有因果作用？

**H3：** 不必然。某些方法可能恢复出**高可预测但低可操纵**的 latents；因此，真正的恢复能力应由“predictiveness + causal intervention effect”共同定义，而不是只看可解释文本或 decoder ratio。这个判断也与近期 mech interp 对“局部手柄 ≠ 完整机制”的整体反思一致。

---

## 4. 研究设计

## 4.1 总体设计

本项目采用**两阶段模型 organism 设计**：

### Phase A：快速 pilot（3–5 天）

**目标行为：JSON-only output**

目的不是发论文，而是快速验证：

1. LoRA 能否造出稳定、可 hold-out 的行为 delta
2. 激活缓存、matched prompts、crosscoder 训练管线是否工作
3. 顶层评估指标是否有分辨力

这个 delta 很窄，容易训练，也便于度量格式遵循率、键完整率、非法自然语言泄漏率等指标。

### Phase B：主实验（2–6 周）

**目标行为：unsupported-citation abstention**
即：当证据不足时，模型更稳定地说“无法验证/不给出引用”，而不是编造文献、编造 DOI、编造作者名。

我建议把它作为主实验，因为它同时满足：

* 行为差分窄
* 语义上比 JSON 更有研究价值
* 可以构造支持/不支持/边界案例
* 因果干预结果更容易讲成完整故事

---

## 4.2 模型选择

### 主模型

**Gemma 2 2B-it** 或 **Qwen2.5-3B-Instruct**

理由是：

1. 模型小，LoRA + activation caching + diff 训练对独立研究者更友好
2. 现有 model diffing / crosscoder 文献常在 2B–9B 范围做验证
3. Gemma 系列还有公开 SAE 资产可供后续 sanity check 或 feature 对照使用。你的报告也把公开特征资产与工具链成熟视为这个窗口期的重要原因。

### 稳健性复现

若主实验有效，再在第二个同量级模型上复现一次，验证方法 ranking 是否稳定。

---

## 4.3 LoRA 训练目标与数据

### Pilot：JSON-only output

* 训练目标：将自然语言回答改写为固定 schema 的 JSON
* 数据来源：现成 instruction-response 数据，自动改写成统一 schema
* 评估指标：

  * JSON valid rate
  * schema adherence
  * extra-text leakage
  * held-out prompt templates 泛化

### 主实验：Unsupported-Citation Abstention

构造三类数据：

1. **Supported**
   给定问题 + 足够证据，允许回答并引用证据片段

2. **Unsupported**
   给定问题 + 不足或无关证据，理想输出是“无法验证/不提供引用”

3. **Borderline**
   给定部分证据或相似实体，测试模型是否会“半懂不懂地编造”

这使得训练 delta 足够窄：不是让模型整体更安全，而是只改变**“证据不足时的引用行为”**。这比泛化 safety tuning 更容易做出干净的差分结论。

---

## 4.4 激活采样与对齐方式

对 base model 与 LoRA model，使用**同一批 matched prompts** 前向推理，缓存 1–2 个中高层 residual stream（优先 top third 的层；若资源允许，可加一个中层）。之所以关注较高层，是因为近期 narrow finetuning / model diffing 工作普遍认为，后训练差异更多体现在较高层表示，并且更接近行为输出。([arXiv][3])

训练时使用两种输入混合：

1. **paired matched activations**：同一 prompt 在 base / LoRA 上的激活对
2. **unpaired generic activations**：一般聊天/问答 prompt，用于防止方法只记住任务模板

这与 Delta-Crosscoder 的核心思想一致：利用 matched data 引入对比不对称性，同时不完全依赖 finetune 训练数据本身。([arXiv][3])

---

## 4.5 比较方法

### A. 非 SAE / 非 crosscoder baselines

* Raw activation difference
* Mean-diff vector
* PCA on activation deltas
* Linear probe 区分 base vs LoRA
* 行为标签 probe（如是否 abstain / 是否 JSON-compliant）

### B. 单模型 baseline

* 仅在 LoRA 模型上训练 SAE，再看 top latents 是否能解释行为变化

### C. Crosscoder family

* Standard Crosscoder
* BatchTopK Crosscoder
  之所以要单列，是因为 Minder 等已表明，L1 crosscoder 容易制造 exclusivity artifact，而 BatchTopK 显著更稳。([arXiv][1])
* Dedicated Feature Crosscoder (DFC)
  DFC 通过结构分区，显式为 A-only、B-only、shared 保留不同 feature 子空间，目标就是更好地隔离模型独有特征。([arXiv][2])
* Delta-Crosscoder
  使用对比配对、shared-feature masking 与 dual-latent allocation，专门面向 narrow finetuning 的小而稀疏差分。([arXiv][3])

---

## 5. 评估协议

这是 proposal 的关键部分。建议把“恢复能力”拆成 4 个维度。

## 5.1 Predictive Recovery

问题：恢复出的 latent 是否真的解释了行为差分？

指标：

* 用 top-k latents 预测：

  * base vs LoRA label
  * target behavior label（如 abstain / fabricate / valid JSON）
* held-out templates 上的 AUROC / F1 / calibration
* few-latent sufficiency：只用少量 latents 能否解释大部分行为差异

**期望结论：**
如果 feature-level diff 真的恢复了训练 delta，那么它应在 held-out prompt 上显著优于 raw diff / PCA。

---

## 5.2 Exclusivity

问题：所谓“LoRA-exclusive feature”是否真是 exclusive，而不是 shared concept 被误判？

指标：

* decoder norm ratio
* latent scaling-style presence measure
* affine-transfer-based exclusivity（若实现成本可控）
* DFC 中 dedicated vs shared 分区的使用情况

这里直接对应文献已指出的 shared-prior / false exclusivity 问题：标准 crosscoder 倾向 shared，而 DFC/Delta-Crosscoder 正是在解决这个。([arXiv][1])

---

## 5.3 Causal Recovery

问题：这些 latents 是否因果地介导了行为变化？

操作：

* 对 recovered target latents 做 activation steering
* 对 recovered target latents 做 ablation / patching
* 看 base 能否被“推向 LoRA 行为”
* 看 LoRA 能否被“拉回 base 行为”

指标：

* target behavior shift
* off-target damage：

  * 普通问答质量
  * 回答长度异常漂移
  * 无关 refusal 增加
  * 格式外副作用

**核心指标建议：**
[
\text{Causal Precision} = \frac{\text{Target Behavior Gain}}{\text{Collateral Drift} + \epsilon}
]

这样你就不是只卖“能 steer”，而是卖“**低副作用地恢复/逆转已知训练 delta**”。

---

## 5.4 Robustness

问题：恢复是否稳定？

测试：

* 不同随机种子 LoRA
* 不同 prompt 模板
* 不同层
* pilot 任务 vs 主任务

若方法 ranking 只在单一 seed 或单一模板上成立，论文说服力会大幅下降。

---

## 6. 预期结果

### 预期 1

在 **pilot 的 JSON delta** 上，几乎所有方法都能找到某种强信号，但 raw diff 与 probe 可能已经很强。
这部分主要作用是验证管线，而不是证明 crosscoder 胜出。

### 预期 2

在 **citation abstention delta** 上，方法差异会真正拉开：

* standard crosscoder 容易给出“看似 exclusive”的伪特征
* BatchTopK 会更稳
* DFC 在 exclusivity 上更好
* Delta-Crosscoder 在 subtle delta、held-out predictiveness 与 causal recovery 上最可能胜出

这与现有文献对各方法的改进方向是一致的：BatchTopK 主要纠正标准 crosscoder 的假 exclusivity；DFC 通过架构分区来强化 model-exclusive discovery；Delta-Crosscoder 则明确面向 narrow finetuning。([arXiv][1])

### 预期 3

你最终最有价值的结论，未必是“某个方法赢了”，而可能是：

* **有些方法恢复的是 prediction handle，不是 causal handle**
* **有些 delta 根本没有稀疏独占特征，而更像全局涂抹**
* **格式型 delta 和语义型 delta 需要不同的 diff 方法**

这些负结果也是有价值的，因为它们直接回答“what diff methods actually recover”。

---

## 7. 创新点

### 创新点 1：已知训练目标下的 controlled benchmark

现有工作虽已展示 crosscoder、DFC、Delta-Crosscoder 在真实模型差异或 model organism 上能找到有意义特征，但“训练目标完全已知、细粒度行为差分可控”的基准仍很缺。([arXiv][2])

### 创新点 2：把恢复能力拆成 predictiveness 与 causality

很多工作停在“找到 feature”或“可 steering”，但 proposal 明确要求用 held-out 预测 + steering/ablation 双重验证，避免把 intervention handle 误当成机制本身。这个角度也正符合你报告里对“不要只证明局部足够”的提醒。

### 创新点 3：独立研究者友好的方法学产出

这题最强的现实价值，是它可以沉淀出一套很可复用的 protocol：

* 如何构造窄 LoRA model organism
* 如何做 matched activation caching
* 如何比较 diff 方法
* 如何定义 causal recovery score

这比单一案例 feature story 更像一个可被后续工作复用的小 benchmark。

---

## 8. 可行性与资源评估

本项目不需要训练大模型，也不需要 frontier 级别算力。你的报告本来就把这条线定位为“对实现细节稍挑剔，但长期价值高”的方向，而 controlled LoRA setting 的优势恰恰是：你不用一开始就碰 base→chat 或 SFT→RLHF 这种巨大、混杂的真实训练差分。

一个现实可行的配置是：

* 1 个 2B–3B instruct 模型
* 1–2 个窄 LoRA 微调
* 1–2 层 activation cache
* 一套 diff 方法比较
* 一次跨模型复现作为增强版

对独立研究者来说，这已经足以支撑一个 workshop / Findings 级别的论文故事。

---

## 9. 风险与应对

### 风险 1：LoRA 根本没造出稳定行为 delta

这是最大风险。你的报告也把它列为首个 kill criterion：如果连稳定、可 hold-out 的行为差分都造不出来，就应该立即停。

**应对：**

* 先做 JSON pilot
* 若 citation abstention 太难，退回到 “更强烈表达不确定性” 作为主任务
* 保证 target behavior 的自动评测清晰可算

### 风险 2：所有 diff 方法都不比 raw diff 强

这也是报告中的明确 kill criterion。

**应对：**

* 缩小任务定义
* 提高 matched prompt 质量
* 先只比较 BatchTopK vs DFC vs Delta-Crosscoder
* 把结论转化为：**在某类窄 LoRA 下，复杂 diff 方法没有带来超出 raw activation drift 的增益**

这也是可发表的负结果。

### 风险 3：找到的 latents 只能 steer 风格，不能改目标行为

**应对：**

* 增加 counterfactual patching
* 关注“ablate finetuned behavior”而不是只做 positive steering
* 加入 off-target drift 评估，防止把“模型变笨了”误当成因果发现

---

## 10. 时间表

### 第 1 周

* 完成 JSON pilot 数据与 LoRA
* 跑 matched prompts
* 搭建 raw diff / probe / standard crosscoder 管线

### 第 2 周

* 跑 BatchTopK crosscoder、DFC
* 做第一轮 exclusivity / predictiveness 比较
* 判断主实验是否值得继续

### 第 3–4 周

* 上 citation abstention 主实验
* 跑 Delta-Crosscoder
* 做 steering / ablation / patching

### 第 5–6 周

* 做 held-out 模板、seed 稳健性
* 画图、清理案例、写初稿

### 第 7–8 周

* 若结果强，补一个第二模型复现
* 若结果一般，转成“negative but informative”版本，强调 narrow finetuning 的边界条件

---

## 11. 预期产出与论文定位

### 最理想标题

**What Do Model Diffing Methods Actually Recover Under Known Training Deltas? A Controlled Study with Narrow LoRA**

### 次优标题

**Recovering Fine-Tuning-Specific Latents in Narrow LoRA Regimes: A Comparison of Crosscoders, DFC, and Delta-Crosscoder**

### 若结果偏负

**When Narrow LoRA Deltas Resist Sparse Recovery: Limits of Crosscoder-Based Model Diffing in Controlled Fine-Tuning**


[1]: https://arxiv.org/abs/2504.02922 "https://arxiv.org/abs/2504.02922"
[2]: https://arxiv.org/pdf/2602.11729 "Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs"
[3]: https://arxiv.org/html/2603.04426v1 "Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes This paper contains text that might be offensive."
