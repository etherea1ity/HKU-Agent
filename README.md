uvicorn app.main:app --reload --port 8000

# PowerShell (example)
$env:DASHSCOPE_API_KEY="<your_api_key_here>"

**记录一下自己的笔记**
# Rag
## 为什么要RAG
**如果全部放到LLM上下文**
1. 长上下文的常见问题是**关键信息**被大量无关文字淹没，**注意力机制**并不保证能稳定命中，所以可能会产生大量幻觉。**因此RAG的目标**就是把证据**变成高相关的少量片段**，再交给模型。
2. 全部放入LLM上下文，会导致**token成本线性增长，延迟线性增长，吞吐下降**（并发变差）。
3. RAG会产生**证据片段+元数据（来源、行号、chunk id）**，可以让回答必须引用证据，减少幻觉。
4. 并且**LLM的参数知识是静态**的，只靠上下文喂文档是可以更新，但是无法每次都喂全部的最新资料。但是RAG可以只更新索引（增量构建/重建），让成本更低。
5. 并且可以做到**权限与数据隔离**。比如很多企业场景只是不能进入模型训练，也不能长期放在提示词里。

**为什么不直接把文档做一次超大的上下文系统提示词，永久塞进去？**
1. 不可维护。每次更新文档都要重写prompt。
2. 不可扩展。prompt越写越大。
3. 不可控。模型会把新旧知识放一起，冲突时难以定位原因。

**为什么不用fine-tune**
1. Fine-tune更加适合**行为/风格/格式**这种，不适合频繁更新的事实库。
2. 难以做到**逐条引用与审计**。
3. **数据治理成本更高**，如清洗、版本、遗忘、合规。RAG只需要更新索引。

**为什么不能完全消灭幻觉**
1. 幻觉的本质不只在于没知识。
2. 生成模型的语言本质是**概率补全**。
3. **用户问题本身就含糊**，证据不足。

## 文件处理 Ingestion
**为什么分块**：
1. LLM上下文长度上限，降低token消耗
2. 提高相关性和证据可用性，返回更精确片段
3. 可追溯和调参（有metadata，比如说行号）
分块：标题分块，字数分块，语义分块。
### Chunking
#### markdown chunking
这里我们采用了**标题分块+字数分块**。先按照标题来分块，然后再按字数来分块。
**语义分块**
#### Group Chunking
遇到的问题：我们chunk的时候，比如说7107这个课，课程文件里的chunk并不一定都包含了7107，但是很明显都是在介绍这门课，因此**语义和关键词都会漏过这个chunk**。
解决方法，1是检索本身先做doc级，然后再在doc里找chunk。
2是我们在拼接context时，把同组的chunk扩展进来。

这里我们采用了2，也就是分为两个阶段。
1. 和之前一样，召回种子chunks，仍然直接retrieve。
2. 扩展，也就是读取corpus.jsonl，在进程内缓存里面建立一个轻量索引。一个是按照chunk_idx排序，一个是按uid排序。
3. 然后对每个种子 hit做扩展，我们用**相邻扩展**（同windows的前后一个chunk）以及**同section扩展**（如果一个section比较大）。
最终去重，去掉重复chunk_uid，并且保持顺序，返回expanded_hits。
## 搜索 Retrieval
### keyword: bm25
优势：长度归一化，避免长文本占便宜。

对每个 query 词 **t**，BM25 对文档/chunk **d** 的贡献可以理解为：

- **IDF(t)**：词越稀有越重要（只在少数 chunk 出现 → 区分度高）
- **TF 饱和**：词出现越多越相关，但不会线性增长（有饱和曲线）
- **长度归一化**：chunk 越长，TF 的有效贡献越被压制（避免长文本占便宜）

> 工程点：BM25 对课程号、术语、编号、标题关键词非常强。  
> 我们还做了 tokenizer 小增强：从 `comp7107a` 提取数字 `7107`，让纯数字 query 能命中。

### Semantic: Embedding+HNSW
在semantic.py中

### embedding
1. 使用 `sentence-transformers` 句向量模型，把一段文本映射到固定维度向量 RdR^dRd。
2. 离线阶段：对每个 chunk 的 `text` 做 `model.encode(texts)`，得到矩阵：
    1. `embeddings.shape == (num_chunks, dim)`
3. 在线阶段：对 query 做同样的 `encode([query])`，再去索引里找 top-k。

**Transformer 的直觉原理**
1. **文本先被 tokenizer 切成 token**（词/子词），每个 token 映射到一个向量（embedding）。Transformer 输出的是 token-level hidden states。
2. self-attention: 每个 token 不只是看自己，而是对句子里所有 token 分配注意力权重，把上下文信息融合进来。多层attention+FFN后，得到遗传上下文化token向量。
3. 我们的Transformer输出的是每个token一个向量，但检索需要整段一个向量，我们使用pooling池化。比如mean pooling对所有token取平均，或者取CLS向量。

**Normalization（为什么 L2 归一化 + IP 实现 cosine）**
1. 语义检索常用相似度：**cosine similarity**（比较方向，更贴近语义）。
2. 我们对向量做 **L2 归一化**（向量长度变为 1），则：
    - **inner product（IP） = cosine similarity**
3. 这样就可以用 FAISS 的 **IP 索引**（`IndexFlatIP` / HNSW 的 IP metric）来做 cosine 检索。
如果不归一化：
4. 内积会受到向量长度影响，可能出现“某些文本向量 norm 大 → 天然得分高”的不稳定排序。也就是Normalization让**分数尺度稳定**。

**算法**
1. **Flat**精确最近邻。这个就是对所有向量暴力算内积，取top-k。但是数据量大时很慢O(Nd).
2. **ANN**近似最近邻
3. **HSNW**：三个关键参数
	1. M：每个节点的最大连边数，越大召回越好，但是慢占内存。
	2. efConstruction：搜索宽度，100-400。这个值越大，插入一个点时，搜索到的候选邻居更多。
	3. efSearch，查询时的搜索宽度，32-100。这个是我们到了最底部最大的那一层后，候选集合的最大规模。

### Hybrid：RRF
这里我们使用RRF Reciprocal Rank Fusion融合，也就是用BM25搜k1个候选，然后用semantic搜k2个候选，然后把两边取并集，对每个chunk按照排名贡献累加得分。

### top k + threshold
这里我们改了一下方法，就是我们用了比较大的top k+threshold，以此来避免出现用户直接搜索“所有课程”但是我们因为top k限制无法得到所有的文档的情况。

### Learned Fusion
#### 为什么比RRF强
这里我们用RRF已经还可以了，但是它只用rank，不吃score的尺度差异，但是有问题。
1. **它不理解“两个检索器谁更可信”**：比如课程号这种 query，BM25 往往比 embedding 更可靠；但 RRF 默认不给“类型自适应权重”。
2. **它不吃结构信号**：同 section、标题命中、文档长度、chunk 位置等信息，RRF 全部忽略。
3. **它无法系统性调参**：你很难用数据告诉它“哪些信号应该更重要”。
我们使用一个模型，先召回，然后**把候选的多种信号拼成特征，训练一个轻量的ranker来输出最终分数**。
#### 数据哪来？
**A) 人工小标注（最干净，规模小也够用）**
- 选 50~200 个典型问题（课程号、对比、列举、概念解释、多条件查询）。
- 对每个 query，标出 relevant chunk_uid（或 relevant section/doc）。

**B) 弱监督（用已有 RAG 流程自动造标签）**
- 用现有系统跑一遍：如果最终 answer 引用了 chunk [i][j]，把这些 chunk 当 positive；同 batch 里未被引用的候选当 negative（噪声更大，但能快速起量）。
**C) 线上反馈
- 用户后续追问/纠错，可作为“当前 evidence 不够”的负反馈；
- 用户满意/停止追问，可作为弱正反馈。
#### 特征工程
**(1) 检索器分数与 rank 特征**
- bm25_score、bm25_rank、1/(k+bm25_rank)
- sem_score、sem_rank、1/(k+sem_rank)

**(2) 结构/metadata 特征（你们现在就有）**
- doc_len（或 chunk_len）
- title_match：query token 是否命中文档标题/课程名
- section_match：query 是否命中 section 标题
- position：chunk 在文档中的相对位置（开头/中间/结尾）

**(3) Group Chunking 相关特征**
- is_seed：是否为“种子召回 chunk”
- expansion_type：adjacent / same_section / none
- seed_score：对应 seed 的分数（扩展 chunk 继承 seed 的置信度）

**(4) 覆盖/多样性辅助特征（为后续 MMR 做准备）**
- section_id、doc_id one-hot 或 hash
- 同 section 已选数量（在线 rerank 时动态特征）
#### 模型，选用Logistic
**Logistic Regression**对每个chunk做二分类。
这里我们的Learned Fusion是代替了RRF，但是没有代替Colbert，一个是粗排序，一个是细选择。

## ColBERT
这里我们是late interaction+MaxSim。
（1）文档编码 Document Encoding，也就是将一个文档用tokenizer分成token序列，然后通过Transformer得到每个token的上下文化向量。
（2）Query编码，将prompt/query编码成token级向量。这里token级匹配成本主要来自于候选文档token数量。
（3）Scoring：用MaxSim/Late Interaction。**也就是每个query token去找文档里最相似的token，然后求和。这里通常用点积或者cosine**（通常会做归一化！）。
也就是，sim用来求相似度，max找到query token的最匹配evidence，sum就是把query的各个信息点汇总。


比[[3. Architecture of Retriever#Bi-Encoder]]和[[3. Architecture of Retriever#Cross-Encoder]]强在哪？
Bi-Encoder是，每个文档都会被嵌入模型分配一个语义向量，prompt也单独被嵌入模型分配一个语义向量。然后进行比对即可。这里文档都是提前处理好的。
Cross-Encoder则是，我们把文档与查询拼接，然后由于有上下文，所以更好分析出是否有关联。但是扩展性极差。
**A) 为什么它比 bi-encoder 强？**
bi-encoder：整段压成一个向量，细粒度对齐会丢失。  
ColBERT：保留 token 级信息，所以对“多关键词、多约束、多实体”更敏感。
**B) 为什么它通常用来 rerank？**
token-level 匹配计算更贵：
- 如果全库做 ColBERT，会非常慢
- 所以通常先用 BM25/embedding 召回 topN，再用 ColBERT 重排
**C) ColBERT 的工程优化**
真实 ColBERT 会对 token embeddings 做压缩/索引，让 MaxSim 计算更快（比如 IVF/PQ 之类），不然存储和计算都会爆。
## 端到端
### A. 离线阶段：把原始文档变成“可检索资产”

1. **解析和分块**
- markdown：标题分块 + 字数分块
- 每个 chunk 附带 metadata：doc_id / section / 行号范围 / chunk_idx 等（用于引用与 open）
2. **Corpus 组织**
- 存储 chunk_uid -> text + metadata
- manifest：记录构建版本、hash、chunk 数、embedding 模型等（用于可复现）
3. **索引构建**
- **BM25**：词项统计 + 倒排索引（对课程号/术语强）
- **Semantic 向量索引**：对每个 chunk 编码成向量，构建 FAISS（Flat 或 HNSW）
- **Reranker 资源准备**：ColBERT/交叉编码器等，用于 online rerank
### B. 在线阶段：把“用户问题”变成“证据 + 引用的答案”
1. **Query Rewrite**
把用户自然语言变成更可检索的形式（补全缩写、结构化约束、抽取关键实体）
2. **Hybrid Recall（召回）**
- BM25 top-k1
- semantic top-k2（HNSW/Flat）
- 合并候选集
3. **融合排序（RRF）**
- 把“两个排序”的排名转成可加和的分数，避免不同打分尺度难融合
- RRF 很稳，因为它更信“**排名一致性**”，不太怕分数分布漂移。
4. **Group Chunking 扩展（你现在的关键特色）**
- 先召回“种子 chunks”，再把同组/相邻 chunk 扩展进来，解决“关键词不在 chunk 内但语义属于同一节”的漏召回问题
5. **Rerank（可选）**
- 对 top-N 候选做更贵但更准的 rerank（例如 ColBERT late interaction）
- 关键 tradeoff：rerank 的收益取决于 recall 阶段有没有把“正确证据”放进候选池。
1. **Context Packing**
    目标：在 token 预算内最大化“证据覆盖 + 可读性”
- 每个 chunk 限制长度（截断/摘要）
- 多 chunk 去重、同 section 合并
- 引用编号稳定（便于回答引用）
1. **Answer with Citations**
- 明确约束：只用 evidence；不足就说不知道
- 输出：答案 + 引用编号（映射到 metadata 的 source_path/行号）
# Agent工程
## 整体流程
### FastAPI
1. **Lifespan/Startup**：这个是我们把所有的重初始化放进去，检查资产是否存在，必要时构建、把模型/索引加载到内存里。
2. **app.state**保存全局的runtime（重对象）。也就是可以把我们的embedding 模型、FAISS 索引、BM25、ColBERT reranker、LLM client 等放到 `app.state.runtime`。  这样 `/ask` 只是取 `runtime` 去跑 `agent/flow_rag.py` 的流程。
3. /ask：无状态，轻逻辑，所有的工作比如下载，向量化，索引，加载模型都在startup完成！
### Prompt处理
我们的用户的prompt被利用两次，一次是作为初始prompt输入到agent的决策系统里。一次是作为embedding的向量和知识向量拼接，然后一起丢入colbert里面进行重排序。

### Planner和Runtime
我们的工具调用具体包括了rag，联网搜索。

我们的Planner输出为一个严格的JSON，包含了tool/args/stop/final/thought。并且加入非法输出回退。我们的planner输出进行检查后（如工具集，schema），要么可以直接变成PlanStep，要么非法回退，并且是可记录的路径。

并且我们的Action逻辑采用了Avatar，也就是一个对比工程来优化了传统的React。

Runtime我们主要做终止策略，有四种终止情况。
1. Planner 直接 `final`
2. 某个工具产生了最终答案（启发式 stop，例如 answer 成功就停） 
3. 预算触顶 soft stop（可提示“查看 trace / 建议下一步”）
4. 异常终止（工具不可用/抛错）必须也能走到 `done`，避免前端卡死
### Memory
1. **可回放轨迹**：step / tool / args / latency / error / content
2. **给 Planner 的精简视图**：把 raw 结果压缩成 planner-friendly 的 state（避免 token 爆炸，也避免噪声污染决策）

这实际上是在做一个很标准的 agent engineering 模式：
> Memory = trajectory store轨迹存放 + state abstraction状态抽离

 state abstraction：（例如把 hits/evidence 做结构化摘要、把失败原因做分类标签）    
stop condition：（例如“证据覆盖率达到阈值”）

## Avatar 动作选择

### Actor–Comparator

AVATAR 把 agent 拆成两个 LLM 角色：
- **Actor LLM**：执行者（你们这里对应“Planner 生成计划/动作序列”）
- **Comparator LLM**：优化器/训练者（离线阶段对比正负样本，生成“整体性改进指令”来更新 Actor 的 prompt）
并且有两个阶段：
1. **Optimization phase**：Actor 先按当前指令跑；Comparator 批量对比“做得好 vs 做得差”的 query，生成更通用的改进指令，迭代更新 Actor。
2. **Deployment phase**：选最优版本的 Actor（最优 prompts 或最优 actions 模板），直接用于新 query。
### 数据和对比推理
AVATAR 的关键是**用阈值把样本分成两组**：
- 定义两个阈值 $\ell$ 和 $h$（0<$h$≤$\ell$<1）
- 对每个 query 算一个 metric（例如 Recall、Accuracy 等）
- 若 metric ≥ $\ell$：**positive（做得好）**
- 若 metric ≤ $h$：**negative（做得差）**
- 然后随机采样一个 mini-batch，正负各 $b/2$ 条，用来做对比推理。

Comparator通过对比模式差异，把性能差距归因到动作序列，并且给出更好的**通用修改建议prompt**，比如说更好的**拆分任务**，**动作选择和组合方式**。

### 工程约束和Memory Bank
- **有效性检查**：每步动作执行前检查工具调用有效性（例如参数缺失、函数名不在工具列表）
- **超时检测**：如果动作链太慢/太长，触发超时约束，逼迫 Actor 学会删掉冗余步骤（例如重复 search、无意义 open）。
并且维护一个memory bank，存储**动作序列**，**对比器指引**，**策略在小训练集的表现**。

