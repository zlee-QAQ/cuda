# Flash attention v1

## FWD

![img](https://pic1.zhimg.com/80/v2-dc365574847fbba47ec00d84bfd03164_720w.png?source=d16d100b)





宏观上来看，

- 外循环 load  K V
- 内循环 load Q

一次外循环，就可以得到 N x d的一个 完整的O ，但是这个O 是基于K 、V片段计算得到的，需要累加，以及更新

一次内循环，可以得到O上  Br∗d 的一小块矩阵

### 详细解释

![img](https://pic1.zhimg.com/80/v2-2caf574e489e1563527fe5b014a1fb9e_720w.png?source=d16d100b)

- O 存储输出， l 存储N 行分别对应的指数和， m 存储N行分别对应的每一行中当前的最大值

![img](https://pic1.zhimg.com/80/v2-37e0433a0a70215487bad483d439eb3e_720w.png?source=d16d100b)

- 沿着seq_len 所在维度切分 Q、K、V

![img](https://pica.zhimg.com/80/v2-57a0a613400cd45341453b11b9aae753_720w.png?source=d16d100b)

- 外循环，load $K_j$ 、 $V_j $，验证seq_len方向切分的$K_j$ 、 $V_j $ 块

![img](https://pic1.zhimg.com/80/v2-7ec5a2fbd37fd20b2225d288ec62cd5c_720w.png?source=d16d100b)



- 这里load的 $Q_i  $是本次 Oi 计算所需要的，$O_i$、 $l_i$ 、 $m_i$ 的则是上一次外循环中相同的内循环计算产生的结果，$O_i$、 $l_i$ 、 $m_i$ 主要用于更新的过程

![img](https://picx.zhimg.com/80/v2-2e5d6669b040e021901e24361639a2cd_720w.png?source=d16d100b)

- 这里是计算当前O所需要的数据，注意 $\tilde{m_{ij}}$是一个向量，保存的是当前O上  Br∗d 的一小块矩阵对应 所有行的最大值， $\tilde{P_{ij}}$ 是softmax计算过程中的分子， $\tilde{l_{ij}}$ 是softmax需要计算的指数和。

![img](https://picx.zhimg.com/80/v2-a0a04b43f2c09b57f6505dda83391a5e_720w.png?source=d16d100b)

- 首先更新当前的softmax所需的最大值 $m_{i}^{new}$，然后根据最大值去更新指数和$l_{i}^{new}$，注意这里需要对指数和累加起来
- $diag(l_i)e^{m_{i} - m_{i}^{new}}O_i$这个计算中 $ e^{m_i - m_{i}^{new}} $ 的作用是对上一次大循环保存下来的 $O_i $中softmax计算的分子进行更新，$diag(l_i)$的作用是把$O_i $中上一次大循环中softmax计算的分母给重新乘上，方便后续$diag(l_i)^{(-1)}$来更新softmax的分母
- $e^{\tilde{m_{ij}} - m_{i}^{new}}\tilde{P_{ij}}V_j$是更新当前内循环，计算出的softmax的分子，然后计算出当前的O，
  - $diag(l_i)^{(-1)}$就是softmax的分母

## BWD

![img](https://pic1.zhimg.com/80/v2-7df265dc1ef9b0d8178d69566016667d_720w.png?source=d16d100b)

宏观上，

- 一次外循环 load K、V上一个 Bc×d 的一个 Kj 、 Vj ,同时计算它们对应的 dKj 、 dVj 
- 一次内循环load O 上一个块 Oi , 然后计算出dKj 、 dVj 累加过程中的中间值 dKj~ 、 dVj~ ，

和正常的bwd比较，这里因为之前没有保存 和P和S ，所以这里需要对P 、 V 进行冲计算，所以需要去load Q



# Flash attention V2

## 详细介绍

### FWD

![img](https://pica.zhimg.com/80/v2-d48c1a13349666e2b0b51737d43baf85_720w.png?source=d16d100b)

- 宏观理解
  - 一次外循环，计算出O上 Br×d 的一个一个行块，每次外循环的结果O是拼接起来的
  - 一次内循环计算出 O 上 Br×d  的一个行块，但是这个行块的累加是不完整的，同时softmax 过程中所需的max_val 和 exp_sum 也都是不完整的，每次内循环的结果是累加更新的

![img](https://pic1.zhimg.com/80/v2-934d9a7169698ba6c5ae3746af0be802_720w.png?source=d16d100b)

微观理解

![img](https://pic1.zhimg.com/80/v2-9fa3e99b80ea5e06165d78ac819f988d_720w.png?source=d16d100b)





-  load Qi , 同时在片上申请 li 、 mi ，
- 注意这里li 、 mi 的shape，都是一维度的，每次内循环会更新一次

![img](https://pic1.zhimg.com/80/v2-bb4e430d54c0c3d924b53bf0a32f2d2b_720w.png?source=d16d100b)



- load K、V 块 

![img](https://picx.zhimg.com/80/v2-f8a20b7853aec36418662b6795abc7dd_720w.png?source=d16d100b)



- 主要计算过程，
  - 首先计算S
  - 计算softmax的分子部分、分母部分
  - 更新O中softmax的分子部分，注意这里分母部分是不更新的。
  - 分母是计算完内循环之后，才会更新的

![img](https://pic1.zhimg.com/80/v2-cfa70a03c9ff84bae366b57f4d44aaaf_720w.png?source=d16d100b)



- 这里内循环结束之后，去用softmax的分母更新O
- 同时，这里有v2的一个优化，不再需要同时保存最大值和指数和，只需要保存 Li 即可。

## 优化点

1. 之前只在head和batch_size的维度进行并行，单一样本中不并行，现在对于单一样本可以实现seqlen维度并行。
   1. fwd 过程中V1 是外循环load K、V，内循环load Q， V2 中是外循环load Q、内循环load K、V，总的load的数据的量是减少的。fwd过程中是沿着行的方向对Q切分，实现并行。
   2. BWD load 顺序不变，沿着seqlen的方向对K、V切分，实现threadblock的并行，需要注意的是这个过程中dq需要跨block通信，实现累加
2. warp内通信的优化，V1中，每个warp 持有完整的Q， K和V是切分到load到不同的warp上，这样，计算QK 和 后续PV的时候，需要进行同步。V2中，每个warp持有完整的K、V，但是Q分散到不同的warp上，这样就避免了K、V的同步和跨warp的通信。
3. 减少非matmul计算，fwd 和 bwd 过程中在更新softmax的分子和分母的时候计算减少，同时fwd 只需要保留 Li=mi+log(li) 即可，不需要保留 mi  和 li  ，bwd load也只需要load Li=mi+log(li) 即可

### 优化1 

V1 的并行维度是 **batch_size** **(批次大小) 和** **num_heads** **(注意力头数)**。它把一个注意力头的计算（针对一个样本）作为一个完整的线程块。

- **总线程块数 =** **batch_size** **×** **num_heads**

这种策略在 batch_size 和 num_heads 很大时非常有效。

但是，当处理**长序列 (long sequence)** 时，问题就出现了。比如序列长度达到 8k、16k 甚至更长，单个样本就会消耗巨大的 GPU 显存。为了防止显存溢出（OOM），我们不得不**大幅减小** **batch_size**（比如降到 1 或 2）。这时，总线程块数可能就远小于 SM数量 了。例如，一个有 32 个头的模型，batch_size 为 2，总共只有 64 个线程块，这意味着 A100 GPU 的 108 个SM里，有 44 个是空闲的，造成了巨大的资源浪费。这就是 V1 在长序列场景下性能无法达到峰值的主要原因。

**V2 的解决方案：增加seqlen维度并行**

V2 的天才之处在于，它在 V1 的基础上，额外增加了沿 **sequence_length** **(序列长度)** 维度的并行。它不再把一整个头的计算作为一个不可分割的线程块，而是将其进一步拆分。

- **前向传播 (Forward Pass)**：
- 计算被划分为按 **行 (row)** 的块。整个注意力输出矩阵 O (大小为 N×d) 被切分成多个行块 Oi。
- 每个线程块现在负责计算**一个行块**的输出。
- 由于每个行块的计算是完全独立的（Oi 的计算只需要 Qi 和完整的 K,V），这些threadblock之间**无需任何通信**。
- **效果**：即使 batch_size 很小，一个注意力头内部也可以被拆分成 N / B_r （Br 是行块大小）个独立的thread。
- load的数据量要是减少的，[*参照这个链接*](https://zhuanlan.zhihu.com/p/665170554)

![img](https://picx.zhimg.com/80/v2-2706fdf1e7511f335305f5d847c3f7bf_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

- **后向传播 (Backward Pass)**：
- 后向传播的计算依赖关系更复杂。V2 巧妙地将其按 **列 (column)** 的块进行并行。
- 每个线程块负责一个列块的计算，这会涉及到对梯度 dQ 的更新。
- 由于多个列块的计算结果需要累加到同一个 dQ 上，V2 使用了**原子操作 (atomic adds)** 来保证数据更新的正确性和线程安全，避免了冲突。

![img](https://pic1.zhimg.com/80/v2-8c6a57a5e92493fa954de8c968d06672_720w.png?source=d16d100b)

添加图片注释，不超过 140 字（可选）

 V2 将后向传播的并行维度也扩展到了**序列长度**上，其划分方式与前向传播的“行并行”不同，采用了巧妙的 **“列并行”** 策略。

- **任务划分**：整个计算被划分为按**列块 (column block)** 进行。GPU 上的每一个线程块（可以理解为一个独立的“计算单元”）被分配去计算**一个列块**的梯度，即 dK_j 和 dV_j。
- **工作流程**：

1. 线程块 j 从 HBM（主显存）加载它负责的 K_j 和 V_j 到片上高速 SRAM。
2. 接着，它开始一个**内循环**，遍历**所有**的行块 i (从 1 到 Tr)。在每次内循环中，它加载对应的 Q_i, O_i, dO_i 和统计量 L_i。
3. 利用这些加载的数据，它重计算出中间值 Pij 和 dSij。
4. 它可以**完整地、独立地**累加计算出最终的 dK_j 和 dV_j，因为这两个梯度的计算只依赖于第 j 列块和所有的行块。当内循环结束时，dK_j 和 dV_j 就计算完毕，可以直接写回 HBM。

- **核心挑战与解决方案：****dQ** **的更新**
- **冲突问题**：当线程块 j 在处理行块 i 时，它会计算出一个对 dQ_i 的**部分贡献**（dS_{ij} K_j）。几乎在同一时刻，另一个线程块 j+1 也在处理行块 i，并计算出它对**同一个** **dQ_i** 的部分贡献。如果它们都直接去写入 HBM 中的 dQ_i，就会发生“写冲突”（Race Condition），后写入的会覆盖先写入的，导致结果错误。
- **V2 的答案：原子加法 (Atomic Add)**。V2 使用了 GPU 提供的原子操作。当线程块 j 计算完对 dQ_i 的贡献后，它不是执行“写入”，而是执行一个“原子加法”操作。这个操作能保证“读取 dQ_i 的旧值 -> 加上自己的贡献 -> 写回新值”这三个步骤是**不可分割的、一次性完成的**。任何其他线程块在此期间都无法访问 dQ_i，只能排队等待。
- **效果**：通过这种方式，V2 完美地解决了并行计算 dQ 时的冲突问题，使得整个后向传播可以沿着列方向大规模并行。

### 优化2

**V1 的方式：“Split-K” 方案**

V1 在一个线程块内（比如用 4 个 Warp），采用的是切分 **K 和 V** 矩阵的策略。

- **工作流程**：

1. 一块 Q (比如 Qi) 被所有 4 个 Warp 共享。
2. K 和 V 被切成 4 份，每个 Warp 分到一份 (Warp1 拿 Kj(1),Vj(1)，Warp2 拿 Kj(2),Vj(2) ...)。
3. 每个 Warp 计算自己分片的 Sij(slice)=Qi⋅(Kj(slice))T。
4. **瓶颈出现**：为了计算最终的输出 Oi，每个 Warp 不仅需要自己的 Vj(slice)，还需要其他 Warp 手里的 Vj 分片。这导致了复杂的内部通信：

- 每个 Warp 必须把自己计算出的中间结果（例如 Pij(slice)）**写入**共享内存。
- 所有 Warp 必须**等待**，直到大家都写完（一次同步开销）。
- 然后，每个 Warp 再从共享内存中**读取**其他 Warp 写入的结果，进行汇总计算。
- **缺点**：这种“写入-同步-读取”的循环，引入了显著的共享内存通信开销和延迟，拖慢了整体计算速度。

**V2 的方式：“Split-Q” 方案**

V2 完全颠覆了 V1 的内部协作模式，改为切分 **Q** 矩阵。

- **工作流程**：

1. 一整块 K 和 V (比如 Kj,Vj) 被所有 4 个 Warp 共享。
2. Q 被切成 4 份，每个 Warp 分到一份 (Qi(1),Qi(2),…)。
3. 每个 Warp 可以**独立、完整地**计算出自己负责的那部分最终输出：Oi(slice)=softmax(Qi(slice)⋅KjT)⋅Vj。

- **优点**：因为每个 Warp 都拥有计算最终输出所需的所有信息（自己的 Q 分片和完整的 K、V），它从头到尾都**不需要和其他 Warp 进行任何数据交换**。整个计算流程一气呵成，完全没有了 V1 中的通信瓶颈。

**小结**：通过将“Split-K”变为“Split-Q”，V2 精妙地消除了线程块内部 Warp 之间的通信需求，使得每个 Warp 都能像一条独立的流水线一样工作，极大地减少了共享内存的访问和同步开销，提升了单个计算块的执行效率。

### 优化3. 更精简的算法实现：减少“昂贵”的计算

这个改进点关注的是计算本身的“性价比”。

**背景知识：Matmul FLOPs vs Non-Matmul FLOPs**

现代 GPU 的 Tensor Core 是为矩阵乘法（Matmul）量身定做的“超级加速引擎”，执行 Matmul 操作的效率极高。相比之下，其他的计算，如逐元素的加法、乘法、指数运算（统称 Non-Matmul），虽然在纸面上的 FLOPs（浮点运算次数）可能不高，但因为没有专用硬件，实际执行起来要比 Matmul “昂贵”得多（V2 论文中提到可能高达 16 倍）。

**V2 的算法微调**

V2 对计算公式进行了两处关键的微调，以最大化地利用“便宜”的 Matmul 操作，减少“昂贵”的 Non-Matmul 操作。

1. **在线 Softmax 更新优化**：

- 在 V1 中，更新输出 O 的公式类似于：O_new = scale_1 * O_old + scale_2 * O_current。这意味着在内层循环的**每一步**，都需要进行两次代价较高的逐元素缩放操作。
- 在 V2 中，公式被调整为：O_accum_new = scale_old * O_accum_old + O_current_unscaled。它将“未缩放”的当前块结果直接累加，只对历史累加值进行缩放。真正的最终缩放只在内层循环**结束时执行一次**。这大大减少了中间步骤中 Non-Matmul 操作的数量。

![img](https://pic1.zhimg.com/80/v2-f2a52180f4b505b8945a8f1ffc015b2c_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

1. **后向传播数据存储优化**：

- 为了在后向传播时重计算，V1 需要存储两个统计量：m (行最大值) 和 l (指数和)。
- V2 的作者发现，其实只需要存储一个组合后的统计量 L = m + log(l) (即 log-sum-exp) 就包含了所有必要信息。
- **效果**：从存储两个值变为存储一个值，减少了读写 HBM 的数据量，降低了 IO 开销，同时也简化了后向传播的计算逻辑。

![img](https://pica.zhimg.com/80/v2-363c39fbbbf2b1c99b35883271ccc16d_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

## Flash attention V3

总结：

1. 优化1: 使用warp-specialize 和 persistent kernel
2. 优化2: Pingpong Scheduling，在warpgroup 间，把softmax 和 gemm的计算overlap，高效利用tensor core和 cuda core
3. 优化3: Warpgroup Pipelining， 当前内循环softmax 会和上一次内循环的gemm0 overlap
4. 优化4，支持fp8

### **优化1: 生产者-消费者异步流水线 (Producer-Consumer Asynchrony via Warp-Specialization):**

- **核心思想：** v3 将执行任务的 GPU 线程束（Warps）明确划分为两种角色：
- **生产者 (Producer):** 专门负责数据传输。它使用 Hopper 架构中的**张量内存加速器 (Tensor Memory Accelerator, TMA)** 这一专用硬件单元，以异步方式将 Q, K, V 矩阵块从 HBM 高效加载到 SMEM 中。
- **消费者 (Consumer):** 专门负责计算。它使用**张量核心 (Tensor Cores)** 执行矩阵乘法（GEMM）等计算密集型任务。
- **工作流程：** 生产者加载数据后，通过同步机制通知消费者数据已准备就绪。由于 TMA 的加载操作是异步的，它不会阻塞消费者的计算。消费者在处理当前数据块的同时，生产者已经在预加载下一个数据块。这种流水线作业模式使得数据加载的延迟被计算时间完全或部分地隐藏起来，极大地提升了 GPU 的利用率。

**优化2 重叠和隐藏低效计算 (Overlapping GEMM and Softmax):**

- **问题背景：** Attention 计算包含两种速度悬殊的操作：由 Tensor Cores 执行的、速度极快的矩阵乘法（GEMM），以及由其他单元执行的、速度慢得多的 Softmax 相关计算（如逐元素乘法、指数、求和）。在 v2 中，这些操作是严格按顺序执行的，慢速的 Softmax 会成为整个流程的瓶颈。
- **v3 的解决方案：** v3 设计了精巧的流水线机制，将慢速的 Softmax 计算与快速的 GEMM 计算重叠执行。

1. **跨线程束重叠 (Pingpong Scheduling):** 将不同的线程束组（Warpgroups）进行协调。当一个线程束组在执行 Softmax 时，另一个线程束组则在执行它的 GEMM 计算，两者交替进行，就像打乒乓球一样。这确保了昂贵的计算单元（Tensor Cores）始终处于忙碌状态。

![img](https://picx.zhimg.com/80/v2-034c1fb0f27f746c7de48dc9d4134692_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

1. **线程束内部重叠 (Intra-Warpgroup Pipelining):** 即使在单个线程束组内部，v3 也通过一个“2阶段流水线”设计打破了计算依赖。它会预先计算下一个循环迭代的 S = QK^T (GEMM0)，并将其结果缓存。然后，在当前迭代执行 O = PV (GEMM1) 的同时，对预先算好的下一个 S 进行 Softmax 计算。这样，原本串行的 GEMM0 -> Softmax -> GEMM1 流程被有效地重叠起来，进一步减少了等待时间。

![img](https://pic1.zhimg.com/80/v2-a2c5feccaa7c8aae130ad7b9a0f3ef80_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

### 2. 硬件加速的低精度计算 (FP8 Support)

为了追求极致的计算吞吐量，v3 引入了对 FP8（8位浮点数）格式的支持，这是 Hopper GPU 的一项关键新特性。

- **FP8 Tensor Core 的利用:**
- 通过适配算法以使用 FP8 Tensor Cores，v3 在理论上可以将矩阵计算的吞吐量翻倍。这需要解决 FP8 运算对内存布局的特殊要求，v3 通过核内转置（in-kernel transpose）等技术巧妙地解决了这一问题。
- **为保持精度而引入的新技术:**
- 直接使用 FP8 会带来明显的精度损失，尤其是在处理大语言模型中常见的“异常值”时。为了解决这个问题，v3 引入了两种关键技术：

1. **块量化 (Block Quantization):** 不同于对整个张量使用单一缩放因子的传统方法，v3 对数据的每个小块（block）使用独立的缩放因子进行量化。这使得量化过程更加精细，能更好地适应数据的局部动态范围。
2. **非相干处理 (Incoherent Processing):** 在量化之前，通过乘以一个随机正交矩阵（如哈达玛矩阵）来“打乱”输入数据。这个操作在数学上不改变最终的 Attention 结果，但能有效地将“异常值”的能量分散到整个向量中，从而降低量化误差

![img](https://picx.zhimg.com/80/v2-3b99d0ded560865c1fddb9effbc25d36_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

Flash attention V3 理解

![img](https://picx.zhimg.com/80/v2-200c557c080cc1eae9d08db461557daf_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

- 生产者，只会load一个  Qi  ， 然后逐个加载对应K、V
- 注意K、V 时load到buffer的不同stage中

![img](https://picx.zhimg.com/80/v2-79b3d4e7683201a88b2f358417daac68_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

- 消费者，等待数据，然后进行计算

![img](https://pica.zhimg.com/80/v2-73cf6e3abb5bfd7d5785fcf5ab6a7b73_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

- 等待Q的到达

![img](https://picx.zhimg.com/80/v2-192d6670126871d6deaae3c9a3007ef5_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

- 等待k的到达，然后计算S，注意是ss-gemm

![img](https://picx.zhimg.com/80/v2-bb4a2ef7c59d8e254ef634cf719aed9f_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

- 更新最大值max_val ,同时计算softmax的分子和分母

![img](https://picx.zhimg.com/80/v2-e7c238bdb0df9aee6955edac18a13ea4_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

- 等待V，计算O，注意里的公式有点错误，应该是去更新分子，不应该用 diag(...)^(-1);
- 释放对应的buffer

![img](https://picx.zhimg.com/80/v2-d77fcc97f4ea1461e814d517c6f93869_720w.png?source=d16d100b)





添加图片注释，不超过 140 字（可选）

- 内循环结束，存储数据。