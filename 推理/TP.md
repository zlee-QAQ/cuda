# TP

基本思想就是把模型的参数纵向切开，放到不同的GPU上进行独立计算，然后再做聚合。

设输入数据为X，参数为W。X的维度 = (b, s, h)，W的维度 = (h, h')。其中：

- b：batch_size，表示批量大小
- s：sequence_length，表示输入序列的长度
- h：hidden_size，表示每个token向量的维度。
- h'：参数W的hidden_size。

则每次forward的过程如下：

![img](https://pic1.zhimg.com/80/v2-dedf8fc07e9f033d2af0d4b9216382ec_720w.png)



## 按照行切分weight

forward

![img](https://picx.zhimg.com/80/v2-614eb741337e894a3c82b59b9e3cc69a_720w.png)



backward

![img](https://picx.zhimg.com/80/v2-d1adbc89c641f3ed16f3cc3b81641236_720w.png)





添加图片注释，不超过 140 字（可选）

- **g** **的backward**：假定现在我们要对 Wi 求梯度，则可推出 
  $$
  \frac{\partial L}{\partial W_i} =  \frac{\partial L}{\partial Y} * \frac{\partial Y}{\partial Y_i} * \frac{\partial Y_i}{\partial W_i} = \frac{\partial L}{\partial Y} * \frac{\partial Y_i}{\partial W_i} 
  $$
  ，也就是说，只要把 ∂L∂Y 同时广播到两块GPU上，两块GPU就可以**独立计算**各自权重的梯度了。
- **f** **的backward**：在上图中，我们只画了模型其中一层的计算过程。当模型存在多层时，梯度要从上一层向下一层传播。比如图中，梯度要先传播到X，然后才能往下一层继续传递。这就是f 的backward的作用。这里也易推出，
  $$
  \frac{\partial L}{\partial X} =  concat[\frac{\partial L}{\partial X_1}, \frac{\partial L}{\partial X_2}] 
  $$
  

## 按照列进行切分

forward

![img](https://picx.zhimg.com/80/v2-816521c9790ea0b10060d0c5da68e640_720w.png)

backward

![img](https://picx.zhimg.com/80/v2-3cdb1514e7fc794f34d3917aa36d2c80_720w.png)



- g的backward：易推出
  $$
  \frac{\partial L}{\partial W_i} =  \frac{\partial L}{\partial Y_i} * \frac{\partial Y_i}{\partial W_i} 
  $$
  
- f 的backward：因为对于损失L，X既参与了XW1的计算，也参与了XW2的计算。因此有 
  $$
  \frac{\partial L}{\partial X} =  \frac{\partial L}{\partial X}|1 + \frac{\partial L}{\partial X}|2 
  $$
   。其中  表示
  $$
  \frac{\partial L}{\partial X}|i 
  $$
  第i块GPU上计算到X时的梯度。

## MLP层

![img](https://pic1.zhimg.com/80/v2-c3224a61cb02c1461f5922ca5e60560a_720w.png)





![img](https://pic1.zhimg.com/80/v2-14528a81bd679afb0d6cb53382d8b6df_720w.png)





在MLP层中，**对A采用“列切割”，对B采用“行切割”**。

- f 的forward计算：把输入X拷贝到两块GPU上，每块GPU即可独立做forward计算。
- g 的forward计算：每块GPU上的forward的计算完毕，取得Z1和Z2后，GPU间做一次**AllReduce**，相加结果产生Z。
- g 的backward计算：只需要把 ∂L∂Z 拷贝到两块GPU上，两块GPU就能各自独立做梯度计算。
- f 的backward计算：当当前层的梯度计算完毕，需要传递到下一层继续做梯度计算时，我们需要求得 ∂L∂X 。则此时两块GPU做一次**AllReduce**，把各自的梯度 ∂L∂X|1 和 ∂L∂X|2 相加即可。

为什么我们对A采用列切割，对B采用行切割呢？**这样设计的原因是，我们尽量保证各GPU上的计算相互独立，减少通讯量**。对A来说，需要做一次GELU的计算，而GELU函数是非线形的，它的性质如下：

![img](https://picx.zhimg.com/80/v2-55670752a71e13e1bbfde17561d1943c_720w.png)





如果对A采用行切割，我们必须在做GELU前，做一次AllReduce，这样就会产生额外通讯量。但是如果对A采用列切割，那每块GPU就可以继续独立计算了。

### 2.2 MLP层的通讯量分析

由2.1的分析可知，MLP层做forward时产生一次AllReduce，做backward时产生一次AllReduce。AllReduce的过程分为两个阶段，Reduce-Scatter和All-Gather，每个阶段的通讯量都相等。

现在我们设每个阶段的通讯量为 Φ ，则一次AllReduce产生的通讯量为 2Φ 。MLP层的总通讯量为 4Φ 。 根据上面的计算图，我们也易知， Φ=b∗s∗h



## Self-Attention层

![img](https://pica.zhimg.com/80/v2-8299f5464a5e20ea7820a35e88008483_720w.png)



- 对三个参数矩阵Q，K，V，**按照“列切割”**，每个头放到一块GPU上，做并行计算。
- 对线性层B，**按照“行切割”**。

## MHA 计算过程对比

为了更好地理解张量并行（TP）的机制，我们首先回顾在单个设备（如单个GPU）上标准的多头注意力（MHA）是如何计算的，然后将其与张量并行的版本进行对比。

### A. 非张量并行（单GPU）下的MHA计算过程

在这种标准情况下，所有的计算和权重都存在于一个设备上。

#### 1. 参数定义

| **符号** | **含义**                                     | **示例维度** |
| -------- | -------------------------------------------- | ------------ |
| `B`      | Batch Size (批处理大小)                      | 4            |
| `S`      | Sequence Length (序列长度)                   | 1024         |
| `H`      | Hidden Dimension (模型隐藏层维度)            | 4096         |
| `N`      | Number of Attention Heads (注意力头数)       | 32           |
| `d_h`    | Dimension per Head (每个头的维度)            | 128          |
| `H_attn` | Total Attention Dimension (注意力机制总维度) | N×dh=4096    |

#### 2. 计算步骤

1. **输入投影 (Q, K, V)**

   - 输入 X 与完整的权重矩阵 Wq,Wk,Wv 相乘，得到 Q, K, V。
     $$
     Q=X⋅Wq
     $$

     $$
     K=X⋅Wk
     $$

     $$
     V=X⋅Wv
     $$

     

   - **形状变化**:

     - $$
       X: [S,B,H]
       $$

       

     - $$
       Wq,Wk,Wv: [H,Hattn]
       $$

       

     - $$
       Q,K,V: [S,B,Hattn]
       $$

       

2. **多头拆分 (Split Heads)**

   - 将 Q, K, V 的总注意力维度 `H_attn` 拆分成 `N` 个头。

   - **形状变化**:

     - $$
       从 [S,B,Hattn] Reshape 和 Transpose 为 [B,N,S,dh]。
       $$

       

3. **计算缩放点积注意力**

   - 在所有 N 个头上并行计算注意力。
     $$
     AttentionScores=softmax(Q·K^T)
     $$

     $$
     Output=AttentionScores⋅V
     $$

     

   - **形状变化**:

     - `Q`: [B,N,S,dh], `K^T`: [B,N,dh,S], `V`: [B,N,S,dh]
     - `Output`: [B,N,S,dh]![img]()

4. **多头合并 (Concatenate Heads)**

   - 将所有头的输出重新拼接在一起。
   - **形状变化**:
     - 从 [B,N,S,dh] Transpose 和 Reshape 回 [S,B,Hattn]。

5. **输出投影**

   - 将合并后的注意力输出通过最终的投影层 Wo 转换回模型的隐藏维度。

     FinalOutput=Output⋅Wo

   - **形状变化**:

     - `Output`: [S,B,Hattn]

     - Wo: [Hattn,H]

     - `FinalOutput`: [S,B,H]

       

### B. 张量并行（TP）下的MHA计算过程

与单GPU情况不同，张量并行将计算和存储分摊到多个设备上。

#### 1. 核心思想

张量并行的核心思想是**将巨大的权重矩阵切分到多个计算设备（如 GPU）上**，让每个设备只负责计算矩阵的一部分，从而使得单个设备无需加载整个模型。对于 MHA，我们通常采用**列并行（Column Parallelism）**处理输入投影，和**行并行（Row Parallelism）**处理输出投影。

#### 2. 参数定义

除了上述参数，我们增加并行度的定义：

| **符号** | **含义**                            | **示例维度** |
| -------- | ----------------------------------- | ------------ |
| `P`      | Parallelism Degree (并行度/GPU数量) | 4            |

#### 3. 计算步骤

1. **输入投影 (Q, K, V) - 列并行**

   - 输入张量 `X` 被复制到所有 `P` 个 GPU 上。权重矩阵 Wq,Wk,Wv 沿着**列**被切分。
   - **权重切分**: 完整的 Wq ([H,Hattn]) 被切分为 `P` 份 Wqi ([H,Hattn/P])。
   - **并行计算**: 每个 GPU `i` 计算分片结果 Qi=X⋅Wqi。
   - **形状变化**:
     - `X`: [S,B,H]
     - Wqi: [H,Hattn/P]
     - Qi,Ki,Vi: [S,B,Hattn/P]

2. **多头拆分 (Split Heads)**

   - 每个 GPU 上的分片结果被独立拆分成 N/P 个头。
   - **形状变化**:
     - 从 [S,B,Hattn/P] 变为 [B,N/P,S,dh]。

3. **独立计算缩放点积注意力**

   - 每个 GPU 独立地为其负责的 N/P 个头计算注意力，此步骤无需通信。

     Outputi=softmax(dhQi⋅KiT)⋅Vi

   - **形状变化**: `Output_i` 的形状为 [B,N/P,S,dh]。

4. **多头合并 & 输出投影 - 行并行**

   - **多头合并**: 每个 GPU 上的 `Output_i` 被 Reshape 回 [S,B,Hattn/P]。
   - **权重切分**: 输出投影矩阵 Wo ([Hattn,H]) 沿着**行**被切分为 `P` 份 Woi ([Hattn/P,H])。
   - **并行计算**: 每个 GPU 计算一个部分输出 PartialOutputi=Outputi⋅Woi。
   - **形状变化**:
     - `Output_i`: [S,B,Hattn/P]
     - Woi: [Hattn/P,H]
     - `PartialOutput_i`: [S,B,H]

5. **结果合并 (All-Reduce)**

   - 通信: 使用 All-Reduce 操作将所有 GPU 上的 $PartialOutput_i$ 相加，得到最终结果。

     $FinalOutput=∑PartialOutputi$

   - **最终输出**: 每个 GPU 都得到一份完整的、形状为 [S,B,H] 的最终输出。

## word embedding

Embedding层一般由两个部分组成：

- **word embedding**：维度(v, h)，其中v表示词表大小。
  - 词表可能很大，需要拆分到不同的卡上
- **positional embedding**：维度(max_s, h)，其中max_s表示模型允许的最大序列长度。
  - max_s本身不会太长，因此每个GPU上都拷贝一份

![img](https://picx.zhimg.com/80/v2-705dc3c1bb790dc3c4fa66f0e0855a8e_720w.png)



对于输入X，过word embedding的过程，就是等于用token的序号去word embedding中查找对应词向量的过程。例如，输入数据为[0, 212, 7, 9]，数据中的每一个元素代表词序号，我们要做的就是去word embedding中的0，212，7，9行去把相应的词向量找出来。

假设词表中有300个词，现在我们将word embedding拆分到两块GPU上，第一块GPU维护词表[0, 150)，第二块GPU维护词表[150, 299)。

- 当输入X去GPU上查找时，
- 能找到的词，就正常返回词向量，
- 找到不到就把词向量中的全部全素都置0。
- 按此方式查找完毕后，每块GPU上的数据做一次AllReduce，就能得到最终的输入



## 张量模型并行 + 数据并行


到这里为止，我们基本把张量模型并行的计算架构说完了。在实际应用中，对Transformer类的模型，采用最经典方法是张量模型并行 + 数据并行，并在数据并行中引入ZeRO做显存优化。具体的架构如下：

![img](https://picx.zhimg.com/v2-0d097babc07ab0846b46c75254152703_1440w.jpg)


其中，node表示一台机器，**一般我们在同一台机器的GPU间做张量模型并行。在不同的机器上做数据并行**。图中颜色相同的部分，为一个数据并行组。凭直觉，我们可以知道这么设计大概率和两种并行方式的通讯量有关。具体来说，**它与TP和DP模式下每一层的通讯量有关，也与TP和DP的backward计算方式有关**。我们分别来看这两点。

