# MHA、MQA、GQA

![img](https://pic2.zhimg.com/v2-2e5067f40a181904c47e77a6f4f2e213_1440w.jpg)

- **在MQA的情况下，一个token所有的heads都共享同一个k和v**。这样在降低param weights大小的同时，还让原本需要保存num_heads份的kv cache降低为只需保存1份。
- 但是，MQA可能造成模型效果上的损失，毕竟原来对于1个token，它的每个head都有各自的k、v信息的，现在却被压缩为一份。**所以GQA作为一种折衷的方案出现了**，即将1个token的head分成num_group组，每个group内共享同一个k，v信息，使得信息压缩不像GQA那样严重。

### 1.2 MLA的整体设计思想


⚠️：**在本节中，我们会以K cache为例，抽象出MLA的核心优化思想**。V cache的优化思想也是同理，但不在这节赘述，而是合并到后文对于MLA的细节讲解中。
现在先让我们回到MHA上（图1.1），来思考一个问题：**为什么对于一个token，我们需要保存它所有heads上的K值作为K cache呢？**


主要原因我们在上文解释过：**这是因为每个k_head附带有不同的信息，它将用这份独有的信息和对应的q_head进行attn的计算**，用公式表示即为$$attn\_weights = (W_{Q}h_{i})^{T} * (W_{K}h_{j})$$，这里的$$W_{Q}, W_{K$$是合并了所有head对应的param weight后的表达。


**我们现在的总目标是节省K cache，当你再次端详上面这幅图时，一个idea在你的头脑中出现：**

![img](https://pic1.zhimg.com/v2-0e7c55f4198b80792a17b5e579903fba_1440w.jpg)

- 当前我要存的K cache是4个k_head（图中深绿色框），**但如果我能从这4个k_head中抽取出1份共有的信息**，然后在做attn计算时，**每个head都用这1份共有的信息做计算**，那么我也只需存这1份共有信息作为K cache了。这样我就**把K cache从原来num_heads = 4变成num_heads = 1**，这不就能节省K cache了吗？
- 但是等等，**现在共有的k_head信息是抽取出来了，那么相异的k_head信息呢？**（**简单来说，就是由****不同head部分学习到的相异信息**）。我们当然是希望k_head间相异的信息也能保留下来，那么该把它们保留至哪里呢？当你回顾attn_weights的计算公式时，一个想法在你脑中闪现：**q部分不是也有heads吗！我可以把每个k_head独有的信息转移到对应的q_head上吗！写成公式解释就是：**
  - 原来 ，括号表示运算顺序，即先各自算2个括号内的，再做 * 计算
  - 现在 ，同理括号表示运算顺序。
  - **也就是说，这里我们通过矩阵乘法的交换律，巧妙地把1个token上k_heads独有的信息转移到了对应的q_head上来，这样1个token上k_heads间共享的相同信息就能被我们当作K cache存储下来。**





![img](https://pica.zhimg.com/v2-2c01ac514e4559fc090c347243c5ad96_1440w.jpg)

- **对于每个token的k_heads，我们需要抽取出它们的相异信息**，而这个相异信息本质上是由 维护的。观测到所有tokens都共享1个 ，所以我们对于q_heads，我们只需做1次对于 的吸收，就能统一获取所有tokens的所有k_heads上的相异信息。
- **对于每个tokens的k_heads，我们还需要抽取出它们的相同信息**，而这个相同信息应该是每个tokens的所有k_heads共享一份，同时不在不同tokens间共享。那么我们自然而然想到，可以学习一个linear参数矩阵，从原始token 中提取出这份共有信息，以此作为我们的K cache。而不管是从“信息提取”还是从“进一步节省K cache大小”的角度来说，似乎这个linear参数参数矩阵如果能把 压缩到一个更低维的空间，会收获更紧密的信息表达和更小的存储量，这也是图中compress_k的由来。
- **最后，我们使用压缩后了共有信息的compress_k，和吸收了相异信息的q_head做计算，得到attn_weights**。



对v cache的优化也是同理，这里额外提几点：

- 事实上，**当我们考虑到v cache优化时，上图中的compress_k其实应该被理解成[compress_kv](https://zhida.zhihu.com/search?content_id=252939704&content_type=Article&match_order=1&q=compress_kv&zhida_source=entity)**，**也就是它是1个token所有k_heads和v_heads的共有信息**。
- $W_V$ 可以和$W_O$ 作吸收，我们在后文会讲这块细节。

# MLA的运作流程

### **2.1 CD** (**CacheDecompressed, dpsk MLA的原生实现**）

现在我们可以来看MLA的运作细节了。

![img](https://pic2.zhimg.com/v2-83cd2dc4a4b6541cb9822738d0e5d42d_1440w.jpg)



- 这里假设q_len = 1，kv_len = 1024，nope表示非pe部分的head_dim，rope表示pe部分的head_dim。其余维度已标注在图中。其中红色表示param_weights，其中：
  - `q_b_proj`：是q计算中的升维矩阵，它包含了 两部分，分别表示对q的nope/rope部分的计算。
  - **`kv_a_proj_with_mqa`：是对原始hidden_states的压缩矩阵，它包含了****两部分，分别用于计算compress_kv（即抽取k_heads和v_heads的共同信息）**，以及计算k_pe的部分。
  - `kv_b_proj`：它包含了 两部分，分别表示对 k_nope 和 v 部分的计算。
  - **以上符号表示皆遵从dpsk原始论文，下标****表示Down降维，****表示Up升维，****表示做Rope（诸如****就表示和K的rope相关）。**

好，现在关于这个MLA的原生实现，我们来讨论几个有意思的点：


**（1）在MLA中，每个head_dim的尺寸更大了**。观察到原始hidden_size = 5120，如果按照num_heads = 128来看的话，正常来说一个head_dim = 40 (5120/128=40)。但是在MLA中，一个head_dim = 128，远大于40。也就说MLA其实是用比一般MHA更大的head_dim（或者也可能是num_heads）来做attn计算的，然后在最终的 矩阵中映射回原来的hidden_size。对此我个人给出一些简单猜测：**如果推理阶段KV cache造成的memory bound的问题已经得到解决的话，那么训练时我就能少一点后顾之忧，然后通过提升模型的复杂度来取得与MHA比肩或更好的效果（训练阶段还有别的优化方式）。这样当我回到推理阶段时，我的整体计算强度就上去了（每读1次，算的次数更多了）只要没有达到compute bound的界限，这样的提升就是有好处的**。


**（2）原生MLA的计算最终展开成了MHA的计算**。这一点可以参见图中q（蓝色），k（绿色），v（黄色），它们最终都变成了标准MHA的计算。从理论上来说，这一点也不奇怪，因为我们在第一部分说过MLA就是MHA的变种，只是它在MHA的基础上做了信息从k/v_head向q_head的转移。**嗯?!!但是等等，从上图这个原生MLA上来看，虽然产出了compress_kv，但是好像并没有做什么信息转移呀**，也就是粗糙来看目前的计算流程还是 而不是转移后的 呀：

- 是的，如果你有这个疑惑，**那么恭喜你发现了原生MLA的问题，也就是它没有做任何的信息转移。**
- 同时，原生MLA保存的KV cache并不是图中绘制的compress_kv，而是图中已经成形的完整的k（绿色）和v（黄色），这一点在上面的代码中可以看见。
- **再有，考虑到这里head_dim = 128（远大于同num_heads数量下的标准head_dim=40），所以原生MLA增加算力所付出的代价是，KV cache显存反而增加了。**



基于这些，我们管原生MLA的实现方式为**CD**（**CacheDecompressed**），即存储的KV cache是没有经过任何压缩的。我们马上就来看后一些做过“信息转移/吸收”的优化方法，不过在此之前，我们先对原生MLA的计算量和KV cache做一个分析。



### **2.2 CC** (**CacheCompressed**）

好，在进入大家从第一部分开始就心心念念的“k/v_head信息向q转移（或者理解成被q吸收）”这个优化介绍前，我们先介绍基于原生实践和这个优化的一个中间态：**CC** (**CacheCompressed**）。**在这个中间态中，我们终于是以compress_kv为kv cache了，但是我们没做任何吸收。之所以要介绍这个中间态，是方便大家更好感受“吸收”的好处**。


我们直接对着2.1的图，列出CC表格：

![img](https://pic4.zhimg.com/v2-d586b08e145d3602fedf5923b0791797_1440w.jpg)

CC


不难发现，在这个中间态CC优化的MLA下：

- 单token KV cache = 1.13 KB ，相比CD有了显著降低。
- 单token的kv计算量 = 33.55 + 0.05 + 0.03 = 33.63 MFLOPs。主要犯罪嫌疑人就在`kv_b_proj`上。简单来说，在没有做吸收/转移前，一个 矩阵需要作用在kv_len = 1024条数据上，但是现在它只需要被q_len=1条数据算1次就好了，即我们把属于kv的计算量转移到了q上。

###   **2.3** **A_CC（AbsorbCacheCompressed）**

现在，终于来到我们心心念念的涉及吸收的优化了

![img](https://pic1.zhimg.com/v2-4b5252182f3873d19ee88e8792d608a0_1440w.jpg)

A_CC

- 单token KV cache = 1.13 KB
- 单token的KV计算量 = 0.15 + 0.13 = 0.28 MFLOPs
- 达到了节省KV cache的同时，维持单token KV计算量不变的需求。

这里解释下为什么A_CC相比于CC，总计算量降低了很多，但单token计算量却没有变化：

- 这是因为单token计算量分成作用在q和作用在kv上的。而q对应的seq_len = 1，kv对应的seq_len=1024
- A_CC相比于CC，把原来属于单kv的计算量转移到q上了，而q的seq_len=1，对总计算量的影响本来就少。

###   **2.4 A_CC_ME**


最后，这个优化其实就是在A_CC的基础上，在计算attn_weights的时候，把nope和rope的部分拆开算，然后再求和。这样做是为了避开无用的数据拷贝和广播（可以看代码，你会发现A_CC为了做数据拼接，是先初始化一个一个拼接好的空张量，再往里塞数据，这样就是2倍的显存开销。而避开拼接各自算各自的，可以直接复用已有的数据）

![img](https://pic4.zhimg.com/v2-b9d4e162f3a8cba7d57f396bc8076f53_1440w.jpg)





# 总结

- CD 在并没有减少KVcache的存储，但是计算量最少，这是因为每次 kv cache 不需要重新计算
- CC 减少了KV chche的存储，但是计算量上升非常多，每次需要完整的重新计算K、Vcache
- A_CC 是在CC的基础上增加了矩阵吸收，计算量相对于CC大量下降，下降的主要原因在于K、V cache的升维计算的改变。
  - CC 中，$attn\_weights = (W_{Q}h_{i})^{T} * (W_{K}h_{j})$
  - A_CC中，$ attn\_weights = (h_{i}^{T}W_{Q}^{T}W_{K}) * h_{j}$
  - CC 中 $W_{K}h_{j}$ 是[h1, h2] @[h2,seqlen]， Acc中，$(h_{i}^{T}W_{Q}^{T}W_{K}) * h_{j}$ 是[1, h2] @[h2,seqlen], 所以计算量下降很多