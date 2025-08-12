# 背景

![img](https://pic2.zhimg.com/v2-2d81599663c241174050a31f3959582d_1440w.jpg)

- **[prefill阶段](https://zhida.zhihu.com/search?content_id=245189307&content_type=Article&match_order=1&q=prefill阶段&zhida_source=entity)**：把整段prompt喂给模型做forward计算。**prefill阶段结束后，模型产出第一个token**（例如图中t3）
- **[decode阶段](https://zhida.zhihu.com/search?content_id=245189307&content_type=Article&match_order=1&q=decode阶段&zhida_source=entity)**：**一个token一个token地产出response**。（例如图中逐一产出t4，t5，t6）



### prefill性能评估指标：TTFT

**TTFT（Time To First Token），表示生成第1个token所用的时间，这是prefill阶段的重要评估指标之一**。

### decode性能评估指标：TPOT

**TPOT（Time Per Output Token），产出每一个response token所用的时间，这是decode阶段的重要评估指标之一**。



# PD分离

**在一些推理框架中，prefill和decode是合并在一起的，可以成一块gpu上既做prefill又做decode。**

**在分离式框架中prefill和decode不再共享一块/一群GPU，而是分布在不同的卡**

- **prefill阶段：拥有计算受限的性质（compute-bound）**，特别是在请求流量较大，用户的prompt也比较长的情况下。prefill阶段算完KV cache并发给deocde阶段后，理论上prefill就不再需要这个KV cache了（当然你也可以采用LRU等策略对KV cache的保存做管理，而不是一股脑地清除）。
- **decode阶段**：**拥有存储受限的性质（memory-bound）**，因为token by token的生成方式，decode阶段要频繁从存储中读取KV Cache，同时也意味着它需要尽可能保存KV cache。

如果使用同一个系统或同一组 GPU 来不加区分地处理这两个阶段，会导致严重的资源浪费和性能瓶M颈：

1. **资源利用率低**: 在 Decode 阶段，强大的 GPU 计算单元大部分时间都在“等待”数据从显存加载，造成了“大马拉小车”的局面。昂贵的计算资源被闲置。
2. **吞吐量受限**: 由于 Decode 阶段是串行的且速度慢，它会长时间占用整个 GPU 资源，导致后续新的请求（需要进行 Prefill）必须排队等待，即使 GPU 的计算单元是空闲的。这严重影响了整个系统的吞吐量。

### 分离的好处

1. **极致的硬件利用率和吞吐量**:
   - **Prefill 集群**: 始终处理计算密集型任务，其强大的计算单元得到充分利用。处理完一批请求后立刻释放，服务于下一批，极大提升了处理新请求的效率。
   - **Decode 集群**: 专门处理访存密集型任务。可以用更多的、甚至单卡性能稍弱但显存带宽合适的 GPU 来并发处理成百上千个用户的 Decode 任务，最大化并发数。
2. **降低成本 (Cost-Efficiency)**:
   - 可以为不同阶段选择最合适的硬件。比如，使用少量顶级的、计算能力强的 GPU（如 H100/A100）用于 Prefill 集群，同时使用大量成本更低的、显存带宽尚可的 GPU（如 L40S, A10）用于 Decode 集群。这比用同一种昂贵 GPU 处理所有任务要经济得多。
3. **系统解耦和专业化**:
   - 每个集群的软件栈、调度算法、批处理策略都可以被独立优化。
   - Prefill 集群可以使用简单的静态批处理。
   - Decode 集群可以使用更复杂的连续批处理（Continuous Batching）来动态管理不同用户的生成过程。
4. **降低延迟**: 虽然在两个集群间交接状态会引入微小的网络延迟，但由于 Prefill 集群可以快速处理完并移交任务，用户能更快地收到第一个 Token（Time to First Token, TTFT）。同时，整个系统的吞吐量大大提高，请求的排队等待时间也显著缩短。



# PD分离中 KV cache的传递方法

根据目前一些框架的形式来看，主要有**中心存储**和**离散式**（分布式）两种方式，当然也可以是两者的结合。

- 所谓中心存储就是建立一个跨设备的KV store，由它统一管理KV值，包括KV的增、删、查、传递等工作，推理实例（P/D）连接KV store后只需负责往里面增、查数据。
- 分布式：P2P的方式进行数据传递，各个实例分别管理自己的存储，比如一个P实例计算完成后，向目标D实例建立通信完成KV值传递，这种方式没有统一的传递中介。

**中心化方案**更像是一种“云原生”的思路，它拥抱解耦和弹性，但在性能上做出了妥协。

- **极致的解耦与灵活性**: 这是最大的优点。Prefill 和 Decode 集群完全解耦。它们可以独立地进行扩缩容，甚至可以使用完全不同型号的硬件。Decode Worker 不需要关心是哪个 Prefill Worker 生成了缓存。
- 简化的故障恢复: 由于状态（KV Cache）存储在可靠的中心化系统中，如果一个 Decode GPU 发生故障，调度器可以轻松地将任务重新分配给另一个健康的 GPU，后者只需从中心缓存重新下载数据即可，无需重新执行昂贵的 Prefill。
- **网络延迟和带宽瓶颈**: 这是最致命的缺点。KV Cache 的体积可能非常大（从几百MB到数GB）。通过常规网络（即使是高速网络）在 Prefill GPU -> 中心缓存 -> Decode GPU 之间传输如此大的数据块，会引入显著的延迟。这可能会完全抵消掉分离架构带来的性能优势，尤其对于延迟敏感的应用。

**分布式方案**则更像是高性能计算（HPC）的思路，它追求极致的硬件效率和最低的通信开销，但牺牲了一部分灵活性和容错能力。

- **极致的低延迟和高带宽**: 这是其核心优势。利用 NVLink 等现代 GPU 互联技术，GPU 之间的显存传输速度远高于常规网络。这使得 KV Cache 的交接速度非常快，最大限度地减少了 Prefill 到 Decode 的切换开销。
- **紧密的硬件和软件耦合**: Prefill 和 Decode Worker 之间需要有高速直连通道，这限制了硬件的部署拓扑。系统对 GPU 之间的物理连接有很强依赖。
- **显存占用 (Memory Pressure)**: KV Cache 占用了宝贵的 GPU HBM。在整个 Decode 过程中，这部分显存都被锁定，可能会限制单个 GPU 能并发处理的请求数量。

###  KV cache传递一定是只有P端到D端吗？

PD分离多对多模式，P和D之间任意映射都要能完成。在vLLM 有prefix cache功能，可以利用历史计算数据，存在一种情况：D端的block保存的数据，P端是否能够利用？根据prefix cache的block管理来看（如下图）这完全是可能的。所以P可以从D拉取已计算数据。

