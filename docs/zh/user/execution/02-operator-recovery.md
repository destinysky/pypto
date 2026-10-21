# 本地 A5 算子恢复

预配置的分布式程序可启用 Simpler 的 SDMA 或 URMA 自动恢复：

```python
from pypto.runtime import OperatorRecoveryConfig

compiled = program.compile(x, y, config=config)
with compiled.prepare(
    operator_recovery=OperatorRecoveryConfig(max_retries=1, timeout_s=30.0)
) as runtime:
    runtime(x, y)
```

PyPTO 根据操作的副作用和 tensor 别名来源分析每个生成的 L3 task，包含其内部
kernel。读取保留输入并确定性地覆盖输出可以重执行；同时读写同一存储需要恢复协议，
目前拒绝重执行。未知操作、不透明外部 kernel、裸指针和未解析别名均保持不可重执行。
派发前还跨 rank 检查实际参数区间，因此不能通过不同参数名绕过输入输出别名检查。
不再使用人工函数名单声明安全性。未证明安全的程序仍可正常执行，故障时进入终止流程。
缺少编译器副作用证明的旧产物必须重新编译。应用只配置重试策略。

PyPTO 从生成代码推导运行、算子身份、参与 rank 和 window/context 参数绑定。
每个序号的完整 rank 组通过校验后立即提交，host callback 同时继续构建后续组。
组之间只保留显式依赖、重叠区域的 RAW/WAR/WAW 依赖和通信域协议顺序。
共享只读输入和不相交区间不再附加“前一完整组”依赖。未知副作用或未纳入证明的原生输出
参数仍构成保守屏障；原生数据依赖与每个通信域的顺序 token 保留。已证明的纯计算任务
使用 execution-only scope，原生旧 metadata 默认仍包含通信域。

完整 rank 分组和当前 A5 单 frame endpoint 约束仍保留。消费者已经推进、endpoint 仍在
执行或缺少停止证明时，仍可拒绝恢复并 GIVE_UP；删除额外边不代表允许覆盖运行中的执行。

恢复期间原生 publication 冻结，资格判断只检查稳定的已发布前缀，不要求 callback 结束。
后续提交等待当前恢复完成，并在 admission 前应用已提交的绑定世代变更。
保留图仍须适合原生 task/heap 容量；增量构图不代表无限容量。后期 callback 出错将终止
运行，不回滚已经完成的副作用。直接设备控制仍须位于首次 task 提交之前。未启用恢复
时维持原有派发路径。私有 GM 读取若缺少已初始化区间证明，同样拒绝重执行。

故障发生后，Simpler 冻结后续派发，对当前 attempt 的全体成员发送带身份校验的 STOP，
收集退役证明，校验重执行资格，恢复所选传输后端并重新绑定参数，然后重执行并提交失败组。
成功提交后释放原图后继。应用不需手动 STOP 或重新派发。资格拒绝、恢复失败或预算耗尽
均进入既有终止流程。

当前支持一个本地 A5 `tensormap_and_ringbuffer` 程序、一个运行期全通信域、显式 rank
派发、各 rank 相同 CHIP 函数序列及共享调用配置。不支持 host callback、sub-worker、
持久域、host 直接派发 builtin collective、多程序 worker 和跨机执行。builtin collective
可以位于 CHIP 函数内部，但未知副作用不能重执行。window 来源的 context 携带绑定信息，
不透明 context 不可恢复。接口不根据故障码选择传输后端；SDMA 与 URMA 共用协调合同，CCU 恢复尚未实现。

本地 A5 URMA 的 `pld.tensor.get_async(dst, peer=peer, src=src)` 提交真实 URMA GET，
并将 CQE 注册到 Simpler。两个 tensor 必须是同一通信域的直接 window 参数，形状相同、
静态正维度、完整连续、元素可按字节寻址，总大小不超过 256 MiB；peer 必须是远端 rank。
每个独立的顶层 `pl.at(CORE_GROUP)` scope 只提交一次 GET，其 TaskId 在 CQE 成功后才完成。
消费者位于有依赖的后续 task；同一 kernel 内提交后读取 payload 会被编译器拒绝。
读写副作用与重执行安全性仍由编译器推导，无需人工声明。原同步 `get` 的 MTE staging
实现不变，切换 URMA workspace 不会将同步 `get` 自动改写为 URMA。

A5 当前每次构建选择一个 workspace 后端：URMA 使用
`SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON`，默认 SDMA 使用 `OFF`。切换后必须重新构建，
只修改执行时环境不能改变旧二进制。生成的 URMA host orchestration 在首次提交 task 前检查
每个 rank 的实际通信域类型，未启用恢复时也检查。URMA 恢复先证明 PTO 已停止，再通过新鲜、
成功的 CQE 排空原队列，最后提交并原址 rearm，保留 HCOMM/HCCL、注册窗口和队列身份。
该流程不销毁 channel、不 reset device；缺少或错误 CQE 证据会返回恢复失败并进入终止路径。
这些验证不宣称能够恢复物理链路损坏。

`tests/st/distributed/test_l3_operator_recovery.py` 编译真实 PyPTO 程序：
warmup → exchange → finish。业务计算和同步 GET 均从 IR 生成，没有人工安全声明。
只有测试生成的 wrapper 包含故障注入与计数：peer 进入通知等待后报告异步 completion
错误，随后由框架 STOP、恢复和重执行。测试校验 SDMA 数据、最终数值、执行次数和顺序。
该测试不模拟物理网络损坏；正式代码不包含注入开关。

transient 用例在最后一个后继提交之前等待框架观察到故障，仅查询状态，不调用 STOP、
恢复或重新派发。`OPEN_GRAPH_RECOVERY_SEEN` 及两个
`SUBMISSION_BINDING_CURRENT ... generation=2` 日志，分别证明构图期间已开始恢复和后续
提交使用当前世代绑定。原有数值、执行次数和顺序断言仍必须通过。

新增的 `test_urma_program_continues` 编译 `@pl.jit.host` 程序，业务 GET 从 IR 生成真实
URMA 传输，仍验证 warmup → exchange → finish。transient 在 rank 0 发布 ready 通知前注入
空地址 completion 条件，rank 1 必须由框架主动 STOP。这是软件 completion 合同错误，
没有伪造硬件 CQE。必须通过数值、执行次数/顺序、原图继续、一次 replay/commit、CQE drain
及通信身份保持断言。默认构建选择原 SDMA 用例，URMA 构建选择新增 URMA 用例。

SDMA 的 baseline/healthy 各在同一个 prepared worker 上执行四次，前者不启用恢复，后者启用。
`HEALTHY_DISPATCH_LATENCY` 输出去掉第一次预热后的三个 dispatch/wait 时间，排除编译和
worker 初始化，包含通信域生命周期与数据拷贝。这是短时性能对比，不承诺零开销。
