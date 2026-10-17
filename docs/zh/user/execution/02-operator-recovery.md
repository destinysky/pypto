# 本机 A5 算子恢复

预备好的分布式程序可以启用 Simpler 的 SDMA 自动恢复：

```python
from pypto.runtime import OperatorRecoveryConfig

compiled = program.compile(x, y, config=config)
with compiled.prepare(
    operator_recovery=OperatorRecoveryConfig(
        replay_safe=("exchange",), max_retries=1, timeout_s=30.0
    )
) as runtime:
    runtime(x, y)
```

`exchange` 是注册的 CHIP 编排函数名。重执行粒度是包含其全部核函数的
L3 task。此声明保证输入在失败后仍有效，部分写入的输出与副作用能够安全重复。
例如，读取不可变输入并完整覆盖私有输出可能满足条件；原址累加或具有外部可见
副作用的发送通常需要先实现恢复或提交协议。未列入 `replay_safe` 的函数保持
`UNKNOWN`，不能重执行；拼错函数名会报错。故障码、张量方向或依赖边不能自动
证明重复执行安全。

PyPTO 从生成代码推导运行与算子身份、参与 rank、window/context 参数绑定。
它先收集各 rank 的派发，验证函数序列一致，再将相同序号组成一个完整 rank 组。
每组依赖前一完整组及显式 task 依赖，原生图仍保留原有张量依赖。恢复模式因此按
完整算子顺序执行。原生 admission 直到全图提交完成才开放，避免早期故障冻结后续
提交、恢复却又等待构图关闭的循环等待。全图须能容纳于配置的原生 task/heap 容量。
不启用时保持原有派发路径。收集编排会增加构图延迟，参数元数据
保留至运行结束。

发生受支持故障后，Simpler 阻止后继执行，对当前 attempt 的全部成员发出带身份
校验的 STOP，收集退役证明，校验重执行资格，恢复 SDMA 资源并重绑定参数，然后
重执行并提交失败组。提交成功后释放原图后继。应用无需手动 STOP 或重新派发。
资格拒绝、恢复失败或预算耗尽时走既有终止流程。

首版支持一个本机 A5 `tensormap_and_ringbuffer` 程序、一个运行期完整通信域、
显式 rank 派发、各 rank 相同 CHIP 函数序列及共享调用配置。拒绝 host callback、
sub-worker task、持久域、host 直接派发 builtin collective、多程序 worker 及跨机
执行；builtin collective 可以放在 CHIP 函数内部。由 window 推导的 context
带有绑定信息，不透明 context 参数不能恢复。此层描述绑定与重复执行语义，不根据
故障码选择传输后端；URMA/CCU 仍需后端实现与验证才能启用恢复。

`tests/st/distributed/test_l3_operator_recovery.py` 编译真实 `@pl.jit.host`
程序：warmup → exchange → finish。测试专用 AIV 外部核在对端进入通知等待后上报
异步 completion 错误；对端需要框架 STOP，下一 attempt 使用新的通知 epoch。
测试检查真实 SDMA 数据、最终数值、执行次数和顺序。它不注入物理网络故障，正式
流程也没有新增故障注入开关。
