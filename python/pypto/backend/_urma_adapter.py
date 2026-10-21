# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Generated A5 URMA GET bridge to the scheduler-owned completion slab."""

URMA_GET_ADAPTER = r"""
static_assert(MAX_COMPLETIONS_PER_TASK == 64, "Update PyPTO's deferred completion budget with the runtime");

static __aicore__ void pypto_urma_get_completion(
    __gm__ int64_t* raw_args, int32_t dst_index, int32_t src_index,
    __gm__ int64_t* dst_context, __gm__ int64_t* src_context,
    int32_t peer, int64_t elements, int64_t width)
{
    AsyncCtx ctx = get_async_ctx(raw_args);
    if (!async_ctx_is_deferred(ctx) || ctx.completion_count == nullptr ||
        ctx.completion_entries == nullptr || ctx.completion_error_code == nullptr) {
        // This adapter may never fall back to an unbounded in-kernel wait.
        trap();
        return;
    }
    if (*ctx.completion_count >= ctx.completion_capacity) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        pto2::detail::defer_flush(ctx);
        return;
    }
    auto* comm = reinterpret_cast<__gm__ CommContext*>(src_context);
    auto* dst = reinterpret_cast<__gm__ Tensor*>(raw_args[dst_index]);
    auto* src = reinterpret_cast<__gm__ Tensor*>(raw_args[src_index]);
    const uint64_t bytes = static_cast<uint64_t>(elements * width);
    const uint64_t source = src->buffer.addr + src->start_offset * width;
    const uint64_t destination = dst->buffer.addr + dst->start_offset * width;
    const uint64_t base = comm->windowsIn[comm->rankId];
    bool valid = dst_context == src_context && peer >= 0 &&
        static_cast<uint32_t>(peer) < comm->rankNum &&
        static_cast<uint32_t>(peer) != comm->rankId && comm->workSpace != 0 &&
        elements > 0 && width > 0 && bytes <= urma_backend::kUrmaMaxTransferBytes &&
        source >= base && destination >= base && bytes <= comm->winSize &&
        source - base <= comm->winSize - bytes && destination - base <= comm->winSize - bytes;
    if (!valid) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        return;
    }
    auto* workspace = reinterpret_cast<__gm__ uint8_t*>(comm->workSpace);
    auto* remote = urma_backend::peer_mr_ptr<uint8_t>(workspace, peer, source - base);
    using Shape = pto::Shape<1, 1, 1, 1, -1>;
    using Stride = pto::Stride<-1, -1, -1, -1, 1>;
    using Payload = pto::GlobalTensor<uint8_t, Shape, Stride>;
    Shape shape(1, 1, 1, 1, bytes);
    Stride stride(bytes, bytes, bytes, bytes, 1);
    Payload output(reinterpret_cast<__gm__ uint8_t*>(destination), shape, stride);
    Payload input(remote, shape, stride);
    (void)send_request_entry(ctx, UrmaTget(output, input, workspace, static_cast<uint32_t>(peer)));
}
"""
