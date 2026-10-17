/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#include <pto/pto-inst.hpp>
#include "backend/sdma/sdma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

using namespace pto;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *flags = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *stats = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[3]);
    const uint32_t phase = static_cast<uint32_t>(args[4]);
    const uint32_t mode = static_cast<uint32_t>(args[5]);
    AsyncCtx ctx = get_async_ctx(args);
    Tile<TileType::Vec, uint8_t, 1, SDMA_SCRATCH_ALIGNMENT> scratch;
    TASSIGN(scratch, 0x0);
    pto::comm::sdma::detail::UbTmpBuf tmp{};
    pto::comm::sdma::detail::MakeTmpBufferFromTile(scratch, tmp);
    auto *local = reinterpret_cast<__gm__ uint8_t *>(flags->buffer.addr + flags->start_offset);
    auto *evidence = reinterpret_cast<__gm__ uint8_t *>(stats->buffer.addr + stats->start_offset * sizeof(uint32_t));
    using pto::comm::sdma::detail::GetValue;
    using pto::comm::sdma::detail::SetValue;

    // Private, preserved counters are also per-attempt notification epochs.
    // Replay overwrites all business outputs; these counters are test evidence.
    if (phase >= 2) {
        const uint32_t stage = phase == 2 ? 0 : 2;
        const uint32_t expected = phase == 2 ? 0 : 2;
        const uint32_t count = GetValue<uint32_t>(local + 512 + stage * 4, tmp) + 1;
        const uint32_t order = GetValue<uint32_t>(local + 640, tmp);
        SetValue<uint32_t>(local + 512 + stage * 4, tmp, 0, count);
        SetValue<uint32_t>(evidence + stage * 4, tmp, 0, count);
        SetValue<uint32_t>(evidence + (stage + 3) * 4, tmp, 0, order == expected ? 1 : 0);
        SetValue<uint32_t>(local + 640, tmp, 0, stage + 1);
        return;
    }
    uint32_t epoch = GetValue<uint32_t>(local + 516, tmp);
    if (phase == 0) {
        ++epoch;
        SetValue<uint32_t>(local + 516, tmp, 0, epoch);
        SetValue<uint32_t>(evidence + 4, tmp, 0, epoch);
        const uint32_t order = GetValue<uint32_t>(local + 640, tmp);
        SetValue<uint32_t>(evidence + 16, tmp, 0, order == 1 ? 1 : 0);
        if (comm->rankId == 0) {
            save_expected_notification_counter(ctx, local, epoch);
            return;
        }
    } else {
        if (comm->rankId == 0 && mode && epoch == 1) {
            // Peer entry was observed by phase 0. It is now waiting for our
            // publication. Fail before publication so framework STOP is needed.
            pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            pto2::detail::defer_flush(ctx);
            return;
        }
        SetValue<uint32_t>(local + 640, tmp, 0, 2);
        if (comm->rankId != 0) return;
    }
    SetValue<uint32_t>(local + 64, tmp, 0, epoch);
    const uint64_t offset = reinterpret_cast<uint64_t>(local) - comm->windowsIn[comm->rankId];
    const uint64_t target = phase == 0 ? 0 : 128;
    auto *remote = reinterpret_cast<__gm__ uint32_t *>(comm->windowsIn[1 - comm->rankId] + offset + target);
    using ShapeT = Shape<1, 1, 1, 1, 8>;
    using StrideT = pto::Stride<8, 8, 8, 8, 1>;
    GlobalTensor<uint32_t, ShapeT, StrideT> source(reinterpret_cast<__gm__ uint32_t *>(local + 64));
    GlobalTensor<uint32_t, ShapeT, StrideT> destination(remote);
    send_request_entry(ctx, SdmaTput(destination, source, scratch,
                                   reinterpret_cast<__gm__ uint8_t *>(comm->workSpace)));
    if (phase == 0) save_expected_notification_counter(ctx, local + 128, epoch);
}
