/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <pto/pto-inst.hpp>

#include "backend/urma/urma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

static inline __aicore__ uint32_t recovery_load(__gm__ uint32_t* value) {
  pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(value), sizeof(uint32_t));
  pipe_barrier(PIPE_ALL);
  return *value;
}

static inline __aicore__ void recovery_store(__gm__ uint32_t* value, uint32_t data) {
  *value = data;
  pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(value), sizeof(uint32_t));
  pipe_barrier(PIPE_ALL);
}

static inline __aicore__ void recovery_wait(AsyncCtx& ctx, __gm__ uint32_t* value, uint32_t epoch) {
  CompletionToken token{reinterpret_cast<uint64_t>(value), epoch, COMPLETION_ENGINE_URMA,
                        COMPLETION_TYPE_COUNTER, 0};
  (void)register_completion_condition(ctx, token);
  pto2::detail::defer_flush(ctx);
}

static inline __aicore__ bool inject_recovery_fault(__gm__ int64_t* args, int flags_index, int stats_index,
                                                    int context_index, int mode_index, uint32_t phase) {
  auto* flags = reinterpret_cast<__gm__ Tensor*>(args[flags_index]);
  auto* stats = reinterpret_cast<__gm__ Tensor*>(args[stats_index]);
  auto* comm = reinterpret_cast<__gm__ CommContext*>(args[context_index]);
  auto* local = reinterpret_cast<__gm__ uint8_t*>(flags->buffer.addr) + flags->start_offset;
  auto* evidence = reinterpret_cast<__gm__ uint32_t*>(stats->buffer.addr) + stats->start_offset;
  auto* order = reinterpret_cast<__gm__ uint32_t*>(local + 640);
  if (phase >= 2) {
    const uint32_t stage = phase == 2 ? 0 : 2;
    auto* count = reinterpret_cast<__gm__ uint32_t*>(local + 512 + stage * 4);
    const uint32_t next = recovery_load(count) + 1;
    recovery_store(count, next);
    recovery_store(evidence + stage, next);
    recovery_store(evidence + stage + 3, recovery_load(order) == (stage == 0 ? 0 : 2));
    recovery_store(order, stage + 1);
    return false;
  }
  AsyncCtx ctx = get_async_ctx(args);
  auto* epoch_ptr = reinterpret_cast<__gm__ uint32_t*>(local + 516);
  uint32_t epoch = recovery_load(epoch_ptr);
  if (phase == 0) {
    ++epoch;
    recovery_store(epoch_ptr, epoch);
    recovery_store(evidence + 1, epoch);
    recovery_store(evidence + 4, recovery_load(order) == 1);
    if (comm->rankId == 0) {
      recovery_wait(ctx, reinterpret_cast<__gm__ uint32_t*>(local), epoch);
      return false;
    }
  } else {
    if (comm->rankId == 0 && args[mode_index] != 0 && epoch == 1) {
      // Deliberately malformed completion, after peer entry and before ready notification.
      recovery_wait(ctx, nullptr, epoch);
      return true;
    }
    recovery_store(order, 2);
    if (comm->rankId != 0) return false;
  }
  auto* workspace = reinterpret_cast<__gm__ uint8_t*>(comm->workSpace);
  const uint32_t peer = 1U - comm->rankId;
  const uint64_t offset = reinterpret_cast<uint64_t>(local) - comm->windowsIn[comm->rankId];
  auto* remote = urma_backend::peer_mr_ptr<uint8_t>(workspace, peer, offset);
  for (uint32_t i = 0; i < 4; ++i) reinterpret_cast<__gm__ uint32_t*>(local + 64)[i] = epoch;
  pto::comm::urma::DcciCachelines(local + 64, 16);
  pipe_barrier(PIPE_ALL);
  pto::comm::AsyncSession session;
  if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(workspace, peer, session)) {
    pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    pto2::detail::defer_flush(ctx);
    return true;
  }
  using Payload = pto::GlobalTensor<uint32_t, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<4, 4, 4, 4, 1>>;
  Payload source(reinterpret_cast<__gm__ uint32_t*>(local + 64));
  Payload destination(reinterpret_cast<__gm__ uint32_t*>(remote + 32));
  pto::comm::Signal signal(reinterpret_cast<__gm__ int32_t*>(remote + (phase == 0 ? 0 : 128)));
  auto event = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(
      destination, source, signal, epoch, pto::comm::NotifyOp::Set, session, peer);
  (void)urma_backend::register_urma_async_event(ctx, event, session, workspace);
  if (phase == 0) recovery_wait(ctx, reinterpret_cast<__gm__ uint32_t*>(local + 128), epoch);
  pto2::detail::defer_flush(ctx);
  return false;
}
