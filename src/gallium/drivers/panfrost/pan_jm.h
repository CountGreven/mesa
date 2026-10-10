/*
 * Copyright (C) 2023 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef __PAN_JM_H__
#define __PAN_JM_H__

#include "pan_jc.h"

struct panfrost_jm_context {
   uint32_t handle;
};

struct panfrost_jm_batch {
   /* Job related fields. */
   struct {
      /* Vertex/tiler/compute job chain. */
      struct pan_jc vtc_jc;

      /* Fragment job, only one per batch. */
      uint64_t frag;
   } jobs;

   /* Bifrost GPU-driven draws: struct poly_heap living in the batch pool and
    * pointing to a growable BO owned by the batch. Varyings of GPU-sized draws
    * are allocated from it by the draw helper kernel, so it must stay alive
    * until the fragment job of this batch is done. 0 until first use.
    */
   uint64_t poly_heap;

   /* Number of GPU-driven draws in the batch. */
   uint32_t gpu_draw_count;
};

#if defined(PAN_ARCH) && PAN_ARCH < 10

#include "genxml/gen_macros.h"
#include "util/u_blitter.h"

struct panfrost_batch;
struct panfrost_context;
struct panfrost_resource;
struct pan_fb_info;
struct pan_tls_info;
struct pipe_draw_info;
struct pipe_grid_info;
struct pipe_draw_start_count_bias;

int GENX(jm_init_context)(struct panfrost_context *ctx);

void GENX(jm_cleanup_context)(struct panfrost_context *ctx);

int
GENX(jm_init_batch)(struct panfrost_batch *batch);

static inline void
GENX(jm_cleanup_batch)(struct panfrost_batch *batch)
{
}

int GENX(jm_submit_batch)(struct panfrost_batch *batch);

static inline void
GENX(jm_prepare_tiler)(struct panfrost_batch *batch, struct pan_fb_info *fb)
{
}

void GENX(jm_preload_fb)(struct panfrost_batch *batch, struct pan_fb_info *fb);
void GENX(jm_emit_fbds)(struct panfrost_batch *batch, struct pan_fb_info *fb,
                        struct pan_tls_info *tls);
void GENX(jm_emit_fragment_job)(struct panfrost_batch *batch,
                                const struct pan_fb_info *pfb);

void GENX(jm_launch_xfb)(struct panfrost_batch *batch,
                         const struct pipe_draw_info *info, unsigned count);

void GENX(jm_launch_grid)(struct panfrost_batch *batch,
                          const struct pipe_grid_info *info);

void GENX(jm_launch_draw)(struct panfrost_batch *batch,
                          const struct pipe_draw_info *info,
                          unsigned drawid_offset,
                          const struct pipe_draw_start_count_bias *draw,
                          unsigned vertex_count);
void GENX(jm_launch_draw_indirect)(struct panfrost_batch *batch,
                                   const struct pipe_draw_info *info,
                                   unsigned drawid_offset,
                                   const struct pipe_draw_indirect_info *indirect);
void GENX(jm_launch_draw_fullscreen)(struct panfrost_batch *batch,
                                     enum blitter_attrib_type type,
                                     const struct blitter_attrib *attrib);

#if PAN_ARCH == 6 || PAN_ARCH == 7
uint64_t GENX(jm_get_poly_heap)(struct panfrost_batch *batch);
void GENX(jm_launch_draw_gpu)(struct panfrost_batch *batch,
                             const struct pipe_draw_info *info, uint64_t cmd,
                             unsigned max_index_count, bool barrier);

/* Split batches before their varying heap or job chain gets too full. */
#define PAN_JM_MAX_GPU_DRAWS_PER_BATCH 1024
#endif

void GENX(jm_emit_write_timestamp)(struct panfrost_batch *batch,
                                   struct panfrost_resource *dst,
                                   unsigned offset);

#endif /* PAN_ARCH < 10 */

#endif
