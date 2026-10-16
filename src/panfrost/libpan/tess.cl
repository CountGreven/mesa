/*
 * Copyright 2026 XclipseTools
 * Copyright 2026 Fredrik Lindlöf
 * SPDX-License-Identifier: MIT
 *
 * Kernel layout from JimVulkan/mesa-unified-drivers (MIT).
 */

/*
 * Tessellation on the job manager, which has no tessellation stage (pan_gs.h): the
 * fixed-function tessellator from src/poly runs as precompiled kernels between the tess
 * control compute job and the tess eval job.
 *
 *   panlib_tess_*         (COUNT)        one workgroup per patch: its index count
 *   panlib_tess_prefix_sum               one thread: prefix sum, and every buffer the rest of
 *                                        the draw writes, sized to the real count
 *   panlib_tess_*         (WITH_COUNTS)  one workgroup per patch: domain points and indices
 *
 * The tess eval shader then runs as a geometry shader, once per tessellator index, and the
 * tiler draws its vertices in order: both jobs get their size here.
 */

#include "compiler/libcl/libcl.h"
#include "genxml/gen_macros.h"
#include "lib/pan_encoder.h"
#include "poly/cl/tessellator.h"
#include "compiler/pan_gs.h"

#if PAN_ARCH == 6 || PAN_ARCH == 7

KERNEL(1)
panlib_tess_isoline(constant struct poly_tess_params *p, uint32_t mode)
{
   if (cl_global_id.x < p->nr_patches)
      poly_tess_isoline_process(p, cl_global_id.x, mode);
}

KERNEL(1)
panlib_tess_tri(constant struct poly_tess_params *p, uint32_t mode)
{
   if (cl_global_id.x < p->nr_patches)
      poly_tess_tri_process(p, cl_global_id.x, mode);
}

KERNEL(1)
panlib_tess_quad(constant struct poly_tess_params *p, uint32_t mode)
{
   if (cl_global_id.x < p->nr_patches)
      poly_tess_quad_process(p, cl_global_id.x, mode);
}

static void
panlib_tess_set_job_type(global uint8_t *job, bool run, enum mali_job_type type)
{
   struct mali_job_header_packed hdr;
   pan_pack(&hdr, JOB_HEADER, cfg) {
      cfg.type = run ? type : MALI_JOB_TYPE_NULL;
   }
   pan_merge((global struct mali_job_header_packed *)job, &hdr, JOB_HEADER);
}

static uint64_t
panlib_tess_alloc(global struct poly_heap *heap, uint size_B)
{
   return size_B ? (uint64_t)poly_heap_alloc(heap, size_B, 64) : 0;
}

/*
 * Serial on purpose: one thread, one pass over the patch counts.
 *
 * gs_out: pan_gs_params::out_pos, the three fields after it being out_psiz, out_general and
 * out_index. count: where the tess eval job reads the number of its invocations
 * (pan_gs_params::num_prims_addr). tes_job: the tess eval COMPUTE job, tiler_job its TILER job
 * (or NULL with rasterizer discard), tiler_bufs the tiler's varying buffers (general, position,
 * point size).
 *
 * When the heap cannot take the draw, the count is 0 and both jobs become NULL jobs: the draw
 * is dropped instead of overrunning memory.
 */
KERNEL(1)
panlib_tess_prefix_sum(global struct poly_tess_params *p, global uint64_t *gs_out,
                       global uint32_t *count, global uint8_t *tes_job,
                       global uint8_t *tiler_job,
                       global struct mali_attribute_buffer_packed *tiler_bufs,
                       uint32_t out_stride, uint32_t writes_psiz)
{
   global struct poly_heap *heap = p->heap;

   uint total = 0;
   for (uint i = 0; i < p->nr_patches; i++) {
      total += p->counts[i];
      p->counts[i] = total;
   }

   /* Every invocation of the last 64-wide workgroup of the tess eval job owns slots. */
   const uint slots = align(total, 64);
   const uint idx_B = total * 4;
   const uint pos_B = slots * 16;
   const uint gen_B = slots * out_stride;
   const uint psiz_B = writes_psiz ? align(slots * 2, 4) : 0;
   const uint gs_idx_B = slots * 4;
   /* Domain points, allocated by the WITH_COUNTS pass: at most one per index. */
   const uint points_B = total * sizeof(struct poly_tess_point);
   const ulong need = (ulong)idx_B + pos_B + gen_B + psiz_B + gs_idx_B + points_B + 6 * 64;

   if (total == 0 || heap->bottom + need >= heap->size) {
      for (uint i = 0; i < p->nr_patches; i++)
         p->counts[i] = 0;
      total = 0;
   }

   if (total) {
      p->index_buffer = (global uint32_t *)panlib_tess_alloc(heap, idx_B);
      gs_out[0] = panlib_tess_alloc(heap, pos_B);
      gs_out[1] = panlib_tess_alloc(heap, psiz_B);
      gs_out[2] = panlib_tess_alloc(heap, gen_B);
      gs_out[3] = panlib_tess_alloc(heap, gs_idx_B);
   }
   *count = total;

   panlib_tess_set_job_type(tes_job, total != 0, MALI_JOB_TYPE_COMPUTE);
   if (total) {
      pan_pack_work_groups_compute(
         (global struct mali_invocation_packed *)(tes_job +
                                                  pan_section_offset(COMPUTE_JOB, INVOCATION)),
         slots / 64, 1, 1, 64, 1, 1, false, false);
   }

   if (tiler_job == NULL)
      return;

   panlib_tess_set_job_type(tiler_job, total != 0, MALI_JOB_TYPE_TILER);
   if (!total)
      return;

   pan_pack_work_groups_compute(
      (global struct mali_invocation_packed *)(tiler_job +
                                               pan_section_offset(TILER_JOB, INVOCATION)),
      1, total, 1, 1, 1, 1, true, false);

   global struct mali_primitive_packed *prim =
      (global struct mali_primitive_packed *)(tiler_job +
                                              pan_section_offset(TILER_JOB, PRIMITIVE));
   pan_unpack(prim, PRIMITIVE, unpacked_prim)
      ;
   pan_pack(prim, PRIMITIVE, cfg) {
      memcpy(&cfg, &unpacked_prim, sizeof(cfg));
      cfg.index_count = total;
   }

   global struct mali_draw_packed *dcd =
      (global struct mali_draw_packed *)(tiler_job + pan_section_offset(TILER_JOB, DRAW));
   pan_unpack(dcd, DRAW, unpacked_dcd)
      ;
   pan_pack(dcd, DRAW, cfg) {
      memcpy(&cfg, &unpacked_dcd, sizeof(cfg));
      cfg.position = gs_out[0];
   }

   if (unpacked_prim.point_size_array_format != MALI_POINT_SIZE_ARRAY_FORMAT_NONE) {
      pan_pack((global struct mali_primitive_size_packed *)(tiler_job +
                                                            pan_section_offset(
                                                               TILER_JOB, PRIMITIVE_SIZE)),
               PRIMITIVE_SIZE, cfg) {
         cfg.size_array = gs_out[1];
      }
   }

   /* general, position, point size: as panvk_varying_buf_id */
   const uint64_t ptr[3] = {gs_out[2], gs_out[0], gs_out[1]};
   const uint size[3] = {gen_B, pos_B, psiz_B};
   for (uint i = 0; i < 3; i++) {
      pan_unpack(&tiler_bufs[i], ATTRIBUTE_BUFFER, unpacked_buf)
         ;
      pan_pack(&tiler_bufs[i], ATTRIBUTE_BUFFER, cfg) {
         memcpy(&cfg, &unpacked_buf, sizeof(cfg));
         cfg.pointer = ptr[i];
         cfg.size = size[i];
      }
   }
}

/*
 * An indirect tessellated draw: the patch count comes from the command (vkCmdDrawIndirect or
 * vkCmdDrawIndexedIndirect, whose first two words are both the count and the instance count),
 * so everything sized by it is sized here, ahead of the tess control job.
 *
 * gp: the draw's pan_gs_params. tcs_job: the tess control COMPUTE job, tcs_local its workgroup
 * size. tess_jobs: the COUNT and WITH_COUNTS tessellator jobs.
 */
KERNEL(1)
panlib_tess_indirect_setup(global uint32_t *cmd, global struct pan_gs_params *gp,
                           global struct poly_tess_params *p, global uint8_t *tcs_job,
                           global uint64_t *tess_jobs, uint32_t patch_size,
                           uint32_t tcs_local, uint32_t tcs_stride_el)
{
   global struct poly_heap *heap = p->heap;
   const uint count = cmd[0];
   const uint instances = cmd[1];
   const uint ppi = count / patch_size;
   uint nr = ppi * instances;

   const ulong need = (ulong)nr * 8 + (ulong)nr * tcs_stride_el * 4 + 3 * 64;
   if (heap->bottom + need >= heap->size)
      nr = 0;

   p->nr_patches = nr;
   p->patches_per_instance = max(ppi, 1u);
   gp->prims_per_instance = ppi;
   gp->vertex_count = count;

   if (nr) {
      const uint64_t per_patch = panlib_tess_alloc(heap, nr * 8);
      p->coord_allocs = (global uint32_t *)per_patch;
      p->counts = (global uint32_t *)(per_patch + nr * 4);
      p->tcs_buffer = (global float *)panlib_tess_alloc(heap, nr * tcs_stride_el * 4);
   }

   panlib_tess_set_job_type(tcs_job, nr != 0, MALI_JOB_TYPE_COMPUTE);
   for (uint i = 0; i < 2; i++)
      panlib_tess_set_job_type((global uint8_t *)tess_jobs[i], nr != 0, MALI_JOB_TYPE_COMPUTE);
   if (!nr)
      return;

   pan_pack_work_groups_compute(
      (global struct mali_invocation_packed *)(tcs_job +
                                               pan_section_offset(COMPUTE_JOB, INVOCATION)),
      ppi, instances, 1, tcs_local, 1, 1, false, false);
   for (uint i = 0; i < 2; i++) {
      pan_pack_work_groups_compute(
         (global struct mali_invocation_packed *)((global uint8_t *)tess_jobs[i] +
                                                  pan_section_offset(COMPUTE_JOB, INVOCATION)),
         nr, 1, 1, 1, 1, 1, false, false);
   }
}

/*
 * A geometry shader after the tess eval shader: its input primitives are the tess eval
 * shader's output records (a list of vpp-vertex primitives over the tessellator's index count),
 * and its outputs are sized from them, both only known here. Runs after
 * panlib_tess_prefix_sum, which allocated the tess eval shader's outputs.
 *
 * tes_p, gs_p: the two stages' pan_gs_params. gs_job: the geometry shader's COMPUTE job,
 * tiler_jobs its TILER jobs, one per layer (layer_count of them, none with rasterizer discard),
 * tiler_bufs the tiler's varying buffers (general, position, point size). The rest describes
 * the geometry shader (pan_gs_info).
 */
KERNEL(1)
panlib_tess_gs_setup(global struct pan_gs_params *tes_p, global struct pan_gs_params *gs_p,
                     global uint32_t *count, global struct poly_heap *heap,
                     global uint8_t *gs_job, global uint64_t *tiler_jobs, uint32_t layer_count,
                     global struct mali_attribute_buffer_packed *tiler_bufs, uint32_t vpp,
                     uint32_t gs_invocations, uint32_t max_verts, uint32_t max_prims,
                     uint32_t output_verts, uint32_t out_stride, uint32_t writes_psiz)
{
   const uint prims = *count / vpp;
   const uint inv = prims * gs_invocations;
   const uint out_verts = inv * max_verts + 1;
   const uint index_count = inv * max_prims * output_verts;

   gs_p->in_pos = tes_p->out_pos;
   gs_p->in_psiz = tes_p->out_psiz;
   gs_p->in_general = tes_p->out_general;
   gs_p->num_input_prims = prims;
   gs_p->prims_per_instance = max(prims, 1u);

   const uint pos_B = out_verts * 16;
   const uint gen_B = out_verts * out_stride;
   const uint psiz_B = writes_psiz ? align(out_verts * 2, 4) : 0;
   const uint idx_B = index_count * 4 * max(layer_count, 1u);
   const ulong need = (ulong)pos_B + gen_B + psiz_B + idx_B + 5 * 64;
   const bool run = inv != 0 && heap->bottom + need < heap->size;

   panlib_tess_set_job_type(gs_job, run, MALI_JOB_TYPE_COMPUTE);
   for (uint l = 0; l < layer_count; l++)
      panlib_tess_set_job_type((global uint8_t *)tiler_jobs[l], run, MALI_JOB_TYPE_TILER);
   if (!run) {
      gs_p->num_input_prims = 0;
      return;
   }

   const uint64_t out_pos = panlib_tess_alloc(heap, pos_B);
   const uint64_t out_psiz = panlib_tess_alloc(heap, psiz_B);
   const uint64_t out_general = panlib_tess_alloc(heap, gen_B);
   const uint64_t out_index = panlib_tess_alloc(heap, idx_B);

   /* Every index slot the shader does not fill points at this vertex, far outside the
    * viewport. */
   global float *dead = (global float *)out_pos + (out_verts - 1) * 4;
   dead[0] = dead[1] = -1.0e6f;
   dead[2] = 0.0f;
   dead[3] = 1.0f;

   gs_p->out_pos = out_pos;
   gs_p->out_psiz = out_psiz;
   gs_p->out_general = out_general;
   gs_p->out_index = out_index;
   gs_p->dead_vertex = out_verts - 1;
   gs_p->layer_count = max(layer_count, 1u);
   gs_p->layer_stride = index_count;

   pan_pack_work_groups_compute(
      (global struct mali_invocation_packed *)(gs_job +
                                               pan_section_offset(COMPUTE_JOB, INVOCATION)),
      DIV_ROUND_UP(inv, PAN_GS_WORKGROUP_SIZE), 1, 1, PAN_GS_WORKGROUP_SIZE, 1, 1, false,
      false);

   if (layer_count == 0)
      return;

   /* Each layer's tiler job draws its own region of the index buffer. */
   for (uint l = 0; l < layer_count; l++) {
      global uint8_t *tiler_job = (global uint8_t *)tiler_jobs[l];

      pan_pack_work_groups_compute(
         (global struct mali_invocation_packed *)(tiler_job +
                                                  pan_section_offset(TILER_JOB, INVOCATION)),
         1, out_verts, 1, 1, 1, 1, true, false);

      global struct mali_primitive_packed *prim =
         (global struct mali_primitive_packed *)(tiler_job +
                                                 pan_section_offset(TILER_JOB, PRIMITIVE));
      pan_unpack(prim, PRIMITIVE, unpacked_prim)
         ;
      pan_pack(prim, PRIMITIVE, cfg) {
         memcpy(&cfg, &unpacked_prim, sizeof(cfg));
         cfg.index_count = index_count;
         cfg.indices = out_index + (uint64_t)l * index_count * 4;
      }

      global struct mali_draw_packed *dcd =
         (global struct mali_draw_packed *)(tiler_job + pan_section_offset(TILER_JOB, DRAW));
      pan_unpack(dcd, DRAW, unpacked_dcd)
         ;
      pan_pack(dcd, DRAW, cfg) {
         memcpy(&cfg, &unpacked_dcd, sizeof(cfg));
         cfg.position = out_pos;
      }

      if (unpacked_prim.point_size_array_format != MALI_POINT_SIZE_ARRAY_FORMAT_NONE) {
         pan_pack((global struct mali_primitive_size_packed *)(tiler_job +
                                                               pan_section_offset(
                                                                  TILER_JOB, PRIMITIVE_SIZE)),
                  PRIMITIVE_SIZE, cfg) {
            cfg.size_array = out_psiz;
         }
      }
   }

   /* general, position, point size: as panvk_varying_buf_id */
   const uint64_t ptr[3] = {out_general, out_pos, out_psiz};
   const uint size[3] = {gen_B, pos_B, psiz_B};
   for (uint i = 0; i < 3; i++) {
      pan_unpack(&tiler_bufs[i], ATTRIBUTE_BUFFER, unpacked_buf)
         ;
      pan_pack(&tiler_bufs[i], ATTRIBUTE_BUFFER, cfg) {
         memcpy(&cfg, &unpacked_buf, sizeof(cfg));
         cfg.pointer = ptr[i];
         cfg.size = size[i];
      }
   }
}

#endif
