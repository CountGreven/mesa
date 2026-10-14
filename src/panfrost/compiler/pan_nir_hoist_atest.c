/*
 * Copyright (C) 2026 Fredrik Lindlöf
 * SPDX-License-Identifier: MIT
 */

#include "pan_nir.h"

/*
 * ATEST makes the fragment's coverage final. When the depth/stencil update
 * has to happen late (the shader can discard), later quads on the same pixels
 * wait for it: on a G72, Minecraft's alpha-tested terrain pass spends most of
 * its time with the shader cores idle (31% core utilisation, fragment queue
 * busy 82%), and stripping the discards cuts the pass from 84 ms to < 30 ms.
 *
 * pan_nir_lower_fs_outputs emits ATEST next to the colour store at the end of
 * the shader, so everything computed after the last discard (fog, lighting)
 * delays the update. Move ATEST, with the coverage it reads, right after the
 * last discard and the alpha value it needs.
 *
 * ATEST also ends the shader's side effects, so shaders that write memory are
 * left alone, as are depth/stencil writes (ZS_EMIT must precede ATEST).
 */
bool
pan_nir_hoist_atest(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_FRAGMENT || nir->info.writes_memory)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_block *block = nir_impl_last_block(impl);
   nir_intrinsic_instr *atest = NULL;

   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_zs_emit_pan)
         return false;
      if (intr->intrinsic == nir_intrinsic_atest_pan) {
         atest = intr;
         break;
      }
   }

   if (!atest)
      return false;

   nir_instr *cov = nir_def_instr(atest->src[0].ssa);
   if (cov->type != nir_instr_type_intrinsic || cov->block != block ||
       nir_instr_as_intrinsic(cov)->intrinsic !=
          nir_intrinsic_load_cumulative_coverage_pan)
      return false;

   /* The coverage must include every discard, and the alpha must exist. */
   nir_instr *alpha = nir_def_instr(atest->src[1].ssa);
   nir_instr *after = NULL;

   nir_foreach_instr(instr, block) {
      if (instr == &atest->instr)
         break;
      if (instr == cov)
         continue;

      if (instr == alpha)
         after = instr;

      if (instr->type == nir_instr_type_intrinsic) {
         switch (nir_instr_as_intrinsic(instr)->intrinsic) {
         case nir_intrinsic_demote:
         case nir_intrinsic_demote_if:
         case nir_intrinsic_terminate:
         case nir_intrinsic_terminate_if:
            after = instr;
            break;
         default:
            break;
         }
      }
   }

   nir_cursor cursor = after ? nir_after_instr(after) : nir_after_phis(block);

   /* Already there. */
   if (nir_cursors_equal(cursor, nir_before_instr(cov)))
      return false;

   nir_instr_remove(&atest->instr);
   nir_instr_remove(cov);
   nir_instr_insert(cursor, cov);
   nir_instr_insert(nir_after_instr(cov), &atest->instr);

   return nir_progress(true, impl, nir_metadata_control_flow);
}
