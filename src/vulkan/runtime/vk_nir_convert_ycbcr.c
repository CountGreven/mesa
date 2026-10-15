/*
 * Copyright © 2017 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "vk_nir_convert_ycbcr.h"

#include "vk_format.h"
#include "vk_ycbcr_conversion.h"
#include "nir_builder.h"

#include "util/u_ycbcr.h"

#include <math.h>

static void
ycbcr_range_to_coeffs(float out[3][2], VkSamplerYcbcrRange range,
                      const unsigned bpc[3])
{
   switch (range) {
   case VK_SAMPLER_YCBCR_RANGE_ITU_FULL:
      util_get_full_range_coeffs(out, bpc);
      break;
   case VK_SAMPLER_YCBCR_RANGE_ITU_NARROW:
      util_get_narrow_range_coeffs(out, bpc);
      break;
   default:
      UNREACHABLE("missing Ycbcr range");
   }
}

static void
ycbcr_model_to_rgb_matrix(float out[3][4], VkSamplerYcbcrModelConversion model)
{
   switch (model) {
   case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601:
      util_get_ycbcr_to_rgb_matrix(out, util_ycbcr_bt601_coeffs);
      break;

   case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709:
      util_get_ycbcr_to_rgb_matrix(out, util_ycbcr_bt709_coeffs);
      break;

   case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020:
      util_get_ycbcr_to_rgb_matrix(out, util_ycbcr_bt2020_coeffs);
      break;

   default:
      UNREACHABLE("missing Ycbcr model");
   }
}

nir_def *
nir_convert_ycbcr_to_rgb(nir_builder *b,
                         VkSamplerYcbcrModelConversion model,
                         VkSamplerYcbcrRange range,
                         nir_def *raw_channels,
                         uint32_t *bpcs)
{
   const unsigned bpc[3] = { bpcs[1], bpcs[2], bpcs[0] };
   float range_coeffs[3][2];
   ycbcr_range_to_coeffs(range_coeffs, range, bpc);

   nir_def *expanded_channels =
      nir_vec4(b,
               nir_ffma_weak_imm12(b, nir_channel(b, raw_channels, 0), range_coeffs[2][0], range_coeffs[2][1]),
               nir_ffma_weak_imm12(b, nir_channel(b, raw_channels, 1), range_coeffs[0][0], range_coeffs[0][1]),
               nir_ffma_weak_imm12(b, nir_channel(b, raw_channels, 2), range_coeffs[1][0], range_coeffs[1][1]),
               nir_channel(b, raw_channels, 3));

   if (model == VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY)
      return expanded_channels;

   float m[3][4];
   ycbcr_model_to_rgb_matrix(m, model);

   nir_def *converted_channels[] = {
      nir_fdot(b, expanded_channels, nir_imm_vec4(b, m[0][2], m[0][0], m[0][1], 0.0f)),
      nir_fdot(b, expanded_channels, nir_imm_vec4(b, m[1][2], m[1][0], m[1][1], 0.0f)),
      nir_fdot(b, expanded_channels, nir_imm_vec4(b, m[2][2], m[2][0], m[2][1], 0.0f))
   };

   return nir_vec4(b,
                   converted_channels[0], converted_channels[1],
                   converted_channels[2], nir_channel(b, raw_channels, 3));
}

struct ycbcr_state {
   nir_builder *builder;
   nir_def *image_size;
   nir_tex_instr *origin_tex;
   nir_tex_src tex_handle;
   const struct vk_ycbcr_conversion_state *conversion;
   const struct vk_format_ycbcr_info *format_ycbcr_info;
   bool explicit_linear_chroma;
};

/* TODO: we should probably replace this with a push constant/uniform. */
static nir_def *
get_texture_size(struct ycbcr_state *state)
{
   if (state->image_size)
      return state->image_size;

   nir_builder *b = state->builder;
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 1);

   tex->op = nir_texop_txs;
   tex->sampler_dim = state->origin_tex->sampler_dim,
   tex->is_array = state->origin_tex->is_array,
   tex->is_shadow = state->origin_tex->is_shadow,
   tex->dest_type = nir_type_int32;

   tex->src[0] = state->tex_handle;

   nir_def_init(&tex->instr, &tex->def, nir_tex_instr_dest_size(tex), 32);
   nir_builder_instr_insert(b, &tex->instr);

   state->image_size = nir_i2f32(b, &tex->def);

   return state->image_size;
}

static nir_def *
implicit_downsampled_coord(nir_builder *b,
                           nir_def *value,
                           nir_def *max_value,
                           int div_scale)
{
   nir_def *scaled_max_value = nir_fmul_imm(b, max_value, div_scale);
   return nir_fadd(b, value, nir_frcp(b, scaled_max_value));
}

static nir_def *
implicit_downsampled_coords(struct ycbcr_state *state,
                            nir_def *old_coords,
                            const struct vk_format_ycbcr_plane *format_plane)
{
   nir_builder *b = state->builder;
   const struct vk_ycbcr_conversion_state *conversion = state->conversion;
   nir_def *image_size = get_texture_size(state);
   nir_def *comp[4] = { NULL, };
   int c;

   for (c = 0; c < ARRAY_SIZE(conversion->chroma_offsets); c++) {
      if (format_plane->denominator_scales[c] > 1 &&
          conversion->chroma_offsets[c] == VK_CHROMA_LOCATION_COSITED_EVEN) {
         comp[c] = implicit_downsampled_coord(b,
                                              nir_channel(b, old_coords, c),
                                              nir_channel(b, image_size, c),
                                              format_plane->denominator_scales[c]);
      } else {
         comp[c] = nir_channel(b, old_coords, c);
      }
   }

   /* Leave other coordinates untouched */
   for (; c < old_coords->num_components; c++)
      comp[c] = nir_channel(b, old_coords, c);

   return nir_vec(b, comp, old_coords->num_components);
}

static nir_def *
create_plane_tex_instr(struct ycbcr_state *state, uint32_t plane,
                       nir_def *coord_override)
{
   nir_builder *b = state->builder;
   const struct vk_ycbcr_conversion_state *conversion = state->conversion;
   const struct vk_format_ycbcr_plane *format_plane =
      &state->format_ycbcr_info->planes[plane];
   nir_tex_instr *old_tex = state->origin_tex;
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, old_tex->num_srcs + 1);

   for (uint32_t i = 0; i < old_tex->num_srcs; i++) {
      tex->src[i].src_type = old_tex->src[i].src_type;

      switch (old_tex->src[i].src_type) {
      case nir_tex_src_coord:
         if (coord_override) {
            tex->src[i].src = nir_src_for_ssa(coord_override);
            break;
         }
         if (format_plane->has_chroma && conversion->chroma_reconstruction) {
            tex->src[i].src =
               nir_src_for_ssa(implicit_downsampled_coords(state,
                                                           old_tex->src[i].src.ssa,
                                                           format_plane));
            break;
         }
         FALLTHROUGH;
      default:
         tex->src[i].src = nir_src_for_ssa(old_tex->src[i].src.ssa);
         break;
      }
   }
   tex->src[tex->num_srcs - 1] = nir_tex_src_for_ssa(nir_tex_src_plane,
                                                     nir_imm_int(b, plane));
   tex->sampler_dim = old_tex->sampler_dim;
   tex->dest_type = old_tex->dest_type;

   tex->op = old_tex->op;
   tex->coord_components = old_tex->coord_components;
   tex->is_new_style_shadow = old_tex->is_new_style_shadow;
   tex->component = old_tex->component;

   tex->embedded_sampler = old_tex->embedded_sampler;
   tex->texture_index = old_tex->texture_index;
   tex->sampler_index = old_tex->sampler_index;
   tex->is_array = old_tex->is_array;

   nir_def_init(&tex->instr, &tex->def, old_tex->def.num_components,
                old_tex->def.bit_size);
   nir_builder_instr_insert(b, &tex->instr);

   return &tex->def;
}

static nir_def *
create_plane_tex_instr_implicit(struct ycbcr_state *state, uint32_t plane)
{
   return create_plane_tex_instr(state, plane, NULL);
}

static bool
can_filter_explicitly(const nir_tex_instr *tex)
{
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_2D || tex->is_shadow)
      return false;

   if (tex->op != nir_texop_tex && tex->op != nir_texop_txl &&
       tex->op != nir_texop_txb)
      return false;

   return nir_tex_instr_src_index(tex, nir_tex_src_offset) < 0;
}

/* Bilinear filtering of a chroma plane in the shader: four samples at texel
 * centres (where the hardware weights are exactly 0 or 1) mixed with fp32
 * weights. YCbCr images have a single level and clamp to edge, so the sampler
 * handles the borders.
 */
static nir_def *
create_plane_tex_explicit_linear(struct ycbcr_state *state, uint32_t plane)
{
   nir_builder *b = state->builder;
   const struct vk_ycbcr_conversion_state *conversion = state->conversion;
   const struct vk_format_ycbcr_plane *format_plane =
      &state->format_ycbcr_info->planes[plane];
   nir_tex_instr *old_tex = state->origin_tex;
   nir_def *coord =
      old_tex->src[nir_tex_instr_src_index(old_tex, nir_tex_src_coord)].src.ssa;

   if (format_plane->has_chroma && conversion->chroma_reconstruction)
      coord = implicit_downsampled_coords(state, coord, format_plane);

   nir_def *image_size = get_texture_size(state);
   nir_def *centre[2][2], *frac[2];
   for (unsigned c = 0; c < 2; c++) {
      nir_def *size = nir_fmul_imm(b, nir_channel(b, image_size, c),
                                   1.0 / format_plane->denominator_scales[c]);
      nir_def *t = nir_fadd_imm(b, nir_fmul(b, nir_channel(b, coord, c), size),
                                -0.5);
      nir_def *i0 = nir_ffloor(b, t);
      frac[c] = nir_fsub(b, t, i0);
      centre[c][0] = nir_fdiv(b, nir_fadd_imm(b, i0, 0.5), size);
      centre[c][1] = nir_fdiv(b, nir_fadd_imm(b, i0, 1.5), size);
   }

   nir_def *texel[2][2];
   for (unsigned y = 0; y < 2; y++) {
      for (unsigned x = 0; x < 2; x++) {
         nir_def *comp[NIR_MAX_VEC_COMPONENTS];
         for (unsigned c = 0; c < coord->num_components; c++)
            comp[c] = nir_channel(b, coord, c);
         comp[0] = centre[0][x];
         comp[1] = centre[1][y];
         texel[y][x] = create_plane_tex_instr(
            state, plane, nir_vec(b, comp, coord->num_components));
      }
   }

   nir_def *row0 = nir_flrp(b, texel[0][0], texel[0][1], frac[0]);
   nir_def *row1 = nir_flrp(b, texel[1][0], texel[1][1], frac[0]);
   return nir_flrp(b, row0, row1, frac[1]);
}

static unsigned
swizzle_to_component(VkComponentSwizzle swizzle)
{
   switch (swizzle) {
   case VK_COMPONENT_SWIZZLE_R:
      return 0;
   case VK_COMPONENT_SWIZZLE_G:
      return 1;
   case VK_COMPONENT_SWIZZLE_B:
      return 2;
   case VK_COMPONENT_SWIZZLE_A:
      return 3;
   default:
      UNREACHABLE("invalid channel");
      return 0;
   }
}

struct lower_ycbcr_tex_state {
   nir_vk_ycbcr_conversion_lookup_cb cb;
   const void *cb_data;
   bool explicit_linear_chroma;
   bool (*explicit_linear_luma)(const void *data, uint32_t set,
                                uint32_t binding, uint32_t array_index);
};

static bool
lower_ycbcr_tex_instr(nir_builder *b, nir_tex_instr *tex, void *_state)
{
   const struct lower_ycbcr_tex_state *state = _state;

   /* For the following instructions, we don't apply any change and let the
    * instruction apply to the first plane.
    */
   if (tex->op == nir_texop_txs ||
       tex->op == nir_texop_query_levels ||
       tex->op == nir_texop_lod)
      return false;

   nir_tex_src tex_handle;
   const struct vk_ycbcr_conversion_state *conversion;
   uint32_t lookup_set, lookup_binding, lookup_index = 0;
   if (tex->embedded_sampler) {
      const int heap_src_idx =
         nir_tex_instr_src_index(tex, nir_tex_src_texture_heap_offset);
      tex_handle = tex->src[heap_src_idx];

      lookup_set = VK_NIR_YCBCR_SET_IMMUTABLE_SAMPLERS;
      lookup_binding = tex->sampler_index;
      conversion = state->cb(state->cb_data, lookup_set, lookup_binding, 0);
   } else {
      const int deref_src_idx =
         nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      if (deref_src_idx < 0)
         return false;

      tex_handle = tex->src[deref_src_idx];
      nir_deref_instr *deref = nir_src_as_deref(tex_handle.src);

      nir_variable *var = nir_deref_instr_get_variable(deref);
      uint32_t set = var->data.descriptor_set;
      uint32_t binding = var->data.binding;

      unsigned array_index = 0;
      if (deref->deref_type != nir_deref_type_var) {
         assert(deref->deref_type == nir_deref_type_array);
         if (!nir_src_is_const(deref->arr.index))
            return false;
         array_index = nir_src_as_uint(deref->arr.index);
      }

      lookup_set = set;
      lookup_binding = binding;
      lookup_index = array_index;
      conversion = state->cb(state->cb_data, set, binding, array_index);
   }
   if (conversion == NULL)
      return false;

   assert(tex->texture_index == 0);

   const struct vk_format_ycbcr_info *format_ycbcr_info =
      vk_format_get_ycbcr_info(conversion->format);

   /* This can happen if the driver hasn't done a good job of filtering on
    * sampler creation and lets through a VkYcbcrConversion object which isn't
    * actually YCbCr.  We're supposed to ignore those.
    */
   if (format_ycbcr_info == NULL)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   VkFormat y_format = VK_FORMAT_UNDEFINED;
   for (uint32_t p = 0; p < format_ycbcr_info->n_planes; p++) {
      if (!format_ycbcr_info->planes[p].has_chroma)
         y_format = format_ycbcr_info->planes[p].format;
   }
   assert(y_format != VK_FORMAT_UNDEFINED);
   uint8_t y_bpc = vk_format_get_bpc(y_format);

   /* |ycbcr_comp| holds components in the order : Cr-Y-Cb */
   nir_def *zero = nir_imm_float(b, 0.0f);
   nir_def *one = nir_imm_float(b, 1.0f);
   /* Use extra 2 channels for following swizzle */
   nir_def *ycbcr_comp[5] = { zero, zero, zero, one, zero };

   uint8_t ycbcr_bpcs[5];
   memset(ycbcr_bpcs, y_bpc, sizeof(ycbcr_bpcs));

   /* Go through all the planes and gather the samples into a |ycbcr_comp|
    * while applying a swizzle required by the spec:
    *
    *    R, G, B should respectively map to Cr, Y, Cb
    */
   for (uint32_t p = 0; p < format_ycbcr_info->n_planes; p++) {
      const struct vk_format_ycbcr_plane *format_plane =
         &format_ycbcr_info->planes[p];

      struct ycbcr_state tex_state = {
         .builder = b,
         .origin_tex = tex,
         .tex_handle = tex_handle,
         .conversion = conversion,
         .format_ycbcr_info = format_ycbcr_info,
      };
      const bool explicit_linear =
         can_filter_explicitly(tex) &&
         (format_plane->has_chroma
             ? state->explicit_linear_chroma &&
                  conversion->chroma_filter == VK_FILTER_LINEAR
             : state->explicit_linear_luma &&
                  state->explicit_linear_luma(state->cb_data, lookup_set,
                                              lookup_binding, lookup_index));
      nir_def *plane_sample =
         explicit_linear ? create_plane_tex_explicit_linear(&tex_state, p)
                         : create_plane_tex_instr_implicit(&tex_state, p);

      for (uint32_t pc = 0; pc < 4; pc++) {
         VkComponentSwizzle ycbcr_swizzle = format_plane->ycbcr_swizzle[pc];
         if (ycbcr_swizzle == VK_COMPONENT_SWIZZLE_ZERO)
            continue;

         unsigned ycbcr_component = swizzle_to_component(ycbcr_swizzle);
         ycbcr_comp[ycbcr_component] = nir_channel(b, plane_sample, pc);

         /* Also compute the number of bits for each component. */
         const struct util_format_description *plane_format_desc =
            vk_format_description(format_plane->format);
         if (plane_format_desc->channel[pc].type == UTIL_FORMAT_TYPE_VOID)
            continue;
         ycbcr_bpcs[ycbcr_component] = plane_format_desc->channel[pc].size;
      }
   }

   /* Now remaps components to the order specified by the conversion. */
   nir_def *swizzled_comp[4] = { NULL, };
   uint32_t swizzled_bpcs[4] = { 0, };

   for (uint32_t i = 0; i < ARRAY_SIZE(conversion->mapping); i++) {
      /* Maps to components in |ycbcr_comp| */
      static const uint32_t swizzle_mapping[] = {
         [VK_COMPONENT_SWIZZLE_ZERO] = 4,
         [VK_COMPONENT_SWIZZLE_ONE]  = 3,
         [VK_COMPONENT_SWIZZLE_R]    = 0,
         [VK_COMPONENT_SWIZZLE_G]    = 1,
         [VK_COMPONENT_SWIZZLE_B]    = 2,
         [VK_COMPONENT_SWIZZLE_A]    = 3,
      };
      const VkComponentSwizzle m = conversion->mapping[i];

      if (m == VK_COMPONENT_SWIZZLE_IDENTITY) {
         swizzled_comp[i] = ycbcr_comp[i];
         swizzled_bpcs[i] = ycbcr_bpcs[i];
      } else {
         swizzled_comp[i] = ycbcr_comp[swizzle_mapping[m]];
         swizzled_bpcs[i] = ycbcr_bpcs[swizzle_mapping[m]];
      }
   }

   nir_def *result = nir_vec(b, swizzled_comp, 4);
   if (conversion->ycbcr_model != VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY) {
      result = nir_convert_ycbcr_to_rgb(b, conversion->ycbcr_model,
                                           conversion->ycbcr_range,
                                           result,
                                           swizzled_bpcs);
   }

   nir_def_replace(&tex->def, result);

   return true;
}

bool nir_vk_lower_ycbcr_tex(nir_shader *nir,
                            nir_vk_ycbcr_conversion_lookup_cb cb,
                            const void *cb_data)
{
   return nir_vk_lower_ycbcr_tex_opts(nir, cb, cb_data, NULL);
}

bool nir_vk_lower_ycbcr_tex_opts(nir_shader *nir,
                                 nir_vk_ycbcr_conversion_lookup_cb cb,
                                 const void *cb_data,
                                 const struct nir_vk_lower_ycbcr_tex_options *opts)
{
   struct lower_ycbcr_tex_state state = {
      .cb = cb,
      .cb_data = cb_data,
      .explicit_linear_chroma = opts && opts->explicit_linear_chroma,
      .explicit_linear_luma = opts ? opts->explicit_linear_luma : NULL,
   };

   return nir_shader_tex_pass(nir, lower_ycbcr_tex_instr,
                              nir_metadata_control_flow, &state);
}
