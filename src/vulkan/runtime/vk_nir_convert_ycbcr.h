/*
 * Copyright © 2020 Jonathan Marek
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

#ifndef VK_NIR_CONVERT_YCBCR_H
#define VK_NIR_CONVERT_YCBCR_H

#include "nir_defines.h"
#include "vulkan/vulkan_core.h"

#ifdef __cplusplus
extern "C" {
#endif

nir_def *
nir_convert_ycbcr_to_rgb(nir_builder *b,
                         VkSamplerYcbcrModelConversion model,
                         VkSamplerYcbcrRange range,
                         nir_def *raw_channels,
                         uint32_t *bpcs);

struct vk_ycbcr_conversion;

/** Passed as the set parameter to nir_vk_ycbcr_conversion_lookup_cb() to
 * indicate that embedded samplers are being used and that binding is the
 * index in the embedded sampler table.
 */
#define VK_NIR_YCBCR_SET_IMMUTABLE_SAMPLERS UINT32_MAX

typedef const struct vk_ycbcr_conversion_state *
   (*nir_vk_ycbcr_conversion_lookup_cb)(const void *data, uint32_t set,
                                        uint32_t binding, uint32_t array_index);

bool nir_vk_lower_ycbcr_tex(nir_shader *nir,
                            nir_vk_ycbcr_conversion_lookup_cb cb,
                            const void *cb_data);

struct nir_vk_lower_ycbcr_tex_options {
   /* Filter LINEAR chroma planes in the shader (four texel-centre samples
    * and fp32 weights) instead of relying on the hardware bilinear weights,
    * for hardware whose weight quantization is not the separable one the
    * spec's subTexelPrecisionBits describes.
    */
   bool explicit_linear_chroma;

   /* Optional: also filter the luma plane in the shader when this returns
    * true for the sampler (both min and mag filters LINEAR). Same arguments
    * as the conversion lookup.
    */
   bool (*explicit_linear_luma)(const void *data, uint32_t set,
                                uint32_t binding, uint32_t array_index);
};

bool nir_vk_lower_ycbcr_tex_opts(nir_shader *nir,
                                 nir_vk_ycbcr_conversion_lookup_cb cb,
                                 const void *cb_data,
                                 const struct nir_vk_lower_ycbcr_tex_options *opts);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VK_NIR_CONVERT_YCBCR_H */
