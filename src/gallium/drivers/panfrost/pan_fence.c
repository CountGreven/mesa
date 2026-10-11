/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates.
 * Copyright (C) 2008 VMware, Inc.
 * Copyright (C) 2014 Broadcom
 * Copyright (C) 2018 Alyssa Rosenzweig
 * Copyright (C) 2019 Collabora, Ltd.
 * Copyright (C) 2012 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 */

#include "pan_fence.h"
#include "pan_context.h"
#include "pan_job.h"
#include "pan_screen.h"

#include "util/os_time.h"
#include "util/u_inlines.h"
#include "kmod/pan_kmod.h"

void
panfrost_fence_reference(struct pipe_screen *pscreen,
                         struct pipe_fence_handle **ptr,
                         struct pipe_fence_handle *fence)
{
   struct panfrost_device *dev = pan_device(pscreen);
   struct pipe_fence_handle *old = *ptr;

   if (pipe_reference(&old->reference, &fence->reference)) {
      pan_kmod_sync_destroy(dev->kmod.dev, old->syncobj);
      free(old);
   }

   *ptr = fence;
}

/*
 * Flush the deferred batches of the fence's own context, then fill the
 * fence's empty syncobj with the context's now-updated syncobj so waiters
 * block on the real GPU work.
 */
static void
panfrost_fence_flush_deferred(struct pipe_screen *pscreen,
                              struct pipe_fence_handle *f)
{
   struct panfrost_context *ctx = f->unflushed_ctx;

   /* This fills the fence's syncobj via the context's pending-fence list. */
   panfrost_flush_all_batches(ctx, "Deferred fence flush");
   f->unflushed_ctx = NULL;
}

/*
 * Fill every deferred fence this context is holding pending, then clear the
 * list. Called from panfrost_flush_all_batches after the context's work has
 * actually been submitted, so ctx->syncobj now tracks that work.
 */
void
panfrost_fill_deferred_fences(struct panfrost_context *ctx)
{
   struct panfrost_device *dev = pan_device(ctx->base.screen);
   unsigned count = util_dynarray_num_elements(&ctx->pending_deferred_fences,
                                               struct pipe_fence_handle *);

   if (!count)
      return;

   int fd = -1, ret;

   ret = pan_kmod_sync_export_file(dev->kmod.dev, ctx->syncobj, &fd);
   if (ret || fd == -1) {
      mesa_loge("export failed\n");
      return;
   }

   util_dynarray_foreach(&ctx->pending_deferred_fences,
                         struct pipe_fence_handle *, fp) {
      struct pipe_fence_handle *f = *fp;

      ret = pan_kmod_sync_import_file(dev->kmod.dev, f->syncobj, fd);
      if (ret)
         mesa_loge("import syncfile failed\n");

      f->unflushed_ctx = NULL;

      /* Drop the reference the pending list was holding. */
      struct pipe_fence_handle *tmp = f;
      panfrost_fence_reference(ctx->base.screen, &tmp, NULL);
   }

   close(fd);
   util_dynarray_init(&ctx->pending_deferred_fences, ctx);
}

bool
panfrost_fence_finish(struct pipe_screen *pscreen, struct pipe_context *ctx,
                      struct pipe_fence_handle *fence, uint64_t timeout)
{
   struct panfrost_device *dev = pan_device(pscreen);
   unsigned flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;
   int ret;

   if (fence->signaled)
      return true;

   /* Only flush when the caller's context matches the one the fence was
    * created with. Poking at another context's internals isn't safe because
    * it may be bound to a different thread.
    */
   if (ctx && ctx == (struct pipe_context *)fence->unflushed_ctx)
      panfrost_fence_flush_deferred(pscreen, fence);

   if (fence->unflushed_ctx) {
      /* The fence is still unflushed and belongs to another (or no) context.
       * We can't safely flush it here. A zero timeout can't wait for work
       * that hasn't been submitted, so report it as not signaled. Otherwise
       * use WAIT_FOR_SUBMIT to block until the other thread submits the
       * work, mirroring iris_fence_finish.
       */
      if (timeout == 0)
         return false;

      flags |= DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;
   }

   uint64_t abs_timeout = os_time_get_absolute_timeout(timeout);
   if (abs_timeout == OS_TIMEOUT_INFINITE)
      abs_timeout = INT64_MAX;

   ret = pan_kmod_sync_wait(dev->kmod.dev, &fence->syncobj, 1, abs_timeout,
                            flags, NULL);

   fence->signaled = (ret >= 0);
   return fence->signaled;
}

int
panfrost_fence_get_fd(struct pipe_screen *screen, struct pipe_fence_handle *f)
{
   struct panfrost_device *dev = pan_device(screen);
   int fd = -1;

   /* A deferred fence can't be exported until its batches have been
    * submitted, and get_fd has no context to check against. Flush its own
    * context, which fills the fence's syncobj via the pending-fence list.
    */
   if (f->unflushed_ctx)
      panfrost_fence_flush_deferred(screen, f);

   pan_kmod_sync_export_file(dev->kmod.dev, f->syncobj, &fd);
   return fd;
}

struct pipe_fence_handle *
panfrost_fence_from_fd(struct panfrost_context *ctx, int fd,
                       enum pipe_fd_type type)
{
   struct panfrost_device *dev = pan_device(ctx->base.screen);
   int ret;

   struct pipe_fence_handle *f = calloc(1, sizeof(*f));
   if (!f)
      return NULL;

   if (type == PIPE_FD_TYPE_NATIVE_SYNC) {
      ret = pan_kmod_sync_create(dev->kmod.dev, 0, &f->syncobj);
      if (ret) {
         mesa_loge("create syncobj failed\n");
         goto err_free_fence;
      }

      ret = pan_kmod_sync_import_file(dev->kmod.dev, f->syncobj, fd);
      if (ret) {
         mesa_loge("import syncfile failed\n");
         goto err_destroy_syncobj;
      }
   } else {
      assert(type == PIPE_FD_TYPE_SYNCOBJ);
      ret = pan_kmod_sync_fd_to_handle(dev->kmod.dev, fd, &f->syncobj);
      if (ret) {
         mesa_loge("import syncobj FD failed\n");
         goto err_free_fence;
      }
   }

   pipe_reference_init(&f->reference, 1);

   return f;

err_destroy_syncobj:
   pan_kmod_sync_destroy(dev->kmod.dev, f->syncobj);
err_free_fence:
   free(f);
   return NULL;
}

struct pipe_fence_handle *
panfrost_fence_create(struct panfrost_context *ctx)
{
   struct panfrost_device *dev = pan_device(ctx->base.screen);
   int fd = -1, ret;

   /* Snapshot the last rendering out fence. We'd rather have another
    * syncobj instead of a sync file, but this is all we get.
    * (HandleToFD/FDToHandle just gives you another syncobj ID for the
    * same syncobj).
    */
   ret = pan_kmod_sync_export_file(dev->kmod.dev, ctx->syncobj, &fd);
   if (ret || fd == -1) {
      mesa_loge("export failed\n");
      return NULL;
   }

   struct pipe_fence_handle *f =
      panfrost_fence_from_fd(ctx, fd, PIPE_FD_TYPE_NATIVE_SYNC);

   close(fd);

   return f;
}

/*
 * Create a deferred fence: a fence with a NEW EMPTY syncobj. The syncobj
 * will be filled with ctx->syncobj when the context's work is submitted
 * (via panfrost_flush_all_batches). This is the iris-style approach that
 * makes WAIT_FOR_SUBMIT work correctly for cross-context waiters.
 */
struct pipe_fence_handle *
panfrost_fence_create_deferred(struct panfrost_context *ctx)
{
   struct panfrost_device *dev = pan_device(ctx->base.screen);

   struct pipe_fence_handle *f = calloc(1, sizeof(*f));
   if (!f)
      return NULL;

   int ret = pan_kmod_sync_create(dev->kmod.dev, 0, &f->syncobj);
   if (ret) {
      mesa_loge("create syncobj failed\n");
      free(f);
      return NULL;
   }

   pipe_reference_init(&f->reference, 1);
   f->unflushed_ctx = ctx;

   return f;
}
