#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include "THC/THC.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <math.h>
#include <vector>
#include <stdio.h>
#include "utils.cu"


__device__
void random_project_step(
        int num_threads_per_out_dim,
        __half2 x_val[HALF2_PER_THREAD], 
        __half2 w_val[HALF2_PER_THREAD],
        __half2 *phi_x_local) {
    __half2 wx = __float2half2_rn(0.f);
    __half2 x_norm = __float2half2_rn(0.f);

    #pragma unroll 
    for (int i = 0;i < HALF2_PER_THREAD; ++ i) {
        wx = __hfma2(w_val[i], x_val[i], wx);
        x_norm = __hfma2(x_val[i], x_val[i], x_norm);
    }
    #pragma unroll 
    for (int offset = num_threads_per_out_dim >> 1;
         offset > 0; 
         offset >>= 1) { 
        wx =  __hadd2(wx, __shfl_down_sync(FULL_MASK, wx, offset));
        x_norm =  __hadd2(x_norm, __shfl_down_sync(FULL_MASK, x_norm, offset));
    }
    if (threadIdx.x == 0) {
        __half wx_half = __hadd(wx.x, wx.y);
        __half x_norm_half = __hadd(x_norm.x, x_norm.y);
        x_norm_half = hsqrt(clamp(x_norm_half));
        wx_half = __hdiv(wx_half, x_norm_half);
        __half sin_wx = hsin(wx_half);
        __half cos_wx = hcos(wx_half);
        __half2 phi_x = __halves2half2(sin_wx, cos_wx);
        *phi_x_local = phi_x;
    }
}


__device__
void random_project_step(
        int num_threads_per_out_dim,
        __half2 x_val[HALF2_PER_THREAD], 
        __half2 y_val[HALF2_PER_THREAD], 
        __half2 w_val[HALF2_PER_THREAD],
        __half2 *phi_x_local,
        __half2 *phi_y_local) {
    __half2 wx = __float2half2_rn(0.f);
    __half2 wy = __float2half2_rn(0.f);
    __half2 x_norm = __float2half2_rn(0.f);
    __half2 y_norm = __float2half2_rn(0.f);

    #pragma unroll 
    for (int i = 0;i < HALF2_PER_THREAD; ++ i) {
        wx = __hfma2(w_val[i], x_val[i], wx);
        wy = __hfma2(w_val[i], y_val[i], wy);
        x_norm = __hfma2(x_val[i], x_val[i], x_norm);
        y_norm = __hfma2(y_val[i], y_val[i], y_norm);
    }
    #pragma unroll 
    for (int offset = num_threads_per_out_dim >> 1;
         offset > 0; 
         offset >>= 1) {
        wx =  __hadd2(wx, __shfl_down_sync(FULL_MASK, wx, offset));
        wy =  __hadd2(wy, __shfl_down_sync(FULL_MASK, wy, offset));
        x_norm =  __hadd2(x_norm, __shfl_down_sync(FULL_MASK, x_norm, offset));
        y_norm =  __hadd2(y_norm, __shfl_down_sync(FULL_MASK, y_norm, offset));
    }
    if (threadIdx.x == 0) {
        __half2 wxy = __halves2half2(
            __hadd(wx.x, wx.y),
            __hadd(wy.x, wy.y)
        );
        __half2 xy_norm = __halves2half2(
            __hadd(x_norm.x, x_norm.y),
            __hadd(y_norm.x, y_norm.y)
        );
        xy_norm = h2sqrt(xy_norm);
        wxy = __h2div(wxy, clamp(xy_norm));
        __half2 sin = h2sin(wxy);
        __half2 cos = h2cos(wxy);
        __half2 phi_x = __halves2half2(sin.x, cos.x);
        __half2 phi_y = __halves2half2(sin.y, cos.y);
        *phi_x_local = phi_x;
        *phi_y_local = phi_y;
    }
}


__device__
void read(
    const __half * __restrict__ x_local,
    const __half * __restrict__ w_local,
    __half2 x_val[HALF2_PER_THREAD], 
    __half2 w_val[HALF2_PER_THREAD]) {

    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) x_val + j) = *((int4*) x_local + j);
    }
    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) w_val + j) = *((int4*) w_local + j);
    }
}


__device__
void read(
    const __half * __restrict__ x_local,
    const __half * __restrict__ y_local,
    const __half * __restrict__ w_local,
    __half2 x_val[HALF2_PER_THREAD], 
    __half2 y_val[HALF2_PER_THREAD], 
    __half2 w_val[HALF2_PER_THREAD]) {

    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) x_val + j) = *((int4*) x_local + j);
        *((int4 *) y_val + j) = *((int4*) y_local + j);
    }
    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) w_val + j) = *((int4*) w_local + j);
    }
}


__global__ 
void random_project(
        const __half * __restrict__ x,
        const __half * __restrict__ w,
        __half2 * __restrict__ phi_x,
        int num_heads,
        int in_dim, 
        int out_dim,
        int num_threads_per_out_dim,
        int num_out_dim_per_block,
        int num_blocks_per_head,
        int num_blocks_per_batch) {
    /*
    Args:
        x: [1, bsz, num_heads, in_dim]
        w: [num_heads, out_dim, in_dim]
        b: [num_heads, out_dim]
        
    Return:
        phi_x: [1, bsz, num_heads, out_dim]
    */

    const int batch_id = blockIdx.x / num_blocks_per_batch;
    const int head_id = (blockIdx.x % num_blocks_per_batch) / num_blocks_per_head;

    const int in_dim_offset = threadIdx.x * DIM_PER_THREAD;
    const int out_dim_id \
        = (blockIdx.x % num_blocks_per_head) * num_out_dim_per_block + threadIdx.y;
    
    const __half * __restrict__ x_local \
        = x + batch_id * num_heads * in_dim \
        + head_id * in_dim + in_dim_offset;
    const __half * __restrict__ w_local \
        = w + head_id * out_dim * in_dim \
        + out_dim_id * in_dim + in_dim_offset;

    __half2 * __restrict__ phi_x_local \
        = phi_x + batch_id * num_heads * out_dim \
        + head_id * out_dim + out_dim_id;
    
    __half2 x_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 w_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    read(x_local, w_local, x_val, w_val);
    random_project_step(
        num_threads_per_out_dim, 
        x_val, w_val,
        phi_x_local
    );
}


__global__ 
void random_project(
        const __half * __restrict__ x,
        const __half * __restrict__ y,
        const __half * __restrict__ w,
        __half2 * __restrict__ phi_x,
        __half2 * __restrict__ phi_y,
        int num_heads,
        int in_dim, 
        int out_dim,
        int num_threads_per_out_dim,
        int num_out_dim_per_block,
        int num_blocks_per_head,
        int num_blocks_per_batch) {
    /*
    Args:
        x: [1, bsz, num_heads, in_dim]
        w: [num_heads, out_dim, in_dim]
        b: [num_heads, out_dim]
        
    Return:
        phi_x: [1, bsz, num_heads, out_dim]
    */

    const int batch_id = blockIdx.x / num_blocks_per_batch;
    const int head_id = (blockIdx.x % num_blocks_per_batch) / num_blocks_per_head;

    const int in_dim_offset = threadIdx.x * DIM_PER_THREAD;
    const int out_dim_id \
        = (blockIdx.x % num_blocks_per_head) * num_out_dim_per_block + threadIdx.y;
    
    int offset = batch_id * num_heads * in_dim \
        + head_id * in_dim + in_dim_offset;
    const __half * __restrict__ x_local = x + offset;
    const __half * __restrict__ y_local = y + offset;
    const __half * __restrict__ w_local \
        = w + head_id * out_dim * in_dim \
        + out_dim_id * in_dim + in_dim_offset;
    
    offset = batch_id * num_heads * out_dim \
        + head_id * out_dim + out_dim_id;
    __half2 * __restrict__ phi_x_local = phi_x + offset;
    __half2 * __restrict__ phi_y_local = phi_y + offset;
    
    __half2 x_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 y_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 w_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    read(x_local, y_local, w_local, x_val, y_val, w_val);
    random_project_step(
        num_threads_per_out_dim, 
        x_val, y_val, w_val,
        phi_x_local, phi_y_local
    );
}


Tensor RandomProject(
        Tensor const& x,
        Tensor const& w) {
    /*
    Args:
        x: [1, bsz, num_heads, in_dim]
        w: [num_heads, out_dim, in_dim]
        
    Return:
        phi_x: [1, bsz, num_heads, out_dim]
    */
    // column major
    const int bsz = x.size(1);
    const int num_heads = x.size(2);
    const int in_dim = x.size(3);
    const int out_dim = w.size(1);

    auto act_options  = x.options().requires_grad(false);
    Tensor phi_x = torch::zeros({1, bsz, num_heads, out_dim * 2}, act_options);

    // num threads per head_dim;
    const int num_threads_per_out_dim = in_dim / DIM_PER_THREAD;
    const int num_out_dim_per_block = min(
        out_dim, NUM_THREADS_PER_BLOCK / num_threads_per_out_dim); 
    const int num_blocks_per_head = max(1, out_dim / num_out_dim_per_block);
    const int num_blocks_per_batch = num_heads * num_blocks_per_head;
    dim3 dim_grid(bsz * num_blocks_per_batch);
    // [x, y]
    dim3 dim_block(num_threads_per_out_dim, num_out_dim_per_block);
    random_project <<<dim_grid, dim_block>>>(
            static_cast<const __half *> (x.data_ptr()),
            static_cast<const __half *> (w.data_ptr()), 
            static_cast<__half2 *> (phi_x.data_ptr()), 
            num_heads,
            in_dim, 
            out_dim,
            num_threads_per_out_dim,
            num_out_dim_per_block,
            num_blocks_per_head,
            num_blocks_per_batch
    );
    return phi_x;
}


std::vector<Tensor> RandomProject(
    Tensor const& x,
    Tensor const& y,
    Tensor const& w) {
    /*
    Args:
        x: [1, bsz, num_heads, in_dim]
        y: [1, bsz, num_heads, in_dim]
        w: [num_heads, out_dim, in_dim]
        
    Return:
        phi_x: [1, bsz, num_heads, out_dim]
    */
    // column major
    const int bsz = x.size(1);
    const int num_heads = x.size(2);
    const int in_dim = x.size(3);
    const int out_dim = w.size(1);

    auto act_options  = x.options().requires_grad(false);
    Tensor phi_x = torch::zeros({1, bsz, num_heads, out_dim * 2}, act_options);
    Tensor phi_y = torch::zeros({1, bsz, num_heads, out_dim * 2}, act_options);

    // num threads per head_dim;
    const int num_threads_per_out_dim = in_dim / DIM_PER_THREAD;
    const int num_out_dim_per_block = min(
        out_dim, NUM_THREADS_PER_BLOCK / num_threads_per_out_dim); 
    const int num_blocks_per_head = max(1, out_dim / num_out_dim_per_block);
    const int num_blocks_per_batch = num_heads * num_blocks_per_head;
    dim3 dim_grid(bsz * num_blocks_per_batch);
    // [x, y]
    dim3 dim_block(num_threads_per_out_dim, num_out_dim_per_block);
    random_project <<<dim_grid, dim_block>>>(
            static_cast<const __half *> (x.data_ptr()),
            static_cast<const __half *> (y.data_ptr()),
            static_cast<const __half *> (w.data_ptr()), 
            static_cast<__half2 *> (phi_x.data_ptr()), 
            static_cast<__half2 *> (phi_y.data_ptr()), 
            num_heads,
            in_dim, 
            out_dim,
            num_threads_per_out_dim,
            num_out_dim_per_block,
            num_blocks_per_head,
            num_blocks_per_batch
    );
    return {phi_x, phi_y};
}