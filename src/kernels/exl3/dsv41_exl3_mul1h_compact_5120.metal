
        // SX_NSG simdgroups per threadgroup, one 16-output tile column each; the row's activations staged in xs
        uint row = threadgroup_position_in_grid.y;                  // assignment
        uint sg = simdgroup_index_in_threadgroup;
        uint tn = threadgroup_position_in_grid.x * SX_NSG + sg;     // output tile column (16 outputs)
        uint lane = thread_index_in_simdgroup;
        uint tid = thread_index_in_threadgroup;
        // the row's activations (float4 storage: 16-byte aligned stores), then the simdgroups' [16][32] partials
        threadgroup float4 xs4[(576u > 128u * SX_NSG) ? 576u : 128u * SX_NSG];
        threadgroup float* xs = reinterpret_cast<threadgroup float*>(xs4);
        const ulong di = ulong(ids[row]) * 2ul;
        const uint K = uint(desc[di + 1ul]);
        const device short* base = code + desc[di];
        sx_stage<2304u, 32u * SX_NSG>(xh + row * 2304u, xs4, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float acc[8];
        for (uint m = 0; m < 8u; ++m) acc[m] = 0.0f;
        switch (K) {                                                // uniform per threadgroup: one tile loop per rate
            case 2u: sx_tiles<2, 144u, 320u, 4>(base, tn, xs, lane, acc); break;
            case 3u: sx_tiles<3, 144u, 320u, 4>(base, tn, xs, lane, acc); break;
            case 4u: sx_tiles<4, 144u, 320u, 4>(base, tn, xs, lane, acc); break;
            default: sx_tiles<5, 144u, 320u, 4>(base, tn, xs, lane, acc); break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);            // every simdgroup is done with xs: reuse it
        sx_reduce(xs + sg * 512u, acc, lane, out + row * 5120u + tn * 16u);
    