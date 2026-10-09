
        uint row = threadgroup_position_in_grid.y;                  // assignment
        uint tn = threadgroup_position_in_grid.x;                   // output tile column (16 outputs)
        uint lane = thread_index_in_simdgroup;
        threadgroup float red[16][32];
        uint sid = ids[row]; uint bk = sid >> 24u;
        const device short* base = (bk == 0u ? code0 : (bk == 1u ? code1 : code2)) + ulong(sid & 0xFFFFFFu) * ulong((bk == 0u ? code0_strides[0] : (bk == 1u ? code1_strides[0] : code2_strides[0])));
        const device float* xp = xh + row * 2304u;
        float acc[8];
        for (uint m = 0; m < 8u; ++m) acc[m] = 0.0f;
        uint dr[8]; uint dc[8];
        for (uint m = 0; m < 8u; ++m) { uint q = m & 3u; dr[m] = 2u * (lane & 3u) + (q < 2u ? 9u : 3u) - q; dc[m] = (lane >> 2) + (m < 4u ? 8u : 0u); }
        // K3 keeps the two-word funnel extraction; K2 / K4 / K5 gather each window with rate_window<K>
        gemv_rate_loop<K, 144u, 320u, 4u>(base, xp, tn, lane, dr, acc);
        for (uint c = 0; c < 16u; ++c) red[c][lane] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint m = 0; m < 8u; ++m) red[dc[m]][lane] += acc[m];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 16u) {
            float s = 0.0f;
            for (uint l = 0; l < 32u; ++l) s += red[lane][l];
            out[row * 5120u + tn * 16u + lane] = s;
        }
    