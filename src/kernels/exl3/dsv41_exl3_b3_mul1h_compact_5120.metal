
        uint row = threadgroup_position_in_grid.y;                  // assignment
        uint tn = threadgroup_position_in_grid.x;                   // output tile column (16 outputs)
        uint lane = thread_index_in_simdgroup;
        threadgroup float red[16][32];
        const uint sid = ids[row];
        if (sid == 0xffffffffu) {
            if (lane < 16u) out[row * 5120u + tn * 16u + lane] = 0.0f;
            return;
        }
        const uint bk = sid >> 24u;
        const ulong di = ulong(sid & 0xFFFFFFu) * 2ul;
        const ulong offset = bk == 0u ? desc0[di] : (bk == 1u ? desc1[di] : desc2[di]);
        const uint K = uint(bk == 0u ? desc0[di + 1ul] : (bk == 1u ? desc1[di + 1ul] : desc2[di + 1ul]));
        const device short* base = (bk == 0u ? code0 : (bk == 1u ? code1 : code2)) + offset;
        const device float* xp = xh + row * 2304u;
        uint wa = 2u * ((3u * lane + 2u) >> 2); uint wb = (wa + 46u) % 48u; uint lp = 8u * ((lane + 1u) & 3u);
        float acc[8];
        for (uint m = 0; m < 8u; ++m) acc[m] = 0.0f;
        uint dr[8]; uint dc[8];
        for (uint m = 0; m < 8u; ++m) { uint q = m & 3u; dr[m] = 2u * (lane & 3u) + (q < 2u ? 9u : 3u) - q; dc[m] = (lane >> 2) + (m < 4u ? 8u : 0u); }
        #pragma clang loop unroll_count(4)
        for (uint tk = 0; tk < 144u; ++tk) {
            const device short* tile = base + (tk * 320u + tn) * (16u * K);
            const device uint* tile32 = reinterpret_cast<const device uint*>(tile);
            if (K == 3) {
            uint rd8 = tile32[wa >> 1u];
            uint rd9 = tile32[wb >> 1u];
            // The lane's 8 windows sit at bit offsets 3m (m = 0..7) of v = (rd9:rd8) >> lp.  Windows 0..5 end at
            // bit 31 at most, 6..7 need bits 18..40: two 32-bit funnel-shifted words cover all eight without any
            // 64-bit shift in the loop (Apple GPUs emulate ulong shifts with several 32-bit ops).
            // lane_p (lp) is one of (0, 8, 16, 24): funnel-shift the two 32-bit halves accordingly.
            uint v0 = (lp == 0u) ? rd8 : ((rd8 >> lp) | (rd9 << (32u - lp)));      // bits 0..31 of v
            uint v1 = (lp < 16u) ? ((rd8 >> (lp + 16u)) | (rd9 << (16u - lp)))     // bits 16..47 of v
                                 : (rd9 >> (lp - 16u));
            for (uint m = 0; m < 6u; ++m) {
                uint window = (v0 >> (3u * m)) & 0xffffu;
                uint rr = window * 0x83DCD12Du;
                uint bs = (rr & 0x00FF00FFu) + ((rr >> 8u) & 0x00FF00FFu);
                half hs = as_type<half>(ushort((bs * 0x00010001u + 0x64000000u) >> 16u));
                float w = float(fma(hs, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u))));
                acc[m] = fma(xp[tk * 16u + dr[m]], w, acc[m]);
            }
            for (uint m = 6u; m < 8u; ++m) {
                uint window = (v1 >> (3u * m - 16u)) & 0xffffu;
                uint rr = window * 0x83DCD12Du;
                uint bs = (rr & 0x00FF00FFu) + ((rr >> 8u) & 0x00FF00FFu);
                half hs = as_type<half>(ushort((bs * 0x00010001u + 0x64000000u) >> 16u));
                float w = float(fma(hs, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u))));
                acc[m] = fma(xp[tk * 16u + dr[m]], w, acc[m]);
            }
            } else {
                for (uint m = 0; m < 8u; ++m) {
                    const uint window = (K == 2u ? rate_window<2>(tile32, 8u * lane + 7u - m) : (K == 4u ? rate_window<4>(tile32, 8u * lane + 7u - m) : rate_window<5>(tile32, 8u * lane + 7u - m)));
                    const float w = float(dig_cb(window));
                    acc[m] = fma(xp[tk * 16u + dr[m]], w, acc[m]);
                }
            }
        }
        for (uint c = 0; c < 16u; ++c) red[c][lane] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint m = 0; m < 8u; ++m) red[dc[m]][lane] += acc[m];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 16u) {
            float s = 0.0f;
            for (uint l = 0; l < 32u; ++l) s += red[lane][l];
            out[row * 5120u + tn * 16u + lane] = s;
        }
    