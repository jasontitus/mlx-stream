    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int WN = 2;
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr int SK = 32;
    constexpr int TM = 2;
    constexpr int TN = 2;
    constexpr int TK = 2;
    constexpr int NSG = 8;
    constexpr int JOBS = 8 / NSG;
    constexpr int LDB = BK + 8;
    constexpr int KT = BK / 16;
    constexpr int NT = BN / 16;
    constexpr int KD = 2304;
    constexpr int ND = 5120;
    constexpr int K16 = KD / 16;
    constexpr int N16 = ND / 16;
    constexpr int NTILES = ND / BN;
    constexpr int S = KD / BK;
    threadgroup uint4 dig_smem[BN * LDB * 2 / 16];
    threadgroup half* Bs = reinterpret_cast<threadgroup half*>(dig_smem);

    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = tid & 31u;
    const int g = int(threadgroup_position_in_grid.x);
    // this threadgroup's expert j: the last table entry whose first threadgroup is <= g (unused entries: INT_MAX)
    int j = 0;
    STEEL_PRAGMA_UNROLL
    for (int q = 1; q < 16; ++q) {
        j = tbl[48 + q] <= g ? q : j;
    }
    const int slot = tbl[j];
    const int row0 = tbl[16 + j];
    const int rows = tbl[32 + j];
    const int local = g - tbl[48 + j];
    const int tiles_m = (rows + BM - 1) / BM;
    const int nt_all = local / tiles_m;
    const int m0 = (local - nt_all * tiles_m) * BM;
    const int op = nt_all / NTILES;
    const int n0 = (nt_all - op * NTILES) * BN;
    const int tm = SM * int(sg / WN);
    const short tn = short(SN * int(sg % WN));
    const int sgp_sm = min(SM, max(0, min(BM, rows - m0) - tm));
    (void)op;
    const device int16_t* code = code0;
    const ulong di = ulong(slot) * 2ul;
    const ulong offset = desc[di];
    const int K = int(desc[di + 1ul]);
    const int WPT = 8 * K;
    // loader (the probe's map): job jj = trellis tile t (kt, nt) of the stage, column half h = sg & 1 (uniform per
    // simdgroup), column c7 = lane & 7 -> B^T row 16 nt + 8 h + c7, K slots 16 kt .. 16 kt + 15
    const uint h = sg & 1u;
    const uint c7 = lane & 7u;
    int dst_off[JOBS];
    int src_off[JOBS];
    STEEL_PRAGMA_UNROLL
    for (int jj = 0; jj < JOBS; ++jj) {
        const int t = int(lane >> 3u) + 4 * (int(sg >> 1u) + (NSG / 2) * jj);
        const int kt = t / NT;
        const int nt = t % NT;
        dst_off[jj] = (nt * 16 + 8 * int(h) + int(c7)) * LDB + kt * 16;
        src_off[jj] = (kt * N16 + nt) * WPT;
    }
    const device uint* cd = reinterpret_cast<const device uint*>(code + offset) + size_t(n0 / 16) * size_t(WPT);
    const device half* xa = x + (size_t(row0) + size_t(m0 + tm)) * size_t(KD);
    const bool full = sgp_sm == SM;
    NAXTile<float, TM, TN> Dtile;
    Dtile.clear();
    // stages: decode + matmul per BK, with the next stage's code words loaded ahead of the matmul
    // (rate_gemm_stages); K is uniform per threadgroup, so it is dispatched here, once
    if (K == 2) rate_gemm_stages<2, KD, ND, JOBS>(cd, src_off, dst_off, c7, h, Bs, xa, sgp_sm, tn, Dtile);
    else if (K == 3) rate_gemm_stages<3, KD, ND, JOBS>(cd, src_off, dst_off, c7, h, Bs, xa, sgp_sm, tn, Dtile);
    else if (K == 4) rate_gemm_stages<4, KD, ND, JOBS>(cd, src_off, dst_off, c7, h, Bs, xa, sgp_sm, tn, Dtile);
    else rate_gemm_stages<5, KD, ND, JOBS>(cd, src_off, dst_off, c7, h, Bs, xa, sgp_sm, tn, Dtile);
    if (sgp_sm > 0) {
        device float* zp = z0 + (size_t(row0) + size_t(m0 + tm)) * size_t(ND) + n0 + tn;
        if (full) {
            Dtile.store(zp, ND);
        } else {
            Dtile.store_safe(zp, ND, short2(SN, sgp_sm));
        }
    }
