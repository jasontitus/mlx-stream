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
    constexpr int KD = 5120;
    constexpr int ND = 2304;
    constexpr int K16 = KD / 16;
    constexpr int N16 = ND / 16;
    constexpr int NTILES = ND / BN;
    constexpr int S = KD / BK;
    constexpr int KMAX = 5;
    threadgroup uint4 dig_smem[BN * LDB * 2 / 16];
    threadgroup uint4 stage_smem[KT * NT * 8 * KMAX / 4];  // one stage's 16 tiles at the top rate: 2560 bytes
    threadgroup half* Bs = reinterpret_cast<threadgroup half*>(dig_smem);
    threadgroup uint* stage_words = reinterpret_cast<threadgroup uint*>(stage_smem);

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
    const device int16_t* code = op == 0 ? code0 : code1;
    const ulong di = ulong(slot) * 2ul;
    const ulong offset = op == 0 ? desc0[di] : desc1[di];
    const int K = int(op == 0 ? desc0[di + 1ul] : desc1[di + 1ul]);
    const int WPT = 8 * K;
    // loader (the probe's map): job = trellis tile t (kt, nt) of the stage, column half h = sg & 1 (uniform per
    // simdgroup), column c7 = lane & 7 -> B^T row 16 nt + 8 h + c7, K slots 16 kt .. 16 kt + 15. The stage's
    // words are staged in threadgroup memory first (rate_stage_jobs), once per stage.
    static_assert(JOBS == 1, "rate_stage_jobs: one job per thread");
    const uint h = sg & 1u;
    const uint c7 = lane & 7u;
    const uint t = (lane >> 3u) + 4u * (sg >> 1u);
    const int kt = int(t) / NT;
    const int nt = int(t) % NT;
    const int dst_off = (nt * 16 + 8 * int(h) + int(c7)) * LDB + kt * 16;
    const device uint* cd = reinterpret_cast<const device uint*>(code + offset) + size_t(n0 / 16) * size_t(WPT);
    const device half* xa = (op == 0 ? x0 : x1) + (size_t(row0) + size_t(m0 + tm)) * size_t(KD);
    const bool full = sgp_sm == SM;
    NAXTile<float, TM, TN> Dtile;
    Dtile.clear();
    for (int s = 0; s < S; ++s) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const device uint* stage = cd + size_t(s) * size_t(KT * N16 * WPT);
        const uint row_words = uint(N16 * WPT);
        if (K == 2) rate_stage_jobs<2>(stage, row_words, tid, h, c7, t, stage_words, Bs + dst_off);
        else if (K == 3) rate_stage_jobs<3>(stage, row_words, tid, h, c7, t, stage_words, Bs + dst_off);
        else if (K == 4) rate_stage_jobs<4>(stage, row_words, tid, h, c7, t, stage_words, Bs + dst_off);
        else rate_stage_jobs<5>(stage, row_words, tid, h, c7, t, stage_words, Bs + dst_off);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgp_sm > 0) {
            STEEL_PRAGMA_NO_UNROLL
            for (int i = 0; i < 2; ++i) {
                NAXTile<half, TM, TK> Atile;
                NAXTile<half, TN, TK> Btile;
                volatile int compiler_barrier;
                if (full) {
                    Atile.load(xa + s * BK + i * SK, KD);
                } else {
                    Atile.load_safe(xa + s * BK + i * SK, KD, short2(SK, sgp_sm));
                }
                Btile.template load<half, LDB, 1>(Bs + tn * LDB + i * SK);
                tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
                (void)compiler_barrier;
            }
        }
    }
    if (sgp_sm > 0) {
        device float* zp = (op == 0 ? z0 : z1) + (size_t(row0) + size_t(m0 + tm)) * size_t(ND) + n0 + tn;
        if (full) {
            Dtile.store(zp, ND);
        } else {
            Dtile.store_safe(zp, ND, short2(SN, sgp_sm));
        }
    }
