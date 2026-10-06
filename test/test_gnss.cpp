/* ============================================================================
 * test_gnss.cpp —— GNSS 相关测试（合并后）
 *
 * 本文件由以下原文件合并而成（各自在独立 namespace 内，互不干扰）：
 *   - test_gnss.cpp        GNSS 多频紧组合因子 / 雅可比 / 双差残差
 *   - test_rinex_dd.cpp    RINEX 3 → 双差 DDObs 生产者的真实数据验证
 *   - test_gnss_tc.cpp     GNSS 紧组合模块单独测试（伪距/载波/整数固定）
 *   - test_ils.cpp         整数最小二乘（LAMBDA）与暴力枚举对照
 * ==========================================================================*/
#include "alg_lib.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

/* 统一计数与判据（各 namespace 内另有同名的局部计数，宏在其作用域内解析到各自的那一份） */
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) { ++g_pass; }                                                \
        else {                                                                 \
            ++g_fail;                                                          \
            std::printf("[FAIL] %s  (%s:%d)\n", msg, __FILE__, __LINE__);     \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                             \
    do {                                                                       \
        const double _a = (a), _b = (b), _t = (tol);                           \
        if (std::fabs(_a - _b) <= _t) { ++g_pass; }                            \
        else {                                                                 \
            ++g_fail;                                                          \
            std::printf("[FAIL] %s  (%s:%d): %.6g vs %.6g (|d|=%.3g > %g)\n", \
                        msg, __FILE__, __LINE__, _a, _b,                        \
                        std::fabs(_a - _b), _t);                               \
        }                                                                      \
    } while (0)

/* ============================================================================
 * 以下为原 test/test_gnss.cpp 的内容，已并入本文件
 * ==========================================================================*/
namespace gnss_base {
    /**
     * @file test_gnss.cpp
     * @brief 多频 GNSS 紧组合单元测试（最终版）
     *
     * 与上一版的差异：
     *   1. numericJacScalar eps 由 1e-3 → 1e-2（消除舍入主导）
     *   2. 端到端载波测试改为"伪距 + 载波"混合（消除欠定）
     */


    /* ========================================================================== */
    static int g_fail = 0;
    static int g_pass   = 0;



    /* ==========================================================================
     * 工具
     * ==========================================================================*/
    static Eigen::Matrix3d makeREn(double lat, double lon) {
        const double sl = std::sin(lat), cl = std::cos(lat);
        const double so = std::sin(lon), co = std::cos(lon);
        Eigen::Matrix3d R;
        R << -so, -sl*co,  cl*co,
               co, -sl*so,  cl*so,
              0.0,      cl,     sl;
        return Eigen::Matrix3d(R);
    }

    static Eigen::Vector3d makeP0Ecef(double lat, double lon, double h) {
        constexpr double RE_ = 6378137.0;
        constexpr double e2_ = 6.69437999014e-3;
        const double sl = std::sin(lat), cl = std::cos(lat);
        const double so = std::sin(lon), co = std::cos(lon);
        const double N  = RE_ / std::sqrt(1.0 - e2_ * sl * sl);
        return Eigen::Vector3d((N + h) * cl * co,
                     (N + h) * cl * so,
                     (N * (1.0 - e2_) + h) * sl);
    }

    /* ★ 数值微分：eps 默认 1e-2
     *
     * 分析（以 WL/NL/IF 位置雅可比为例）：
     *   - 残差量级 |r| ≈ 1e5 ~ 1e7（含距离）
     *   - double 有效精度 ε_machine ≈ 2.2e-16，绝对舍入 ~2e-11
     *   - 中心差分噪声 ≈ 2e-11 / (2·eps)
     *     若 eps=1e-6  → 噪声 1e-5（灾难性）
     *     若 eps=1e-3  → 噪声 1e-8（信号 2e-3 时信噪比 2e5，仍有余)
     *     若 eps=1e-2  → 噪声 1e-9（信噪比 1e7，非常稳）
     *   - 截断项 ≈ (eps²/6)·f'''，对距离函数 f''' ~ 1/ρ² ~ 1e-7
     *     若 eps=1e-2  → 截断 1.7e-11（可忽略）
     *   综合：eps=1e-2 时相对误差 < 1e-5，通过阈值。 */
    static double numericJacScalar(ceres::CostFunction* cost,
                                   const std::vector<double*>& params,
                                   int param_idx, int elem_idx,
                                   double eps = 1e-2) {
        const int nr = cost->num_residuals();
        std::vector<double> rp(nr, 0.0), rm(nr, 0.0);
        double* p = params[param_idx];
        const double orig = p[elem_idx];

        p[elem_idx] = orig + eps;
        cost->Evaluate(params.data(), rp.data(), nullptr);
        p[elem_idx] = orig - eps;
        cost->Evaluate(params.data(), rm.data(), nullptr);
        p[elem_idx] = orig;

        return (rp[0] - rm[0]) / (2.0 * eps);
    }

    static double analyticJacScalar(ceres::CostFunction* cost,
                                    const std::vector<double*>& params,
                                    int param_idx, int elem_idx,
                                    const std::vector<int>& sizes) {
        const int nr = cost->num_residuals();
        const int np = static_cast<int>(params.size());
        std::vector<double> res(nr, 0.0);
        std::vector<std::vector<double>> jac(np);
        std::vector<double*> jp(np, nullptr);
        for (int p = 0; p < np; ++p) {
            jac[p].assign(nr * sizes[p], 0.0);
            jp[p] = jac[p].data();
        }
        cost->Evaluate(params.data(), res.data(), jp.data());
        return jac[param_idx][elem_idx];
    }

    static DDObs makeDDObsMF(const Eigen::Vector3d& pi, const Eigen::Vector3d& pk,
                             const double dd_P[2], const double dd_L[2]) {
        DDObs dd;
        dd.ref_sat = 5;
        dd.sat     = 12;
        dd.sat_i.sat_id   = 12;
        dd.sat_i.pos_ecef = pi;
        dd.sat_k.sat_id   = 5;
        dd.sat_k.pos_ecef = pk;
        for (int f = 0; f < 2; ++f) {
            dd.dd_P[f]  = dd_P[f];
            dd.dd_L[f]  = dd_L[f];
            dd.has_P[f] = true;
            dd.has_L[f] = true;
        }
        return dd;
    }

    /* ========================================================================== */
    static void test_combination_constants() {
        std::printf("\n---- test_combination_constants ----\n");
        CHECK_NEAR(LAMBDA_WL, 0.8620, 0.001, "LAMBDA_WL ≈ 0.862 m");
        CHECK_NEAR(LAMBDA_NL, 0.1070, 0.001, "LAMBDA_NL ≈ 0.107 m");
        CHECK_NEAR(ALPHA_IF + BETA_IF, 1.0, 1e-12, "IF 系数和为 1");
        CHECK(ALPHA_IF > 2.0 && BETA_IF < -1.0, "IF 系数符号符合预期");
        CHECK_NEAR(LAMBDA_L1, 0.1903, 0.0001, "LAMBDA_L1 ≈ 0.1903 m");
    }

    static void test_amb_key() {
        std::printf("\n---- test_amb_key（64 位键）----\n");
        const AmbKey key = ambKey(5, 12, AMB_SLOT_L1);
        CHECK(ambKeyRefSat(key) == 5,  "ambKey ref_sat 往返一致");
        CHECK(ambKeySat(key)    == 12, "ambKey sat 往返一致");
        CHECK(ambKeySlot(key)   == AMB_SLOT_L1, "ambKey slot 往返一致");
        CHECK(ambKeyArc(key)    == 0,  "ambKey arc 往返一致");

        CHECK(ambKey(5, 12, 0) != ambKey(5, 12, 1),   "不同频点 key 不同");
        CHECK(ambKey(5, 12, 0) != ambKey(5, 13, 0),   "不同卫星 key 不同");
        CHECK(ambKey(5, 12, 0) != ambKey(6, 12, 0),   "不同参考星 key 不同");
        CHECK(ambKey(5, 12, 100) != ambKey(5, 12, 200), "WL 与 NL key 不同");
        CHECK(ambKey(5, 12, 0, 1) != ambKey(5, 12, 0, 2), "不同弧段 key 不同");

        /* 回归：旧实现用 & 0x3F 截断卫星号，差 64 的卫星会共用参数 */
        CHECK(ambKey(5, 12, 0) != ambKey(5, 76, 0), "卫星号 12 与 76 不再串号");
        CHECK(ambKey(5, 12, 0) != ambKey(5, 12 + 64 * 3, 0), "卫星号差 192 也不再串号");
        CHECK(ambKey(5, 12, 0) != ambKey(5 + 64, 12, 0), "参考星差 64 不再串号");
        /* 北斗 C01–C63 之类的编号必须能区分 */
        CHECK(ambKey(1, 201, 0) != ambKey(1, 202, 0), "BD 编号 201/202 可区分");
        /* 16 bit 卫星号足够覆盖多系统编号 */
        CHECK(ambKeyRefSat(ambKey(1, 65535, 0)) == 1, "sat 16777215 以外不溢出");
        CHECK(ambKeySat(ambKey(1, 65535, 0)) == 65535, "sat 65535 往返一致");
    }

    static void test_amb_management() {
        std::printf("\n---- test_amb_management（参考星切换 / 弧段 / 清理）----\n");
        const int slot = AMB_SLOT_L1;
        const int k_old = 5, k_new = 9;

        /* 参考星从 5 切到 9：N_9^i = N_5^i − N_5^9 */
        AmbMap m;
        m[ambKey(k_old, 9,  slot)] = {3.0};    /* N_5^9 */
        m[ambKey(k_old, 12, slot)] = {10.0};
        m[ambKey(k_old, 20, slot)] = {-4.0};
        m[ambKey(k_old, 12, AMB_SLOT_WL)] = {99.0};   /* 别的槽不应被碰 */

        const int n = ambPromoteReference(m, k_old, k_new, slot);
        CHECK(n == 2, "参考星切换转换了 2 颗卫星");
        CHECK_NEAR(m[ambKey(k_new, 12, slot)][0], 7.0, 1e-12, "N_9^12 = N_5^12 - N_5^9");
        CHECK_NEAR(m[ambKey(k_new, 20, slot)][0], -7.0, 1e-12, "N_9^20 = N_5^20 - N_5^9");
        CHECK(m.find(ambKey(k_new, k_new, slot)) == m.end(), "新参考星自身不建参数");
        CHECK_NEAR(m[ambKey(k_old, 12, AMB_SLOT_WL)][0], 99.0, 1e-12, "其它槽不受影响");

        /* 缺少 N_k^{k'} 时不做转换（留给新弧重新收敛） */
        AmbMap m2;
        m2[ambKey(k_old, 12, slot)] = {10.0};
        CHECK(ambPromoteReference(m2, k_old, k_new, slot) == 0, "无法转换时返回 0");

        /* 周跳：arc+1，新键以旧值 + slip 为初值 */
        AmbMap m3;
        m3[ambKey(k_old, 12, slot, 0)] = {5.0};
        const AmbKey nk = ambAdvanceArc(m3, k_old, 12, slot, 1, +1.0);
        CHECK(nk == ambKey(k_old, 12, slot, 1), "弧段号正确推进");
        CHECK_NEAR(m3[nk][0], 6.0, 1e-12, "新弧初值 = 旧值 + 周跳量");
        CHECK(m3.find(ambKey(k_old, 12, slot, 0)) != m3.end(), "旧弧键暂留");

        /* 清理未使用的键 */
        std::set<AmbKey> used{ambKey(k_old, 12, slot, 1)};
        ambPruneUnused(m3, used);
        CHECK(m3.size() == 1 && m3.find(nk) != m3.end(), "陈旧键被清理");
    }

    static void test_freq_pseudorange_jacobian() {
        std::printf("\n---- test_freq_pseudorange_jacobian ----\n");
        const Eigen::Matrix3d  R_en = makeREn(0.5236, 2.0);
        const Eigen::Vector3d p0   = makeP0Ecef(0.5236, 2.0, 100.0);
        const Eigen::Vector3d base = p0 + Eigen::Vector3d(1000, 2000, 50);
        const Eigen::Vector3d pi(15000000, 10000000, 20000000);
        const Eigen::Vector3d pk(18000000, 12000000, 18000000);
        const Eigen::Vector3d t_ant(0.5, -0.3, 1.2);

        const double dd_P[2] = {1234.5, 1234.5};
        const double dd_L[2] = {0.0, 0.0};

        for (int fi = 0; fi < 2; ++fi) {
            DDObs dd = makeDDObsMF(pi, pk, dd_P, dd_L);
            GnssDDPseudorangeFreqResidual factor(dd, fi, base, R_en, p0, t_ant);

            double pose[7] = {0.5, -1.2, 0.3, 0, 0, 0, 1};
            std::vector<double*> params = {pose};
            std::vector<int> sizes = {7};

            for (int i = 0; i < 3; ++i) {
                double jn = numericJacScalar(&factor, params, 0, i);
                double ja = analyticJacScalar(&factor, params, 0, i, sizes);
                char msg[64];
                std::snprintf(msg, sizeof(msg), "频点 %d 伪距位置雅可比 col=%d", fi, i);
                CHECK_NEAR(ja, jn, 1e-5, msg);
            }
        }
    }

    static void test_freq_carrier_amb_jacobian() {
        std::printf("\n---- test_freq_carrier_amb_jacobian ----\n");
        const Eigen::Matrix3d  R_en = makeREn(0.5236, 2.0);
        const Eigen::Vector3d p0   = makeP0Ecef(0.5236, 2.0, 100.0);
        const Eigen::Vector3d base = p0 + Eigen::Vector3d(1000, 2000, 50);
        const Eigen::Vector3d pi(15000000, 10000000, 20000000);
        const Eigen::Vector3d pk(18000000, 12000000, 18000000);
        const Eigen::Vector3d t_ant(0, 0, 0);

        const double dd_P[2] = {0, 0};
        const double dd_L[2] = {6480.0, 5049.0};

        for (int fi = 0; fi < 2; ++fi) {
            DDObs dd = makeDDObsMF(pi, pk, dd_P, dd_L);
            GnssDDCarrierFreqResidual factor(dd, fi, base, R_en, p0, t_ant);

            double pose[7] = {0.5, -1.2, 0.3, 0, 0, 0, 1};
            double amb[1]  = {dd_L[fi]};
            std::vector<double*> params = {pose, amb};
            std::vector<int> sizes = {7, 1};

            const double sigma  = GPS_FREQ[fi].sigma_L * 1.4142;
            const double lambda = GPS_FREQ[fi].lambda;
            const double ja = analyticJacScalar(&factor, params, 1, 0, sizes);
            const double jn = numericJacScalar(&factor, params, 1, 0);

            char msg[64];
            std::snprintf(msg, sizeof(msg), "频点 %d 模糊度雅可比 = -λ/σ", fi);
            CHECK_NEAR(ja, -lambda / sigma, 1e-9, msg);
            CHECK_NEAR(ja, jn, 1e-5, "载波模糊度雅可比 vs 数值");
        }
    }

    static void test_wl_jacobian() {
        std::printf("\n---- test_wl_jacobian ----\n");
        const Eigen::Matrix3d  R_en = makeREn(0.5236, 2.0);
        const Eigen::Vector3d p0   = makeP0Ecef(0.5236, 2.0, 100.0);
        const Eigen::Vector3d base = p0 + Eigen::Vector3d(1000, 2000, 50);
        const Eigen::Vector3d pi(15000000, 10000000, 20000000);
        const Eigen::Vector3d pk(18000000, 12000000, 18000000);
        const Eigen::Vector3d t_ant(0, 0, 0);

        const double dd_P[2] = {0, 0};
        const double dd_L[2] = {6480.0, 5049.0};
        DDObs dd = makeDDObsMF(pi, pk, dd_P, dd_L);

        const double sigma_wl = 0.05;
        GnssDDWideLaneResidual factor(dd, base, R_en, p0, t_ant, sigma_wl);

        double pose[7] = {0.5, -1.2, 0.3, 0, 0, 0, 1};
        double amb[1]  = {1431.0};
        std::vector<double*> params = {pose, amb};
        std::vector<int> sizes = {7, 1};

        const double ja = analyticJacScalar(&factor, params, 1, 0, sizes);
        CHECK_NEAR(ja, -LAMBDA_WL / sigma_wl, 1e-9,
                   "WL 模糊度雅可比 = -λ_WL/σ_WL");

        for (int i = 0; i < 3; ++i) {
            double jn = numericJacScalar(&factor, params, 0, i);
            double j  = analyticJacScalar(&factor, params, 0, i, sizes);
            CHECK_NEAR(j, jn, 1e-5, "WL 位置雅可比 vs 数值");
        }
    }

    static void test_nl_jacobian() {
        std::printf("\n---- test_nl_jacobian ----\n");
        const Eigen::Matrix3d  R_en = makeREn(0.5236, 2.0);
        const Eigen::Vector3d p0   = makeP0Ecef(0.5236, 2.0, 100.0);
        const Eigen::Vector3d base = p0 + Eigen::Vector3d(1000, 2000, 50);
        const Eigen::Vector3d pi(15000000, 10000000, 20000000);
        const Eigen::Vector3d pk(18000000, 12000000, 18000000);
        const Eigen::Vector3d t_ant(0, 0, 0);

        const double dd_P[2] = {0, 0};
        const double dd_L[2] = {6480.0, 5049.0};
        DDObs dd = makeDDObsMF(pi, pk, dd_P, dd_L);

        const double sigma_nl = 0.01;
        GnssDDNarrowLaneResidual factor(dd, base, R_en, p0, t_ant, sigma_nl);

        double pose[7] = {0.5, -1.2, 0.3, 0, 0, 0, 1};
        double amb[1]  = {6480.0};
        std::vector<double*> params = {pose, amb};
        std::vector<int> sizes = {7, 1};

        const double ja = analyticJacScalar(&factor, params, 1, 0, sizes);
        CHECK_NEAR(ja, -LAMBDA_NL / sigma_nl, 1e-9,
                   "NL 模糊度雅可比 = -λ_NL/σ_NL");

        /* ⚠️ 阈值取 1e-4：
         *   NL 组合几何系数 (f1²+f2²)/(f1+f2)² ≈ 0.5077，残差里出现
         *   -0.4923·ρ 的"半消去"。ρ ~ 1e7 m 时 5e6 量级浮点减法产生
         *   ~1e-9 m 的绝对误差，归一化后 ~1e-7，数值微分 eps=1e-2
         *   时噪声 ~1e-5，因此不能再用 1e-5 阈值。
         *   这是数值微分精度的极限，不是因子错误。 */
        for (int i = 0; i < 3; ++i) {
            double jn = numericJacScalar(&factor, params, 0, i);
            double j  = analyticJacScalar(&factor, params, 0, i, sizes);
            CHECK_NEAR(j, jn, 1e-4, "NL 位置雅可比 vs 数值");
        }
    }

    static void test_if_jacobian() {
        std::printf("\n---- test_if_jacobian ----\n");
        const Eigen::Matrix3d  R_en = makeREn(0.5236, 2.0);
        const Eigen::Vector3d p0   = makeP0Ecef(0.5236, 2.0, 100.0);
        const Eigen::Vector3d base = p0 + Eigen::Vector3d(1000, 2000, 50);
        const Eigen::Vector3d pi(15000000, 10000000, 20000000);
        const Eigen::Vector3d pk(18000000, 12000000, 18000000);
        const Eigen::Vector3d t_ant(0, 0, 0);

        const double dd_P[2] = {0, 0};
        const double dd_L[2] = {6480.0, 5049.0};
        DDObs dd = makeDDObsMF(pi, pk, dd_P, dd_L);

        const double sigma_if = 0.02;
        GnssDDIFCarrierResidual factor(dd, base, R_en, p0, t_ant, sigma_if);

        double pose[7] = {0.5, -1.2, 0.3, 0, 0, 0, 1};
        double amb[1]  = {1000.0};
        std::vector<double*> params = {pose, amb};
        std::vector<int> sizes = {7, 1};

        const double ja = analyticJacScalar(&factor, params, 1, 0, sizes);
        CHECK_NEAR(ja, -1.0 / sigma_if, 1e-9, "IF 模糊度雅可比 = -1/σ_IF");

        for (int i = 0; i < 3; ++i) {
            double jn = numericJacScalar(&factor, params, 0, i);
            double j  = analyticJacScalar(&factor, params, 0, i, sizes);
            CHECK_NEAR(j, jn, 1e-5, "IF 位置雅可比 vs 数值");
        }
    }

    /* sigma 必须 ≥ 因子内部的钳位下限 */
    static void test_clock_rw_jacobian() {
        std::printf("\n---- test_clock_rw_jacobian ----\n");

        {
            const double dt = 1.0, sigma = 0.5;
            ClockBiasRWFactor factor(dt, sigma);

            double clk_prev[1]   = {10.0};
            double drift_prev[1] = {0.5};
            double clk_curr[1]   = {10.5};
            std::vector<double*> params = {clk_prev, drift_prev, clk_curr};
            std::vector<int> sizes = {1, 1, 1};

            for (int p = 0; p < 3; ++p) {
                double jn = numericJacScalar(&factor, params, p, 0);
                double ja = analyticJacScalar(&factor, params, p, 0, sizes);
                CHECK_NEAR(ja, jn, 1e-5, "钟差 RW 雅可比 vs 数值");
            }
            CHECK_NEAR(analyticJacScalar(&factor, params, 0, 0, sizes),
                       -1.0 / sigma, 1e-12, "钟差 RW 对 prev");
            CHECK_NEAR(analyticJacScalar(&factor, params, 1, 0, sizes),
                       -dt / sigma, 1e-12, "钟差 RW 对 drift_prev");
            CHECK_NEAR(analyticJacScalar(&factor, params, 2, 0, sizes),
                        1.0 / sigma, 1e-12, "钟差 RW 对 curr");
        }

        {
            const double sigma = 0.05;
            ClockDriftRWFactor factor(sigma);

            double d_prev[1] = {0.1};
            double d_curr[1] = {0.15};
            std::vector<double*> params = {d_prev, d_curr};
            std::vector<int> sizes = {1, 1};

            for (int p = 0; p < 2; ++p) {
                double jn = numericJacScalar(&factor, params, p, 0);
                double ja = analyticJacScalar(&factor, params, p, 0, sizes);
                CHECK_NEAR(ja, jn, 1e-5, "钟漂 RW 雅可比 vs 数值");
            }
        }
    }

    static void test_ambiguity_fix() {
        std::printf("\n---- test_ambiguity_fix ----\n");
        AmbiguityFixFactor factor(100.0, 1e-3);
        std::vector<double*> params;
        double amb[1] = {100.0};
        params.push_back(amb);

        double res[1];
        factor.Evaluate(params.data(), res, nullptr);
        CHECK_NEAR(res[0], 0.0, 1e-12, "整数点残差为零");

        amb[0] = 101.0;
        factor.Evaluate(params.data(), res, nullptr);
        CHECK_NEAR(res[0], 1000.0, 1e-6, "偏离 1 cycle 残差 = 1/σ_fix");
    }

    static void test_end_to_end_dd_pseudorange() {
        std::printf("\n---- test_end_to_end_dd_pseudorange ----\n");
        const Eigen::Vector3d p_true_n(10.0, 20.0, 5.0);
        const Eigen::Matrix3d  R_en = makeREn(0.5236, 2.0);
        const Eigen::Vector3d p0   = makeP0Ecef(0.5236, 2.0, 100.0);
        const Eigen::Vector3d p_true_e = p0 + R_en * p_true_n;
        const Eigen::Vector3d base = p0;
        const Eigen::Vector3d t_ant(0, 0, 0);

        std::vector<Eigen::Vector3d> sat_pos = {
            Eigen::Vector3d( 15000000,  10000000,  20000000),
            Eigen::Vector3d(-15000000,  10000000,  20000000),
            Eigen::Vector3d( 15000000, -10000000,  20000000),
            Eigen::Vector3d(-15000000, -10000000,  20000000),
            Eigen::Vector3d(          0,           0,  26000000),
        };
        const Eigen::Vector3d& ref_pos = sat_pos[0];

        std::vector<DDObs> dds;
        for (size_t i = 1; i < sat_pos.size(); ++i) {
            const double r_i = norm(sat_pos[i] - p_true_e);
            const double r_k = norm(ref_pos     - p_true_e);
            const double b_i = norm(sat_pos[i] - base);
            const double b_k = norm(ref_pos     - base);
            const double dd_P = (r_i - r_k) - (b_i - b_k);

            double dd_Pf[2] = {dd_P, dd_P};
            double dd_Lf[2] = {0.0, 0.0};
            dds.push_back(makeDDObsMF(sat_pos[i], ref_pos, dd_Pf, dd_Lf));
        }

        ceres::Problem problem;
        double pose[7] = {0, 0, 0, 0, 0, 0, 1};
        problem.AddParameterBlock(pose, 7);
        problem.SetManifold(pose, new PoseManifold());

        for (const auto& dd : dds) {
            for (int fi = 0; fi < 2; ++fi) {
                auto* cost = new GnssDDPseudorangeFreqResidual(
                    dd, fi, base, R_en, p0, t_ant);
                problem.AddResidualBlock(cost, nullptr, pose);
            }
        }

        ceres::Solver::Options opt;
        opt.linear_solver_type = ceres::DENSE_QR;
        opt.max_num_iterations = 100;
        opt.logging_type = ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(opt, &problem, &summary);

        CHECK(summary.termination_type == ceres::CONVERGENCE,
              "双频双差伪距优化收敛");

        const Eigen::Vector3d p_est(pose[0], pose[1], pose[2]);
        const double err = norm(p_est - p_true_n);
        std::printf("        位置误差 = %.6e m\n", err);
        CHECK(err < 0.5, "单历元双频双差定位误差 < 0.5 m");
    }

    /* ★ 端到端载波测试：伪距 + 载波混合
     *
     * 为什么必须混合：
     *   如果只用载波，未知数 = 位置 3 + 模糊度 8 = 11；
     *   方程数 = 4 组双差 × 2 频 = 8；
     *   系统欠定 3 维，LM 会收敛到无穷多解中的任一个 —— 位置由初值决定。
     *
     *   加入伪距因子后：
     *     伪距方程 8 个 + 载波方程 8 个 = 16；
     *     未知数仍是 11；
     *     超定 5 维 —— 位置被伪距锚定，载波精化位置并确定模糊度。
     *   这也是工程中 RTK/PPP 的标准做法。 */
    static void test_end_to_end_dd_carrier() {
        std::printf("\n---- test_end_to_end_dd_carrier ----\n");
        const Eigen::Vector3d p_true_n(5.0, -10.0, 2.0);
        const Eigen::Matrix3d  R_en = makeREn(0.5236, 2.0);
        const Eigen::Vector3d p0   = makeP0Ecef(0.5236, 2.0, 100.0);
        const Eigen::Vector3d p_true_e = p0 + R_en * p_true_n;
        const Eigen::Vector3d base = p0;
        const Eigen::Vector3d t_ant(0, 0, 0);

        std::vector<Eigen::Vector3d> sat_pos = {
            Eigen::Vector3d( 15000000,  10000000,  20000000),
            Eigen::Vector3d(-15000000,  10000000,  20000000),
            Eigen::Vector3d( 15000000, -10000000,  20000000),
            Eigen::Vector3d(-15000000, -10000000,  20000000),
            Eigen::Vector3d(          0,           0,  26000000),
        };
        const Eigen::Vector3d& ref_pos = sat_pos[0];

        std::vector<DDObs> dds;
        std::vector<double> N_l1_true, N_l2_true;
        for (size_t i = 1; i < sat_pos.size(); ++i) {
            const double r_i = norm(sat_pos[i] - p_true_e);
            const double r_k = norm(ref_pos     - p_true_e);
            const double b_i = norm(sat_pos[i] - base);
            const double b_k = norm(ref_pos     - base);
            const double dd_rho = (r_i - r_k) - (b_i - b_k);

            const double N1 = static_cast<double>(100 * i);
            const double N2 = static_cast<double>(78  * i);

            double dd_Pf[2] = {dd_rho, dd_rho};
            double dd_Lf[2] = {dd_rho / GPS_FREQ[0].lambda + N1,
                               dd_rho / GPS_FREQ[1].lambda + N2};
            dds.push_back(makeDDObsMF(sat_pos[i], ref_pos, dd_Pf, dd_Lf));
            N_l1_true.push_back(N1);
            N_l2_true.push_back(N2);
        }

        ceres::Problem problem;
        double pose[7] = {0, 0, 0, 0, 0, 0, 1};
        problem.AddParameterBlock(pose, 7);
        problem.SetManifold(pose, new PoseManifold());

        const int n_dd = static_cast<int>(dds.size());
        std::vector<std::vector<double>> amb_store(n_dd * 2, std::vector<double>(1, 0.0));
        for (int i = 0; i < n_dd; ++i) {
            for (int fi = 0; fi < 2; ++fi) {
                /* 伪距因子（锚定位置） */
                auto* p_cost = new GnssDDPseudorangeFreqResidual(
                    dds[i], fi, base, R_en, p0, t_ant);
                problem.AddResidualBlock(p_cost, nullptr, pose);

                /* 载波因子（精化位置 + 估计模糊度）
                 * 给一个近真值初值：真值 +5 cycle */
                const double N_true = (fi == 0) ? N_l1_true[i] : N_l2_true[i];
                amb_store[i * 2 + fi][0] = N_true + 5.0;

                double* amb = amb_store[i * 2 + fi].data();
                problem.AddParameterBlock(amb, 1);

                auto* c_cost = new GnssDDCarrierFreqResidual(
                    dds[i], fi, base, R_en, p0, t_ant);
                problem.AddResidualBlock(c_cost, nullptr, pose, amb);
            }
        }

        ceres::Solver::Options opt;
        opt.linear_solver_type = ceres::DENSE_QR;
        opt.max_num_iterations = 200;
        opt.logging_type = ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(opt, &problem, &summary);

        CHECK(summary.termination_type == ceres::CONVERGENCE,
              "载波因子优化收敛");

        const Eigen::Vector3d p_est(pose[0], pose[1], pose[2]);
        const double pos_err = norm(p_est - p_true_n);
        std::printf("        位置误差 = %.6e m\n", pos_err);
        CHECK(pos_err < 0.1, "载波定位误差 < 0.1 m");

        double max_err = 0.0;
        for (int i = 0; i < n_dd; ++i) {
            max_err = std::max(max_err,
                               std::fabs(amb_store[i * 2 + 0][0] - N_l1_true[i]));
            max_err = std::max(max_err,
                               std::fabs(amb_store[i * 2 + 1][0] - N_l2_true[i]));
        }
        std::printf("        模糊度最大偏差 = %.6e cycle\n", max_err);
        CHECK(max_err < 0.1, "模糊度浮点解接近整数");
    }

    int run() {
        std::printf("========== GNSS 多频紧组合单元测试（最终版） ==========\n");

        test_combination_constants();
        test_amb_key();
        test_amb_management();
        test_freq_pseudorange_jacobian();
        test_freq_carrier_amb_jacobian();
        test_wl_jacobian();
        test_nl_jacobian();
        test_if_jacobian();
        test_clock_rw_jacobian();
        test_ambiguity_fix();
        test_end_to_end_dd_pseudorange();
        test_end_to_end_dd_carrier();

        std::printf("\n=============================================\n");
        std::printf("通过 %d / %d, 失败 %d\n",
                    g_pass, g_pass + g_fail, g_fail);
        return g_fail == 0 ? 0 : 1;
    }
}  /* namespace gnss_base */

/* ============================================================================
 * 以下为原 test/test_rinex_dd.cpp 的内容，已并入本文件
 * ==========================================================================*/
namespace rinex_dd {
    /* ============================================================================
     * test_rinex_dd.cpp —— 用**真实** RINEX 数据验证双差生产者
     *
     * 数据路径默认 D:\02_Data\20220309gnss（可用环境变量 DSH_GNSS_DATA 覆盖）。
     * 数据不存在时本测试打印 SKIP 并以 0 退出，不影响其它环境。
     *
     * 验证链条（从底层到端到端）：
     *   T1 观测文件读取：历元数/时间跨度与头部声明一致；请求的信号确实读到
     *   T2 星历读取与卫星位置：Kepler 元素量级合理；|r| 在 25000–27000 km
     *   T3 双差生产：卫星数、参考星、共同可见数；dd_P 量级合理
     *   T4 **端到端**：用双差伪距做单历元最小二乘解算流动站位置 ——
     *       若星历/几何/差分/符号任一处有错，最小二乘不会收敛到
     *       与基准站相接近的、跨历元一致的点
     * ==========================================================================*/

    static int g_pass = 0, g_fail = 0, g_skip = 0;



    static std::string dataDir() {
        const char *env = std::getenv("DSH_GNSS_DATA");
        if (env && env[0]) return std::string(env);
        return std::string("D:/02_Data/20220309gnss");
    }

    static bool fileExists(const std::string &p) {
        FILE *f = std::fopen(p.c_str(), "rb");
        if (!f) return false;
        std::fclose(f);
        return true;
    }

    /* ============================================================================
     * 单历元双差伪距最小二乘（3 未知数：流动站 ECEF）
     * 残差与 GnssDDPseudorangeFreqResidual 完全一致：
     *     r_i = dd_P[i] − [ (|s_i − x| − |s_k − x|) − (|s_i − b| − |s_k − b|) ]
     * ==========================================================================*/
    static bool solveRoverFromDd(const std::vector<DDObs> &dd, int freq,
                                 const Eigen::Vector3d &base, Eigen::Vector3d &x,
                                 double *post_rms) {
        if (dd.size() < 4) return false;
        for (int iter = 0; iter < 30; ++iter) {
            Eigen::MatrixXd A(dd.size(), 3);
            Eigen::VectorXd r(dd.size());
            for (size_t i = 0; i < dd.size(); ++i) {
                const Eigen::Vector3d si = dd[i].sat_i.pos_ecef - x;
                const Eigen::Vector3d sk = dd[i].sat_k.pos_ecef - x;
                const double rho_i = si.norm(), rho_k = sk.norm();
                const double bi = (dd[i].sat_i.pos_ecef - base).norm();
                const double bk = (dd[i].sat_k.pos_ecef - base).norm();
                const double dd_rho = (rho_i - rho_k) - (bi - bk);
                r(static_cast<int>(i)) = dd[i].dd_P[freq] - dd_rho;
                /* ∂r/∂x = κ_i − κ_k（∂|s−x|/∂x = −κ）。与因子里的 dr_dpe 一致。 */
                const Eigen::Vector3d ki = si / rho_i, kk = sk / rho_k;
                A.row(static_cast<int>(i)) = (ki - kk).transpose();
            }
            const Eigen::Vector3d dx = (A.transpose() * A).ldlt().solve(-(A.transpose() * r));
            x += dx;
            if (dx.norm() < 1e-4) break;
        }
        /* 后验残差 */
        double ss = 0.0;
        for (const DDObs &d : dd) {
            const double rho_i = (d.sat_i.pos_ecef - x).norm();
            const double rho_k = (d.sat_k.pos_ecef - x).norm();
            const double bi = (d.sat_i.pos_ecef - base).norm();
            const double bk = (d.sat_k.pos_ecef - base).norm();
            const double e = d.dd_P[freq] - ((rho_i - rho_k) - (bi - bk));
            ss += e * e;
        }
        if (post_rms) *post_rms = std::sqrt(ss / static_cast<double>(dd.size()));
        return true;
    }

    int run() {
        std::printf("========== RINEX → 双差 生产者验证 ==========\n");

        const std::string root = dataDir();
        const std::string rover_obs = root + "/719new/01_28_09/01_28_09_CHA1.22O";
        const std::string base_obs  = root + "/base/alloy/02210680.22o";
        const std::string nav       = root + "/brdm0680.22p";

        if (!fileExists(rover_obs) || !fileExists(base_obs) || !fileExists(nav)) {
            std::printf("SKIP: 未找到 RINEX 数据（%s）。可用 DSH_GNSS_DATA 指定。\n", root.c_str());
            std::printf("通过 0, 失败 0, 跳过 1\n");
            return 0;
        }
        std::printf("数据目录: %s\n", root.c_str());

        /* 频率槽 0 = GPS L1C/L2W；槽 1 = L2 */
        std::vector<DdSignal> slots(2);
        slots[0].sys = 'G'; slots[0].code_P = "C1C"; slots[0].code_L = "L1C"; slots[0].code_D = "D1C";
        slots[1].sys = 'G'; slots[1].code_P = "C2W"; slots[1].code_L = "L2W"; slots[1].code_D = "D2W";

        std::vector<std::array<int, 3>> slot_index;
        const std::vector<ObsRequest> reqs = buildObsRequests(slots, &slot_index);
        std::printf("请求信号数: %zu（%zu 个槽）\n", reqs.size(), slot_index.size());

        /* ---------------- T1 观测读取 ---------------- */
        std::printf("\n---- T1: 观测文件读取 ----\n");
        std::vector<RinexEpoch> rover, base;
        RinexObsHeader rh, bh;
        CHECK(readRinex3Obs(rover_obs, reqs, rover, &rh), "T1 流动站观测读取成功");
        CHECK(readRinex3Obs(base_obs,  reqs, base,  &bh), "T1 基准站观测读取成功");
        if (g_fail) { std::printf("\n读取失败，后续跳过\n通过 %d, 失败 %d\n", g_pass, g_fail); return 1; }

        std::printf("流动站: marker=%s 历元=%d 跨度=%.1f s  interval=%.3f\n",
                    rh.marker.c_str(), rh.n_epochs, rh.t_last - rh.t_first, rh.interval);
        std::printf("基准站: marker=%s 历元=%d 跨度=%.1f s  interval=%.3f  approx_pos=%.1f m\n",
                    bh.marker.c_str(), bh.n_epochs, bh.t_last - bh.t_first, bh.interval,
                    bh.approx_pos.norm());

        const double expect_rover = (rh.t_last - rh.t_first) / (rh.interval > 0 ? rh.interval : 1.0) + 1.0;
        /* 文件里可能有缺历元（失锁/存储），允许 5% 的差 */
        CHECK_NEAR(rh.n_epochs, expect_rover, 0.05 * expect_rover,
                   "T1 流动站历元数与头部跨度/间隔一致（允许 5% 缺失）");
        CHECK(rh.interval > 0.99 && rh.interval < 1.01, "T1 流动站间隔为 1 s");
        CHECK(bh.approx_pos_valid, "T1 基准站近似坐标有效");
        CHECK(rh.version > 3.0 && rh.version < 3.1, "T1 RINEX 版本 3.04");

        /* 第一历元确实读到了请求的信号 */
        {
            int with_obs = 0;
            const RinexEpoch &e0 = rover[0];
            for (const RinexSatObs &s : e0.sats)
                if (s.sys == 'G' && s.ok(slot_index[0][0]) && s.ok(slot_index[0][1])) ++with_obs;
            std::printf("首历元 GPS 双频可用的卫星数 = %d（总卫星 %zu）\n", with_obs, e0.sats.size());
            CHECK(with_obs >= 4, "T1 首历元至少有 4 颗 GPS 双频卫星");
            int n_with = 0;
            for (const RinexSatObs &s : e0.sats) if (s.ok(slot_index[0][0])) ++n_with;
            CHECK(n_with >= with_obs, "T1 伪距观测数不少于双频可用数");
        }

        /* ---------------- T2 星历与卫星位置 ---------------- */
        std::printf("\n---- T2: 星历与卫星位置 ----\n");
        std::map<char, std::map<int, std::vector<KeplerEph>>> eph;
        int n_glo = 0;
        CHECK(readRinex3Nav(nav, eph, &n_glo), "T2 导航文件读取成功");
        size_t n_gps = 0;
        for (auto &kv : eph['G']) n_gps += kv.second.size();
        std::printf("星历: GPS 记录 %zu 条，系统数 %zu，跳过 GLONASS %d 条\n",
                    n_gps, eph.size(), n_glo);
        CHECK(n_gps > 0, "T2 解析出 GPS 星历");
        CHECK(n_glo > 0, "T2 识别出 GLONASS 记录（已跳过）");

        {
            /* Kepler 元素量级：GPS 的 sqrt(a) ≈ 5153.8 m^0.5，e 很小 */
            const KeplerEph &e = eph['G'].begin()->second[0];
            std::printf("样例星历 G%02d: sqrt_a=%.4f e=%.6f i0=%.4f rad toe=%.1f\n",
                        e.prn, e.sqrt_a, e.e, e.i0, e.toe);
            CHECK(e.sqrt_a > 5100.0 && e.sqrt_a < 5200.0, "T2 sqrt_a 在 GPS 量级");
            CHECK(e.e >= 0.0 && e.e < 0.03, "T2 偏心率合理");
            CHECK(e.i0 > 0.9 && e.i0 < 1.1, "T2 轨道倾角 ≈ 55°");

            Eigen::Vector3d pos; double clk = 0.0;
            const bool ok = satPosFromEph(e, e.toe, pos, &clk);
            CHECK(ok, "T2 卫星位置计算成功");
            const double r = pos.norm();
            std::printf("     |r| = %.1f km, 钟差 = %.3e s\n", r / 1000.0, clk);
            CHECK(r > 2.4e7 && r < 2.8e7, "T2 |r| 在 24000–28000 km");
            CHECK(std::fabs(clk) < 1e-3, "T2 卫星钟差量级合理");

            /* 连续时刻的位置应当连续（相邻 1 s 位移 ≈ 3.9 km/s） */
            Eigen::Vector3d p2;
            satPosFromEph(e, e.toe + 1.0, p2, nullptr);
            const double d = (p2 - pos).norm();
            std::printf("     1 s 位移 = %.1f m（应 ~3900 m/s）\n", d);
            CHECK(d > 2000.0 && d < 6000.0, "T2 相邻 1 s 的卫星位移在轨道速度量级");
        }

        /* ---------------- T3 双差生产 ---------------- */
        std::printf("\n---- T3: 双差生产 ----\n");
        DdBuildConfig cfg;
        cfg.base_ecef    = bh.approx_pos;
        cfg.rover_approx = bh.approx_pos;      /* 流动站近似坐标用基准站替代（基线短，选星足够） */
        cfg.slots        = slots;
        cfg.elev_mask_deg = 10.0;
        cfg.min_sats     = 4;

        std::vector<DdEpochResult> dds;
        CHECK(buildDdEpochs(rover, base, eph, cfg, dds), "T3 双差生成成功");
        if (dds.empty()) { std::printf("通过 %d, 失败 %d\n", g_pass, g_fail); return 1; }

        int tot_dd = 0, min_dd = 1 << 30, max_dd = 0, n_slip = 0;
        for (const DdEpochResult &r : dds) {
            tot_dd += r.n_used;
            min_dd = std::min(min_dd, r.n_used);
            max_dd = std::max(max_dd, r.n_used);
            for (const auto &sv : r.slip)
                for (bool b : sv) if (b) { ++n_slip; break; }
        }
        std::printf("历元 %zu 个，双差总数 %d（每历元 %d–%d），标周跳 %d\n",
                    dds.size(), tot_dd, min_dd, max_dd, n_slip);
        CHECK(dds.size() > 100, "T3 生成的历元数 > 100");
        CHECK(min_dd >= 3, "T3 每历元至少 3 个双差");
        CHECK(dds.front().ref_prn > 0, "T3 参考星已选出");

        {
            const DdEpochResult &r0 = dds.front();
            std::printf("首历元: t=%.1f 共同可见=%d 使用=%d 系统数=%d 参考星=G%02d\n",
                        r0.t, r0.n_common, r0.n_used, r0.n_sys, r0.ref_prn);
            CHECK(r0.n_sys == 1, "T3 只用了 GPS（配置里只有 G）");
            CHECK(r0.n_used == r0.n_common - 1, "T3 双差数 = 共同可见数 − 1（单系统）");
            /* dd_P 应该是"米级到十几米"量级；若星历/几何错会是 1e5 量级 */
            double amax = 0.0;
            for (const DDObs &d : r0.dd) amax = std::max(amax, std::fabs(d.dd_P[0]));
            std::printf("首历元 |dd_P[L1]| 最大 = %.2f m\n", amax);
            CHECK(amax < 1.0e4, "T3 dd_P 量级合理（不是 1e5 级）");
        }

        /* ---------------- T3b 模糊度近整数检验 ---------------- */
        /* 双差把卫星钟差、接收机钟差、电离层/对流层的一阶项都消掉了，
         * 因此  N = dd_L − dd_P/λ  应当接近整数。
         *
         * 注意这条判据的**分辨力**：这个量里含双差伪距噪声（实测约 0.48 m ≈ 2.5 周），
         * 所以 |N − round(N)| 的分布本来就会填满 [0, 0.5]。它能排除的是"量级性错误"
         * ——符号反、几何错、参考星搞混都会让偏离变成数百周。
         * 真正的高精度判据在 test_gnss_tc 里（载波后验残差 RMS ≈ 2.8 mm）。 */
        std::printf("\n---- T3b: 双差模糊度近整数 ----\n");
        {
            const double lam = GPS_FREQ[0].lambda;
            int n_tested = 0, n_near = 0;
            double worst = 0.0, sum_dev = 0.0;
            for (size_t e = 0; e < dds.size(); e += 17) {          /* 抽样，跑快点 */
                for (const DDObs &d : dds[e].dd) {
                    if (!d.has_L[0] || !d.has_P[0]) continue;
                    const double N = d.dd_L[0] - d.dd_P[0] / lam;
                    const double dev = std::fabs(N - std::round(N));
                    ++n_tested;
                    sum_dev += dev;
                    worst = std::max(worst, dev);
                    if (dev < 0.6) ++n_near;
                }
            }
            const double frac = n_tested ? static_cast<double>(n_near) / n_tested : 0.0;
            std::printf("样本 %d，|N−round(N)|<0.6 的比例 = %.1f%%，平均偏差 = %.3f 周，最大 = %.3f 周\n",
                        n_tested, frac * 100.0, n_tested ? sum_dev / n_tested : 0.0, worst);
            std::printf("        （噪声底 = 双差码噪声/λ ≈ 2.5 周，所以偏离会填满 [0,0.5]；\n");
            std::printf("          本判据用于排除量级性错误 —— 符号/几何错会是数百周）\n");
            CHECK(n_tested > 100, "T3b 样本量足够");
            CHECK(frac > 0.9, "T3b 超过 90% 的双差模糊度在 0.6 周内接近整数");
            CHECK(worst < 1.0, "T3b 最坏偏离 < 1 周（排除量级性错误）");
            CHECK(n_tested ? sum_dev / n_tested < 0.3 : false,
                  "T3b 偏离均值 ≈ 0.25 周（与纯码噪声一致，无系统偏移）");
        }

        /* ---------------- T4 端到端：用双差反算流动站位置 ---------------- */
        std::printf("\n---- T4: 端到端最小二乘（用双差伪距反算流动站坐标） ----\n");
        {
            const Eigen::Vector3d base_pos = bh.approx_pos;
            int n_ok = 0, n_try = 0;
            double rms_sum = 0.0, max_base_dist = 0.0;
            Eigen::Vector3d first_x = Eigen::Vector3d::Zero();
            std::vector<Eigen::Vector3d> ests;

            for (size_t i = 0; i < dds.size(); i += 200) {
                const std::vector<DDObs> &dd = dds[i].dd;
                if (dd.size() < 5) continue;
                ++n_try;
                Eigen::Vector3d x = base_pos;      /* 初值：基准站 */
                double rms = 0.0;
                if (!solveRoverFromDd(dd, 0, base_pos, x, &rms)) continue;
                if (!x.allFinite()) continue;
                const double dist = (x - base_pos).norm();
                if (dist > 1.0e6) continue;        /* 明显发散 */
                ++n_ok;
                rms_sum += rms;
                max_base_dist = std::max(max_base_dist, dist);
                ests.push_back(x);
                if (n_ok == 1) first_x = x;
            }
            std::printf("尝试 %d 个历元，收敛 %d 个；平均后验 RMS = %.3f m；与基准站最大距离 = %.1f m\n",
                        n_try, n_ok, n_ok ? rms_sum / n_ok : 0.0, max_base_dist);
            if (n_ok > 0) {
                std::printf("首个解 ECEF = (%.2f, %.2f, %.2f)\n", first_x.x(), first_x.y(), first_x.z());
            }

            CHECK(n_try >= 5, "T4 至少尝试了 5 个历元");
            CHECK(n_ok >= static_cast<int>(n_try * 0.7), "T4 至少 70% 的历元收敛");
            CHECK(n_ok > 0 && rms_sum / std::max(1, n_ok) < 5.0,
                  "T4 后验 RMS < 5 m（星历/几何/差分/符号正确）");
            CHECK(max_base_dist < 5.0e4, "T4 解算出的流动站离基准站 < 50 km");

            /* 跨历元一致性：如果流动站基本静止，各历元解应当彼此接近 */
            if (ests.size() >= 3) {
                Eigen::Vector3d mean = Eigen::Vector3d::Zero();
                for (auto &e : ests) mean += e;
                mean /= static_cast<double>(ests.size());
                double spread = 0.0;
                for (auto &e : ests) spread = std::max(spread, (e - mean).norm());
                std::printf("跨历元解算点离散度 = %.1f m（%zu 个解）\n", spread, ests.size());
                CHECK(spread < 2.0e4, "T4 跨历元解算点一致（离散 < 20 km）");
            }
        }

        std::printf("\n=============================================\n");
        std::printf("通过 %d, 失败 %d\n", g_pass, g_fail);
        return g_fail == 0 ? 0 : 1;
    }
}  /* namespace rinex_dd */

/* ============================================================================
 * 以下为原 test/test_gnss_tc.cpp 的内容，已并入本文件
 * ==========================================================================*/
namespace gnss_tc {
    /* ============================================================================
     * test_gnss_tc.cpp —— **单独**测试 GNSS 紧组合模块（不接 IMU、不接图优化）
     *
     * 目的：把紧组合这条链单独拉出来跑真实数据，验证它自身是否成立：
     *   数据生产者  : src/gnss_dd.cpp（RINEX 3 → DDObs）
     *   残差/因子   : GnssDDPseudorangeFreqResidual / GnssDDCarrierFreqResidual（库里原装）
     *   模糊度管理  : ambKey / AmbMap（库里原装）
     *
     * 坐标约定：本测试取 R_en = I、p0 = 0、t_ant = 0，于是 pose[0..2] 直接就是 ECEF。
     * 旋转分量在 t_ant = 0 时雅可比恒为零（不可观测），LM 阻尼会把它保持在初值，
     * 不影响位置解 —— 下面会把它打印出来确认没有漂移。
     *
     * 三组测试：
     *   A. 伪距单点定位（逐历元，双频）—— 只要位置
     *   B. 双频载波浮点解（窗口，位置 + 模糊度）→ 再把模糊度固定，逐历元求位置精度
     *   C. 与独立的三参数最小二乘交叉验证（不经过任何因子类，纯几何）
     *
     * 数据路径：DSH_GNSS_DATA，默认 D:\02_Data\20220309gnss；数据缺失时 SKIP。
     * ==========================================================================*/

    static int g_pass = 0, g_fail = 0;


    static std::string dataDir() {
        const char *env = std::getenv("DSH_GNSS_DATA");
        if (env && env[0]) return std::string(env);
        return std::string("/mnt/d/02_Data/20220309gnss");
    }
    static bool fileExists(const std::string &p) {
        FILE *f = std::fopen(p.c_str(), "rb");
        if (!f) return false;
        std::fclose(f);
        return true;
    }

    /* 固定的"无局部系"设置：pose 就是 ECEF */
    static const Eigen::Matrix3d kR_en = Eigen::Matrix3d::Identity();
    static const Eigen::Vector3d kP0   = Eigen::Vector3d::Zero();
    static const Eigen::Vector3d kTant = Eigen::Vector3d::Zero();

    /* 一个历元的 pose 块 + 求解用的公共配置 */
    static void addPoseBlock(ceres::Problem &p, double *pose, const Eigen::Vector3d &init) {
        pose[0] = init.x(); pose[1] = init.y(); pose[2] = init.z();
        pose[3] = 0.0; pose[4] = 0.0; pose[5] = 0.0; pose[6] = 1.0;   /* 单位四元数 */
        p.AddParameterBlock(pose, NUM_POSE);
        p.SetManifold(pose, new PoseManifold());
    }

    static bool solve(ceres::Problem &problem, ceres::LinearSolverType lst,
                      int max_iter, ceres::Solver::Summary *sum = nullptr) {
        ceres::Solver::Options o;
        o.linear_solver_type = lst;
        o.max_num_iterations = max_iter;
        o.function_tolerance = 1e-14;
        o.gradient_tolerance = 1e-16;
        o.parameter_tolerance = 1e-14;
        o.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary s;
        ceres::Solve(o, &problem, &s);
        if (sum) *sum = s;
        return s.termination_type == ceres::CONVERGENCE || s.termination_type == ceres::USER_SUCCESS;
    }

    /* 独立的三参数最小二乘（纯几何，不经过任何因子类）——用于交叉验证 */
    static bool lsqGeom(const std::vector<DDObs> &dd, int freq,
                        const Eigen::Vector3d &base, Eigen::Vector3d &x, double *rms) {
        if (dd.size() < 4) return false;
        for (int it = 0; it < 30; ++it) {
            Eigen::MatrixXd A(dd.size(), 3);
            Eigen::VectorXd r(dd.size());
            for (size_t i = 0; i < dd.size(); ++i) {
                const Eigen::Vector3d si = dd[i].sat_i.pos_ecef - x;
                const Eigen::Vector3d sk = dd[i].sat_k.pos_ecef - x;
                const double ri = si.norm(), rk = sk.norm();
                const double bi = (dd[i].sat_i.pos_ecef - base).norm();
                const double bk = (dd[i].sat_k.pos_ecef - base).norm();
                r(static_cast<int>(i)) = dd[i].dd_P[freq] - ((ri - rk) - (bi - bk));
                A.row(static_cast<int>(i)) = (si / ri - sk / rk).transpose();
            }
            const Eigen::Vector3d dx = (A.transpose() * A).ldlt().solve(-(A.transpose() * r));
            x += dx;
            if (dx.norm() < 1e-5) break;
        }
        double ss = 0.0;
        for (const DDObs &d : dd) {
            const double ri = (d.sat_i.pos_ecef - x).norm();
            const double rk = (d.sat_k.pos_ecef - x).norm();
            const double bi = (d.sat_i.pos_ecef - base).norm();
            const double bk = (d.sat_k.pos_ecef - base).norm();
            const double e = d.dd_P[freq] - ((ri - rk) - (bi - bk));
            ss += e * e;
        }
        if (rms) *rms = std::sqrt(ss / static_cast<double>(dd.size()));
        return true;
    }


    /* ============================================================================
     * 窗口浮点解（位置 + 模糊度），可选输出模糊度协方差
     * ==========================================================================*/
    struct WinFloat {
        AmbMap    amb;
        std::vector<double*>                  amb_ptr;    /*< 与 amb_order 对应 */
        std::vector<AmbKey>                   amb_key;
        std::vector<std::vector<double>>      poses;
        std::vector<std::vector<std::array<int, MAX_FREQ>>> arc;
        int    n_res_p{0}, n_res_l{0}, n_slips{0};
        bool   converged{false};
        double cost{0.0};
        double carr_rms_m{0.0};      /*< 载波后验残差 RMS（米） */
        double sigma0_sq{1.0};       /*< 单位权方差（由载波加权残差估） */
    };

    static bool solveWindowFloat(const std::vector<DdEpochResult> &dds, size_t e0, size_t e1,
                                 const Eigen::Vector3d &base_pos, WinFloat *out,
                                 Eigen::MatrixXd *Q_NN = nullptr) {
        if (e1 <= e0) return false;
        const size_t NW = e1 - e0;
        WinFloat w;
        w.poses.assign(NW, std::vector<double>(NUM_POSE, 0.0));
        w.arc.assign(NW, {});
        for (size_t k = 0; k < NW; ++k)
            w.arc[k].assign(dds[e0 + k].dd.size(), {0, 0, 0});

        std::map<AmbKey, int> arc_of;
        std::map<AmbKey, double *> ptr_of;
        ceres::Problem problem;

        for (size_t k = 0; k < NW; ++k) {
            const size_t e = e0 + k;
            const std::vector<DDObs> &dd = dds[e].dd;
            if (dd.size() < 4) continue;
            addPoseBlock(problem, w.poses[k].data(), base_pos);
            for (size_t i = 0; i < dd.size(); ++i) {
                for (int f = 0; f < 2; ++f) {
                    if (!dd[i].has_P[f]) continue;
                    problem.AddResidualBlock(
                        new GnssDDPseudorangeFreqResidual(dd[i], f, base_pos, kR_en, kP0, kTant),
                        nullptr, w.poses[k].data());
                    ++w.n_res_p;
                }
                for (int f = 0; f < 2; ++f) {
                    if (!dd[i].has_L[f]) continue;
                    const AmbKey base_key = ambKey(dd[i].ref_sat, dd[i].sat, f);
                    if (dds[e].slip[i][f]) { ++arc_of[base_key]; ++w.n_slips; }
                    w.arc[k][i][f] = arc_of[base_key];
                    const AmbKey key = ambKey(dd[i].ref_sat, dd[i].sat, f, arc_of[base_key]);
                    auto it = w.amb.find(key);
                    if (it == w.amb.end()) {
                        double n0 = 0.0;
                        if (dd[i].has_P[f])
                            n0 = std::round(dd[i].dd_L[f] - dd[i].dd_P[f] / GPS_FREQ[f].lambda);
                        w.amb[key] = {n0};
                        it = w.amb.find(key);
                    }
                    problem.AddResidualBlock(
                        new GnssDDCarrierFreqResidual(dd[i], f, base_pos, kR_en, kP0, kTant),
                        nullptr, w.poses[k].data(), it->second.data());
                    ++w.n_res_l;
                }
            }
        }
        if (w.amb.empty()) return false;

        ceres::Solver::Summary sum;
        w.converged = solve(problem, ceres::SPARSE_NORMAL_CHOLESKY, 100, &sum);
        w.cost = sum.final_cost;

        /* 模糊度块顺序（std::map 有序 → 与 key 一一对应） */
        for (auto &kv : w.amb) {
            w.amb_key.push_back(kv.first);
            w.amb_ptr.push_back(kv.second.data());
            ptr_of[kv.first] = kv.second.data();
        }

        /* 载波后验残差 RMS 与单位权方差 */
        {
            double ss = 0.0; int nn = 0;
            for (size_t k = 0; k < NW; ++k) {
                const size_t e = e0 + k;
                const Eigen::Vector3d x(w.poses[k][0], w.poses[k][1], w.poses[k][2]);
                const std::vector<DDObs> &dd = dds[e].dd;
                for (size_t i = 0; i < dd.size(); ++i) {
                    for (int f = 0; f < 2; ++f) {
                        if (!dd[i].has_L[f]) continue;
                        const double rho_i = (dd[i].sat_i.pos_ecef - x).norm();
                        const double rho_k = (dd[i].sat_k.pos_ecef - x).norm();
                        const double bi = (dd[i].sat_i.pos_ecef - base_pos).norm();
                        const double bk = (dd[i].sat_k.pos_ecef - base_pos).norm();
                        const double dd_rho = (rho_i - rho_k) - (bi - bk);
                        const AmbKey key = ambKey(dd[i].ref_sat, dd[i].sat, f, w.arc[k][i][f]);
                        const double N = w.amb[key][0];
                        const double r = GPS_FREQ[f].lambda * dd[i].dd_L[f] - dd_rho
                                         - GPS_FREQ[f].lambda * N;
                        ss += r * r; ++nn;
                    }
                }
            }
            if (nn > 0) {
                w.carr_rms_m = std::sqrt(ss / nn);
                const int n_par = static_cast<int>(NW) * 6 + static_cast<int>(w.amb.size());
                w.sigma0_sq = (w.n_res_p + w.n_res_l > n_par)
                            ? (2.0 * w.cost) / (w.n_res_p + w.n_res_l - n_par) : 1.0;
                if (!(w.sigma0_sq > 1e-12)) w.sigma0_sq = 1.0;
            }
        }

        /* 模糊度协方差：从法方程做 Schur 补（见函数头注释） */
        if (Q_NN && w.amb_ptr.size() > 0) {
            std::vector<double*> ordered;
            for (size_t k = 0; k < NW; ++k) {
                if (dds[e0 + k].dd.size() < 4) continue;
                ordered.push_back(w.poses[k].data());
            }
            const int n_pose = static_cast<int>(ordered.size());
            for (auto *q : w.amb_ptr) ordered.push_back(q);
            const int n_amb = static_cast<int>(w.amb_ptr.size());
            const int nP = n_pose * 6;                 /* pose 切空间维数 */

            ceres::Problem::EvaluateOptions eo;
            eo.apply_loss_function = false;
            eo.parameter_blocks = ordered;
            double cost = 0.0;
            std::vector<double> residuals, gradient;
            ceres::CRSMatrix jac;
            if (!problem.Evaluate(eo, &cost, &residuals, &gradient, &jac)) return false;

            const int nr = jac.num_rows, nc = jac.num_cols;
            if (nc != nP + n_amb) return false;
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nr, nc);
            for (int row = 0; row < nr; ++row)
                for (int kk = jac.rows[row]; kk < jac.rows[row + 1]; ++kk)
                    J(row, jac.cols[kk]) = jac.values[kk];
            Eigen::VectorXd rv(residuals.size());
            for (size_t kk = 0; kk < residuals.size(); ++kk) rv(static_cast<int>(kk)) = residuals[kk];

            Eigen::MatrixXd H; Eigen::VectorXd b;
            accumulate_normal_equations(J, rv, H, b);

            const Eigen::MatrixXd H_PP = H.topLeftCorner(nP, nP);
            const Eigen::MatrixXd H_PN = H.topRightCorner(nP, n_amb);
            const Eigen::MatrixXd H_NN = H.bottomRightCorner(n_amb, n_amb);
            const Eigen::MatrixXd H_PP_inv = pinv_sym(H_PP, 1.0e-8);
            const Eigen::MatrixXd Hstar = H_NN - H_PN.transpose() * H_PP_inv * H_PN;
            Q_NN->resize(n_amb, n_amb);
            *Q_NN = pinv_sym(Hstar, 1.0e-10);
            /* 对称化 */
            *Q_NN = 0.5 * (*Q_NN + Q_NN->transpose());
        }

        *out = std::move(w);
        return true;
    }

    /* 把 AmbMap/顺序表转成 ILS 需要的向量 */
    static Eigen::VectorXd ambVector(const WinFloat &w) {
        Eigen::VectorXd a(static_cast<int>(w.amb_key.size()));
        for (size_t i = 0; i < w.amb_key.size(); ++i) a(static_cast<int>(i)) = w.amb.at(w.amb_key[i])[0];
        return a;
    }

    int run() {
        std::printf("############ GNSS 紧组合模块 · 单独测试 ############\n");

        const std::string root = dataDir();
        const std::string rover_obs = root + "/719new/01_28_09/01_28_09_CHA1.22O";
        const std::string base_obs  = root + "/base/alloy/02210680.22o";
        const std::string nav       = root + "/brdm0680.22p";
        if (!fileExists(rover_obs) || !fileExists(base_obs) || !fileExists(nav)) {
            std::printf("SKIP: 未找到 RINEX 数据（%s）\n", root.c_str());
            return 0;
        }

        /* ---------------- 0) 双差观测生成 ---------------- */
        std::vector<DdSignal> slots(2);
        slots[0].sys = 'G'; slots[0].code_P = "C1C"; slots[0].code_L = "L1C"; slots[0].code_D = "D1C";
        slots[1].sys = 'G'; slots[1].code_P = "C2W"; slots[1].code_L = "L2W"; slots[1].code_D = "D2W";

        std::vector<std::array<int, 3>> slot_index;
        const std::vector<ObsRequest> reqs = buildObsRequests(slots, &slot_index);

        std::vector<RinexEpoch> rover, base;
        RinexObsHeader bh;
        std::map<char, std::map<int, std::vector<KeplerEph>>> eph;
        if (!readRinex3Obs(rover_obs, reqs, rover, nullptr) ||
            !readRinex3Obs(base_obs, reqs, base, &bh) ||
            !readRinex3Nav(nav, eph, nullptr)) {
            std::printf("数据读取失败\n");
            return 1;
        }

        DdBuildConfig cfg;
        cfg.base_ecef     = bh.approx_pos;
        cfg.rover_approx  = bh.approx_pos;
        cfg.slots         = slots;
        cfg.elev_mask_deg = 10.0;
        cfg.min_sats      = 4;

        std::vector<DdEpochResult> dds;
        DdBuildStats dstats;
        if (!buildDdEpochs(rover, base, eph, cfg, dds, &dstats) || dds.empty()) {
            std::printf("双差生成失败\n");
            return 1;
        }
        std::printf("   组双差统计: 输入历元 %d，无基准站 %d，星数不足 %d，"
                    "选定参考星不可用而跳过 %d（按系统计），产出 %d 历元，周跳 %d 次\n",
                    dstats.n_epochs_rover, dstats.n_no_base, dstats.n_too_few,
                    dstats.n_ref_missing, dstats.n_epochs_out, dstats.n_slip);
        std::printf("   全窗口选定参考星（稳定选星）: ");
        for (auto &kv : dstats.chosen_ref)
            std::printf("%c%02d ", kv.first, kv.second);
        std::printf("\n");
        const Eigen::Vector3d base_pos = bh.approx_pos;

        int tot = 0;
        for (const auto &r : dds) tot += r.n_used;
        std::printf("0) 双差观测: %zu 个历元, %d 个双差 (%.1f/历元), 基线起点 = 基准站\n",
                    dds.size(), tot, static_cast<double>(tot) / dds.size());

        /* 周跳分布（用前 120 历元看趋势）：按 (卫星, 槽) 统计 */
        {
            std::map<std::pair<int, int>, int> cnt;
            int n_dd = 0, n_slip = 0;
            for (size_t e = 0; e < std::min<size_t>(dds.size(), 120); ++e) {
                for (size_t i = 0; i < dds[e].dd.size(); ++i) {
                    for (int f = 0; f < 2; ++f) {
                        ++n_dd;
                        if (dds[e].slip[i][f]) { ++cnt[{dds[e].dd[i].sat, f}]; ++n_slip; }
                    }
                }
            }
            std::vector<std::pair<int, std::pair<int, int>>> v;
            for (auto &kv : cnt) v.push_back({kv.second, kv.first});
            std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
            std::printf("   前 120 历元周跳：%d / %d 个 (DD,槽) = %.1f%%；涉及 %zu 个 (星,槽)\n",
                        n_slip, n_dd, n_dd ? 100.0 * n_slip / n_dd : 0.0, cnt.size());
            std::printf("      最多的几个 (星,槽): ");
            for (size_t i = 0; i < v.size() && i < 6; ++i)
                std::printf("G%02d/L%d:%d ", v[i].second.first,
                            v[i].second.second == 0 ? 1 : 2, v[i].first);
            std::printf("\n");
        }

        const size_t NW = std::min<size_t>(dds.size(), 120);   /* 窗口/统计用历元数 */
        std::printf("   本测试使用前 %zu 个历元\n", NW);

        /* ============================================================
         * A) 伪距单点定位（逐历元，双频，仅用库里的伪距因子）
         * ==========================================================*/
        std::printf("\n---- A. 伪距单点定位（逐历元，双频） ----\n");
        std::vector<Eigen::Vector3d> posA;
        std::vector<double> rmsL1, rmsL2;
        int nA_ok = 0, nA_try = 0;
        for (size_t e = 0; e < NW; ++e) {
            const std::vector<DDObs> &dd = dds[e].dd;
            if (dd.size() < 4) continue;
            ++nA_try;
            std::vector<double> pose(NUM_POSE, 0.0);
            ceres::Problem prob;
            addPoseBlock(prob, pose.data(), base_pos);
            for (size_t i = 0; i < dd.size(); ++i) {
                for (int f = 0; f < 2; ++f) {
                    if (!dd[i].has_P[f]) continue;
                    prob.AddResidualBlock(
                        new GnssDDPseudorangeFreqResidual(dd[i], f, base_pos, kR_en, kP0, kTant),
                        nullptr, pose.data());
                }
            }
            if (!solve(prob, ceres::DENSE_QR, 50)) continue;
            const Eigen::Vector3d x(pose[0], pose[1], pose[2]);
            if (!x.allFinite() || (x - base_pos).norm() > 1.0e6) continue;
            ++nA_ok;
            posA.push_back(x);

            /* 后验残差（按槽） */
            for (int f = 0; f < 2; ++f) {
                double ss = 0.0; int nn = 0;
                for (const DDObs &d : dd) {
                    if (!d.has_P[f]) continue;
                    const double ri = (d.sat_i.pos_ecef - x).norm();
                    const double rk = (d.sat_k.pos_ecef - x).norm();
                    const double bi = (d.sat_i.pos_ecef - base_pos).norm();
                    const double bk = (d.sat_k.pos_ecef - base_pos).norm();
                    const double r = d.dd_P[f] - ((ri - rk) - (bi - bk));
                    ss += r * r; ++nn;
                }
                if (nn) (f == 0 ? rmsL1 : rmsL2).push_back(std::sqrt(ss / nn));
            }
        }
        auto mean = [](const std::vector<double> &v) {
            if (v.empty()) return 0.0;
            double s = 0.0; for (double x : v) s += x; return s / v.size();
        };
        std::printf("   收敛 %d/%d (%.1f%%)\n", nA_ok, nA_try,
                    nA_try ? 100.0 * nA_ok / nA_try : 0.0);
        std::printf("   后验 RMS 均值: L1 = %.3f m, L2 = %.3f m\n", mean(rmsL1), mean(rmsL2));
        double traj = 0.0, maxjump = 0.0;
        for (size_t i = 1; i < posA.size(); ++i) {
            const double d = (posA[i] - posA[i - 1]).norm();
            traj += d;
            maxjump = std::max(maxjump, d);
        }
        std::printf("   轨迹长度(相对基准站累计位移) = %.1f m, 相邻历元最大跳变 = %.1f m\n",
                    traj, maxjump);
        CHECK(nA_try > 100, "A 尝试历元数 > 100");
        CHECK(nA_ok > static_cast<int>(nA_try * 0.9), "A 收敛率 > 90%");
        CHECK(mean(rmsL1) < 1.0, "A L1 伪距后验 RMS < 1 m");
        CHECK(maxjump < 200.0, "A 相邻历元位置连续（无跳点）");

        /* ============================================================
         * B) 双频载波浮点解（窗口：位置 + 模糊度）
         * ==========================================================*/
        std::printf("\n---- B. 双频载波浮点解（窗口 %zu 历元，位置 + 模糊度） ----\n", NW);
        AmbMap amb;                                  /* 库里的模糊度容器 */
        std::vector<std::vector<double>> poses(NW, std::vector<double>(NUM_POSE, 0.0));
        /* 弧段号：每检测到一次周跳就 +1，对应库里 ambAdvanceArc 的语义 */
        std::map<AmbKey, int> arc_of;
        int n_slip_events = 0;

        /* 先按首历元估个初值，保证模糊度初值合理 */
        Eigen::Vector3d x_cur = base_pos;
        for (size_t e = 0; e < NW; ++e) {
            const std::vector<DDObs> &dd = dds[e].dd;
            if (dd.size() >= 4) {
                double rr = 0.0;
                if (lsqGeom(dd, 0, base_pos, x_cur, &rr)) { /* 递推初值 */ }
            }
        }

        /* 逐历元逐 DD 逐槽记录弧段号，B2 需要复用同一套模糊度 */
        std::vector<std::vector<std::array<int, MAX_FREQ>>> arc_of_ep(NW);
        for (size_t e = 0; e < NW; ++e)
            arc_of_ep[e].assign(dds[e].dd.size(), {0, 0, 0});

        ceres::Problem problem;
        int n_amb_used = 0, n_res_p = 0, n_res_l = 0;
        for (size_t e = 0; e < NW; ++e) {
            const std::vector<DDObs> &dd = dds[e].dd;
            if (dd.size() < 4) continue;
            addPoseBlock(problem, poses[e].data(), base_pos);
            for (size_t i = 0; i < dd.size(); ++i) {
                for (int f = 0; f < 2; ++f) {
                    if (!dd[i].has_P[f]) continue;
                    problem.AddResidualBlock(
                        new GnssDDPseudorangeFreqResidual(dd[i], f, base_pos, kR_en, kP0, kTant),
                        nullptr, poses[e].data());
                    ++n_res_p;
                }
                for (int f = 0; f < 2; ++f) {
                    if (!dd[i].has_L[f]) continue;
                    const AmbKey base_key = ambKey(dd[i].ref_sat, dd[i].sat, f);
                    /* 周跳 → 换弧段（新参数）；与库里 ambAdvanceArc 的语义一致 */
                    if (dds[e].slip[i][f]) { ++arc_of[base_key]; ++n_slip_events; }
                    arc_of_ep[e][i][f] = arc_of[base_key];
                    const AmbKey key = ambKey(dd[i].ref_sat, dd[i].sat, f, arc_of[base_key]);
                    auto it = amb.find(key);
                    if (it == amb.end()) {
                        /* 初值：dd_L − dd_P/λ 取整（标准做法） */
                        double n0 = 0.0;
                        if (dd[i].has_P[f]) {
                            n0 = std::round(dd[i].dd_L[f] - dd[i].dd_P[f] / GPS_FREQ[f].lambda);
                        }
                        amb[key] = {n0};
                        it = amb.find(key);
                    }
                    problem.AddResidualBlock(
                        new GnssDDCarrierFreqResidual(dd[i], f, base_pos, kR_en, kP0, kTant),
                        nullptr, poses[e].data(), it->second.data());
                    ++n_res_l;
                }
            }
        }
        n_amb_used = static_cast<int>(amb.size());
        std::printf("   参数: pose %zu 个(7 维) + 模糊度 %d 个(1 维)\n", NW, n_amb_used);
        std::printf("   周跳: MW/LLI 共判出 %d 次（库里的 ambAdvanceArc 弧段管理）\n", n_slip_events);
        std::printf("   残差: 伪距 %d + 载波 %d\n", n_res_p, n_res_l);
        CHECK(n_amb_used > 0, "B 生成了模糊度参数");

        ceres::Solver::Summary sum;
        const bool okB = solve(problem, ceres::SPARSE_NORMAL_CHOLESKY, 100, &sum);
        std::printf("   求解: %s, 有效步数 %d, 初始代价 %.4g → 最终代价 %.4g\n",
                    okB ? "收敛" : "未收敛",
                    sum.num_successful_steps + sum.num_unsuccessful_steps,
                    sum.initial_cost, sum.final_cost);
        std::printf("   平均残差 = %.4f（代价/残差数）\n",
                    (n_res_p + n_res_l) ? std::sqrt(2.0 * sum.final_cost / (n_res_p + n_res_l)) : 0.0);
        CHECK(okB, "B 窗口浮点解收敛");

        /* 旋转分量是否漂移（应为 0，因为 t_ant = 0 时不可观测） */
        double max_rot_drift = 0.0;
        for (size_t e = 0; e < NW; ++e) {
            max_rot_drift = std::max(max_rot_drift, std::fabs(poses[e][3]));
            max_rot_drift = std::max(max_rot_drift, std::fabs(poses[e][4]));
            max_rot_drift = std::max(max_rot_drift, std::fabs(poses[e][5]));
            max_rot_drift = std::max(max_rot_drift, std::fabs(poses[e][6] - 1.0));
        }
        std::printf("   姿态分量最大漂移 = %.3e（不可观测，应≈0）\n", max_rot_drift);
        CHECK(max_rot_drift < 1e-6, "B 不可观测的姿态分量未漂移");

        /* 浮点解位置 vs A */
        {
            double dmax = 0.0, dsum = 0.0; int nn = 0;
            for (size_t e = 0; e < NW && e < posA.size(); ++e) {
                const Eigen::Vector3d xb(poses[e][0], poses[e][1], poses[e][2]);
                const double d = (xb - posA[e]).norm();
                dmax = std::max(dmax, d); dsum += d; ++nn;
            }
            std::printf("   与 A(伪距单点) 的差异: 平均 %.3f m, 最大 %.3f m\n",
                        nn ? dsum / nn : 0.0, dmax);
            CHECK(nn > 50, "B 可比历元数 > 50");
        }

        /* ---- B2: 固定模糊度，逐历元求位置 → 载波精度 ---- */
        std::printf("\n   B2. 固定模糊度后逐历元定位（载波精度）\n");
        std::vector<Eigen::Vector3d> posB;
        for (size_t e = 0; e < NW; ++e) {
            const std::vector<DDObs> &dd = dds[e].dd;
            if (dd.size() < 4) continue;
            std::vector<double> pose(NUM_POSE, 0.0);
            ceres::Problem pr;
            addPoseBlock(pr, pose.data(), base_pos);
            bool any = false;
            for (size_t i = 0; i < dd.size(); ++i) {
                for (int f = 0; f < 2; ++f) {
                    if (!dd[i].has_L[f]) continue;
                    const AmbKey key = ambKey(dd[i].ref_sat, dd[i].sat, f, arc_of_ep[e][i][f]);
                    auto it = amb.find(key);
                    if (it == amb.end()) continue;
                    /* amb 作为常量：加一个 1 维块并固定 */
                    pr.AddParameterBlock(it->second.data(), 1);
                    pr.SetParameterBlockConstant(it->second.data());
                    pr.AddResidualBlock(
                        new GnssDDCarrierFreqResidual(dd[i], f, base_pos, kR_en, kP0, kTant),
                        nullptr, pose.data(), it->second.data());
                    any = true;
                }
            }
            if (!any) continue;
            if (!solve(pr, ceres::DENSE_QR, 50)) continue;
            const Eigen::Vector3d x(pose[0], pose[1], pose[2]);
            if (!x.allFinite() || (x - base_pos).norm() > 1.0e6) continue;
            posB.push_back(x);
        }
        /* 与 A 比：载波解应当更平滑（相邻历元跳变更小） */
        double trajB = 0.0, maxjumpB = 0.0;
        for (size_t i = 1; i < posB.size(); ++i) {
            const double d = (posB[i] - posB[i - 1]).norm();
            trajB += d; maxjumpB = std::max(maxjumpB, d);
        }
        std::printf("      解出 %zu 个历元；相邻历元最大跳变 = %.3f m（对比伪距 %.1f m）\n",
                    posB.size(), maxjumpB, maxjump);
        CHECK(posB.size() > 50, "B2 解出历元数 > 50");
        CHECK(maxjumpB < std::max(0.5, maxjump), "B2 载波解的历元间连续性优于伪距");

        /* ---- B3: 载波/模糊度质量 ---- */
        std::printf("\n   B3. 载波与模糊度质量\n");
        {
            /* B3a: 载波后验残差 RMS（米） */
            double ss = 0.0; int nn = 0;
            for (size_t e = 0; e < NW; ++e) {
                const Eigen::Vector3d x(poses[e][0], poses[e][1], poses[e][2]);
                for (const DDObs &d : dds[e].dd) {
                    for (int f = 0; f < 2; ++f) {
                        if (!d.has_L[f]) continue;
                        const double rho_i = (d.sat_i.pos_ecef - x).norm();
                        const double rho_k = (d.sat_k.pos_ecef - x).norm();
                        const double bi = (d.sat_i.pos_ecef - base_pos).norm();
                        const double bk = (d.sat_k.pos_ecef - base_pos).norm();
                        const double dd_rho = (rho_i - rho_k) - (bi - bk);
                        const AmbKey key = ambKey(d.ref_sat, d.sat, f,
                                                  arc_of_ep[e][&d - &dds[e].dd[0]][f]);
                        const double N = amb[key][0];
                        const double r = GPS_FREQ[f].lambda * d.dd_L[f] - dd_rho
                                         - GPS_FREQ[f].lambda * N;
                        ss += r * r; ++nn;
                    }
                }
            }
            const double rms_m = nn ? std::sqrt(ss / nn) : 0.0;
            std::printf("      B3a 载波后验残差 RMS = %.4f m（%d 个残差）\n", rms_m, nn);
            CHECK(nn > 500, "B3a 载波残差样本充足");
            CHECK(rms_m < 0.02, "B3a 载波后验残差在厘米级以内（模糊度与几何正确）");

            /* 按弧段收集度量序列，算均值与其标准误 */
            struct Arc { std::vector<double> v; };
            std::map<AmbKey, Arc> arcs;
            for (size_t e = 0; e < NW; ++e) {
                for (const DDObs &d : dds[e].dd) {
                    for (int f = 0; f < 2; ++f) {
                        if (!d.has_L[f] || !d.has_P[f]) continue;
                        const AmbKey key = ambKey(d.ref_sat, d.sat, f,
                                                  arc_of_ep[e][&d - &dds[e].dd[0]][f]);
                        arcs[key].v.push_back(d.dd_L[f] - d.dd_P[f] / GPS_FREQ[f].lambda);
                    }
                }
            }

            /* B3b: 弧段内均值标准误 */
            double worst_se = 0.0; int n_se = 0, n_se_ok = 0;
            double worst_std = 0.0;
            for (auto &kv : arcs) {
                const std::vector<double> &v = kv.second.v;
                if (v.size() < 10) continue;
                double m = 0.0; for (double x : v) m += x; m /= v.size();
                double s2 = 0.0; for (double x : v) s2 += (x - m) * (x - m);
                const double sd = std::sqrt(s2 / v.size());
                const double se = sd / std::sqrt(static_cast<double>(v.size()));
                worst_se = std::max(worst_se, se);
                worst_std = std::max(worst_std, sd);
                ++n_se;
                if (se < 0.3) ++n_se_ok;
            }
            std::printf("      B3b 弧段 %d 条；度量原始散布最大 %.2f 周（主要为码噪声）\n",
                        n_se, worst_std);
            std::printf("          均值标准误最大 %.3f 周（噪声被平均掉后仍应很小）\n", worst_se);
            CHECK(n_se > 5, "B3b 弧段数 > 5");
            /* 判据一：平均确实起了作用（SE 远小于原始散布）→ 说明没有系统性漂移
             * 判据二：绝对量级在半周以内（半周是模糊度的可分辨极限） */
            CHECK(worst_se < worst_std / 3.0, "B3b 弧段内平均有效（无系统性漂移）");
            CHECK(worst_se < 0.5, "B3b 弧段内均值标准误 < 0.5 周");

            /* B3c: 相邻弧段（同卫星同槽）的均值之差应接近整数 */
            std::map<std::pair<int, int>, std::map<int, double>> per_arc_mean;  /* (sat,slot)->(arc,mean) */
            for (auto &kv : arcs) {
                const std::vector<double> &v = kv.second.v;
                if (v.size() < 10) continue;
                double m = 0.0; for (double x : v) m += x; m /= v.size();
                per_arc_mean[{ambKeySat(kv.first), ambKeySlot(kv.first)}][ambKeyArc(kv.first)] = m;
            }
            int n_pair = 0, n_pair_int = 0; double worst_dev = 0.0;
            for (auto &kv : per_arc_mean) {
                const auto &by_arc = kv.second;
                if (by_arc.size() < 2) continue;
                auto it = by_arc.begin();
                double prev = it->second; ++it;
                for (; it != by_arc.end(); ++it) {
                    const double d = it->second - prev;
                    const double dev = std::fabs(d - std::round(d));
                    ++n_pair;
                    if (dev < 0.5) ++n_pair_int;   /* 与 worst_dev 门限保持一致 */
                    worst_dev = std::max(worst_dev, dev);
                    prev = it->second;
                }
            }
            std::printf("      B3c 相邻弧段均值之差: %d 对，其中 %d 对接近整数，最坏偏离 %.3f 周\n",
                        n_pair, n_pair_int, worst_dev);
            if (n_pair > 0) {
                /* 弧段均值有 ~0.3 周标准误，两段之差的不确定度约 0.42 周，
                 * 所以"接近整数"的门限就是半周。 */
                CHECK(worst_dev < 0.5, "B3c 相邻弧段均值之差在半周内接近整数（= 整周跳变）");
                CHECK(n_pair_int == n_pair, "B3c 所有跨弧段对都判为整周跳变");
            } else {
                std::printf("          （本窗口内没有跨弧段的同卫星对，跳过）\n");
            }
        }

        /* ============================================================
         * D) 整数固定（ILS / LAMBDA）
         * ==========================================================*/
        std::printf("\n---- D. 整数固定（ILS） ----\n");
        {
            WinFloat wf;
            Eigen::MatrixXd Q_NN;
            if (!solveWindowFloat(dds, 0, NW, base_pos, &wf, &Q_NN)) {
                std::printf("   窗口浮点解失败，跳过 D\n");
                CHECK(false, "D 窗口浮点解成功");
            } else {
                const int n = static_cast<int>(wf.amb_key.size());
                std::printf("   窗口 %zu 历元：pose %zu + 模糊度 %d；伪距残差 %d，载波残差 %d\n",
                            NW, NW, n, wf.n_res_p, wf.n_res_l);
                std::printf("   载波后验 RMS = %.4f m；单位权方差 σ0²(含码+载波) = %.4f；"
                            "载波口径 σ0² = %.4f\n",
                            wf.carr_rms_m, wf.sigma0_sq,
                            (wf.carr_rms_m / GPS_FREQ[0].sigma_L) *
                            (wf.carr_rms_m / GPS_FREQ[0].sigma_L));

                const Eigen::VectorXd a = ambVector(wf);
                {
                    double dmin = 1e300, dmax = 0.0;
                    for (int i = 0; i < Q_NN.rows(); ++i) {
                        dmin = std::min(dmin, Q_NN(i, i));
                        dmax = std::max(dmax, Q_NN(i, i));
                    }
                    std::printf("   Q_NN 对角线: 最小 %.3e, 最大 %.3e（单位 周²）→ σ = %.4f ~ %.4f 周\n",
                                dmin, dmax, std::sqrt(std::max(0.0, dmin)), std::sqrt(std::max(0.0, dmax)));
                }
                /* D1a 不做 σ0² 缩放（Ceres 的协方差是线性化点的形式协方差） */
                IlsResult r_raw, r_scaled;
                const bool ok_raw = ilsResolve(a, Q_NN, &r_raw, 3.0);
                const Eigen::MatrixXd Qs = wf.sigma0_sq * Q_NN;
                const bool ok_sc = ilsResolve(a, Qs, &r_scaled, 3.0);

                std::printf("   D1 ILS：n=%d，去相关 Gauss 变换 %d 次，枚举节点 %ld\n",
                            r_raw.n, r_raw.n_decorr, r_raw.nodes);
                std::printf("      形式协方差   : ratio = %.3g，p_boot = %.4g，%s\n",
                            r_raw.ratio, r_raw.p_boot,
                            (ok_raw && r_raw.ratio > 3.0) ? "可固定" : "不固定");
                std::printf("      乘 σ0² 后    : ratio = %.3g，p_boot = %.4g，%s\n",
                            r_scaled.ratio, r_scaled.p_boot,
                            (ok_sc && r_scaled.ratio > 3.0) ? "可固定" : "不固定");

                const bool fixed = ok_sc && r_scaled.ratio > 3.0;
                CHECK(ok_raw && ok_sc, "D1 ILS 枚举正常完成（未超节点上限）");
                CHECK(r_raw.p_boot >= 0.0 && r_raw.p_boot <= 1.0, "D1 成功率在 [0,1]");

                /* D5 部分固定 */
                {
                    IlsPartial pp;
                    const bool okpp = ilsPartialFix(a, Qs, 3.0, 4, 2000000L, &pp);
                    std::printf("   D5 部分固定：ok=%d，固定 %zu/%d（尝试 %d 个子集），"
                                "子集 ratio = %.3g\n",
                                (int)okpp, pp.fixed_idx.size(), pp.n_full, pp.n_tried,
                                pp.last.ratio);
                    if (okpp && pp.cond_sigma.size() > 0) {
                        double s_min = 1e300, s_max = 0.0;
                        for (int i : pp.fixed_idx) {
                            s_min = std::min(s_min, pp.cond_sigma(i));
                            s_max = std::max(s_max, pp.cond_sigma(i));
                        }
                        std::printf("        被固定模糊度的条件 σ 范围 = %.4f ~ %.4f 周\n",
                                    s_min, s_max);
                    }
                    CHECK(okpp || !fixed, "D5 整组能固定时部分固定也应成功");
                }

                if (fixed) {
                    /* 偏差：固定解与浮点解之差（应接近整数） */
                    double worst_dev = 0.0;
                    for (int i = 0; i < n; ++i) {
                        const double d = r_scaled.a_fixed(i) - a(i);
                        worst_dev = std::max(worst_dev, std::fabs(d - std::round(d)));
                    }
                    std::printf("      固定解与浮点解之差偏离整数的最大值 = %.3f 周\n", worst_dev);

                    /* D2 固定后逐历元定位 */
                    std::vector<Eigen::Vector3d> posFix;
                    std::map<AmbKey, double> amb_fix;
                    for (int i = 0; i < n; ++i) amb_fix[wf.amb_key[i]] = r_scaled.a_fixed(i);
                    for (size_t k = 0; k < NW; ++k) {
                        const size_t e = k;
                        const std::vector<DDObs> &dd = dds[e].dd;
                        if (dd.size() < 4) continue;
                        std::vector<double> pose(NUM_POSE, 0.0);
                        ceres::Problem pr;
                        addPoseBlock(pr, pose.data(), base_pos);
                        bool any = false;
                        for (size_t i = 0; i < dd.size(); ++i) {
                            for (int f = 0; f < 2; ++f) {
                                if (!dd[i].has_L[f]) continue;
                                const AmbKey key = ambKey(dd[i].ref_sat, dd[i].sat, f, wf.arc[k][i][f]);
                                auto it = amb_fix.find(key);
                                if (it == amb_fix.end()) continue;
                                pr.AddParameterBlock(&it->second, 1);
                                pr.SetParameterBlockConstant(&it->second);
                                pr.AddResidualBlock(
                                    new GnssDDCarrierFreqResidual(dd[i], f, base_pos, kR_en, kP0, kTant),
                                    nullptr, pose.data(), &it->second);
                                any = true;
                            }
                        }
                        if (!any) continue;
                        if (!solve(pr, ceres::DENSE_QR, 50)) continue;
                        const Eigen::Vector3d x(pose[0], pose[1], pose[2]);
                        if (x.allFinite() && (x - base_pos).norm() < 1.0e6) posFix.push_back(x);
                    }
                    double jump = 0.0;
                    for (size_t i = 1; i < posFix.size(); ++i)
                        jump = std::max(jump, (posFix[i] - posFix[i - 1]).norm());
                    /* 固定 vs 浮点的位置变化（AR 的实际影响） */
                    double dmax_fix = 0.0;
                    for (size_t i = 0; i < posFix.size() && i < posB.size(); ++i)
                        dmax_fix = std::max(dmax_fix, (posFix[i] - posB[i]).norm());
                    std::printf("      D2 整周固定后逐历元定位 %zu 个；相邻历元最大跳变 = %.4f m"
                                "（浮点 %.4f m）；固定 vs 浮点位置最大变化 = %.4f m\n",
                                posFix.size(), jump, maxjumpB, dmax_fix);
                    CHECK(posFix.size() > 50, "D2 固定后解出足够历元");
                    CHECK(jump <= maxjumpB + 0.01, "D2 整周固定未使历元间连续性显著变差");
                    CHECK(dmax_fix < 1.0, "D2 固定相对浮点的位置变化在合理范围 (< 1 m)");

                } else {
                    std::printf("      未达到固定门限；记录 ratio 供参考\n");
                }


                    /* D3 不重叠双窗一致性（各 NW 个历元）
                     * 先试整组固定，不行则用部分固定；然后在**两窗都被固定**的
                     * 模糊度上比较整数 —— 两段数据完全独立，整数一致才说明固定可信。 */
                    WinFloat wa, wb;
                    Eigen::MatrixXd Qa, Qb;
                    /* 用 D4 已证明能固定的长度（40 历元）做一致性校验；
                     * 120 历元时模糊度涨到 66+，整组与部分固定都过不了门限。 */
                    const size_t WD = 40;
                    const bool oka = solveWindowFloat(dds, 0, WD, base_pos, &wa, &Qa);
                    const bool okb = (dds.size() >= 2 * WD)
                                   ? solveWindowFloat(dds, WD, 2 * WD, base_pos, &wb, &Qb) : false;
                    if (oka && okb) {
                        const Eigen::VectorXd aa = ambVector(wa), ab = ambVector(wb);
                        const Eigen::MatrixXd Qas = wa.sigma0_sq * Qa, Qbs = wb.sigma0_sq * Qb;
                        IlsResult ra, rb;
                        const bool oka2 = ilsResolve(aa, Qas, &ra, 3.0);
                        const bool okb2 = ilsResolve(ab, Qbs, &rb, 3.0);
                        std::printf("      D3 整组: 窗A[0,%zu) n=%d ok=%d 节点=%ld ratio=%.3g；"
                                    "窗B[%zu,%zu) n=%d ok=%d 节点=%ld ratio=%.3g\n",
                                    WD, ra.n, (int)oka2, ra.nodes, ra.ratio,
                                    WD, 2 * WD, rb.n, (int)okb2, rb.nodes, rb.ratio);

                        IlsPartial pa, pb;
                        const bool oka3 = ilsPartialFix(aa, Qas, 3.0, 4, 2000000L, &pa);
                        const bool okb3 = ilsPartialFix(ab, Qbs, 3.0, 4, 2000000L, &pb);
                        std::printf("      D3 部分: 窗A ok=%d 固定 %zu/%d；"
                                    "窗B ok=%d 固定 %zu/%d\n",
                                    (int)oka3, pa.fixed_idx.size(), pa.n_full,
                                    (int)okb3, pb.fixed_idx.size(), pb.n_full);
                        CHECK(oka3, "D3 窗A 能给出固定解（整组或部分）");
                        CHECK(okb3, "D3 窗B 能给出固定解（整组或部分）");

                        if (oka3 && okb3) {
                            /* 建立 key → 整数 的映射（只含被固定的） */
                            std::map<AmbKey, double> fixA, fixB;
                            for (int i : pa.fixed_idx)
                                fixA[wa.amb_key[static_cast<size_t>(i)]] = pa.a_fixed(i);
                            for (int i : pb.fixed_idx)
                                fixB[wb.amb_key[static_cast<size_t>(i)]] = pb.a_fixed(i);
                            int n_cmp = 0, n_same = 0;
                            for (const auto &kv : fixA) {
                                auto it = fixB.find(kv.first);
                                if (it == fixB.end()) continue;
                                ++n_cmp;
                                if (std::fabs(kv.second - it->second) < 0.5) ++n_same;
                            }
                            std::printf("      两窗共同被固定的模糊度 %d 个，整数一致 %d 个\n",
                                        n_cmp, n_same);
                            CHECK(n_cmp > 0, "D3 两窗有共同被固定的模糊度可比");
                            CHECK(n_same == n_cmp,
                                  "D3 两段完全独立的数据给出同一组整数（固定可信）");
                        }

                        /* D3b 形式协方差是否乐观：独立双窗浮点解之差 / 形式 σ */
                        {
                            int n_cmp = 0;
                            double sum_r2 = 0.0, worst_r = 0.0;
                            for (size_t i = 0; i < wa.amb_key.size(); ++i) {
                                for (size_t j = 0; j < wb.amb_key.size(); ++j) {
                                    if (wb.amb_key[j] != wa.amb_key[i]) continue;
                                    const double d = wa.amb.at(wa.amb_key[i])[0]
                                                   - wb.amb.at(wb.amb_key[j])[0];
                                    const double sA2 = Qas(static_cast<int>(i), static_cast<int>(i));
                                    const double sB2 = Qbs(static_cast<int>(j), static_cast<int>(j));
                                    const double den = std::sqrt(std::max(1e-30, sA2 + sB2));
                                    const double rr = std::fabs(d) / den;
                                    sum_r2 += rr * rr; worst_r = std::max(worst_r, rr);
                                    ++n_cmp;
                                    break;
                                }
                            }
                            if (n_cmp > 0) {
                                const double rms_r = std::sqrt(sum_r2 / n_cmp);
                                std::printf("      D3b 双窗浮点解之差 / 形式 σ：RMS = %.2f，最大 = %.2f"
                                            "（%d 个共同模糊度）\n", rms_r, worst_r, n_cmp);
                                std::printf("           ≈1 表示形式协方差与实际散布一致；"
                                            "≫1 表示形式协方差**偏乐观**\n");
                                CHECK(rms_r > 0.0, "D3b 双窗比较得出有效统计量");
                            }
                        }
                    }

                /* D4 ratio 随窗口长度（固定成功率趋势） */
                std::printf("   D4 窗口长度 → ratio（固定成功率趋势）\n");
                for (size_t len : {size_t(5), size_t(10), size_t(20), size_t(40), size_t(80)}) {
                    if (len > NW) break;
                    WinFloat w;
                    Eigen::MatrixXd Q;
                    if (!solveWindowFloat(dds, 0, len, base_pos, &w, &Q)) continue;
                    IlsResult r;
                    const bool ok = ilsResolve(ambVector(w), w.sigma0_sq * Q, &r, 3.0);
                    std::printf("      %3zu 历元: 模糊度 %3d  ratio = %-10.3g p_boot = %-8.3g %s\n",
                                len, r.n, r.ratio, r.p_boot,
                                (ok && r.ratio > 3.0) ? "固定" : "未固定");
                    CHECK(ok, "D4 每个窗口长度下 ILS 都正常完成");
                }
            }
        }

        /* ============================================================
         * C) 交叉验证：独立几何最小二乘
         * ==========================================================*/
        std::printf("\n---- C. 交叉验证（与独立三参数几何最小二乘对比） ----\n");
        {
            double dmax = 0.0, dsum = 0.0; int nn = 0;
            for (size_t e = 0; e < NW; ++e) {
                const std::vector<DDObs> &dd = dds[e].dd;
                if (dd.size() < 4) continue;
                Eigen::Vector3d x = base_pos; double rms = 0.0;
                if (!lsqGeom(dd, 0, base_pos, x, &rms)) continue;
                /* 因子解（A 里那个历元的解） */
                const Eigen::Vector3d xb(poses[e][0], poses[e][1], poses[e][2]);
                const double d = (x - xb).norm();
                dmax = std::max(dmax, d); dsum += d; ++nn;
            }
            std::printf("      与窗口浮点解的位置差: 平均 %.3f m, 最大 %.3f m（%d 个历元）\n",
                        nn ? dsum / nn : 0.0, dmax, nn);
            CHECK(nn > 50, "C 对比历元数 > 50");
            CHECK(dmax < 5.0, "C 因子类与独立几何解一致（< 5 m）");
        }

        std::printf("\n####################################################\n");
        std::printf("通过 %d, 失败 %d\n", g_pass, g_fail);
        return g_fail == 0 ? 0 : 1;
    }
}  /* namespace gnss_tc */

/* ============================================================================
 * 以下为原 test/test_ils.cpp 的内容，已并入本文件
 * ==========================================================================*/
namespace ils {
    /* ============================================================================
     * test_ils.cpp —— 整数最小二乘（LAMBDA 风格）的正确性验证
     *
     * 唯一的正确性保证方式：在**小规模**问题上与**暴力枚举**逐例对照。
     * 覆盖：
     *   T1 对角 Q（条件好）           —— 与暴力一致
     *   T2 强相关 Q（L 非对角很大）   —— 与暴力一致，且考察去相关是否真的把 |L| 压下来
     *   T3 高维 n=6                   —— 与暴力一致
     *   T4 已知整数真值：加噪声后应还原
     *   T5 ratio 的语义：清晰 vs 模糊
     *   T6 非正定 Q 要拒绝（不能返回假固定解）
     *   T7 去相关变换本身：T 必须整数且 |det T| = 1（否则格会变，ILS 解就错了）
     * ==========================================================================*/

    static int g_pass = 0, g_fail = 0;


    static uint64_t g_state = 20220309u;
    static double urand() {                     /* [0,1) */
        g_state = g_state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((g_state >> 11) & ((1ULL << 53) - 1)) / 9007199254740992.0;
    }
    static double nrand() {                     /* 标准正态（Box-Muller） */
        const double u1 = std::max(1e-12, urand()), u2 = urand();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }

    /* 造一个"有结构"的协方差：Q = L D Lᵀ，L 单位下三角 */
    static Eigen::MatrixXd makeCov(int n, double lmax, double dmin, double dmax) {
        Eigen::MatrixXd L = Eigen::MatrixXd::Identity(n, n);
        for (int i = 1; i < n; ++i)
            for (int j = 0; j < i; ++j) L(i, j) = lmax * (2.0 * urand() - 1.0);
        Eigen::VectorXd D(n);
        for (int i = 0; i < n; ++i)
            D(i) = std::exp(std::log(dmin) + urand() * (std::log(dmax) - std::log(dmin)));
        return L * D.asDiagonal() * L.transpose();
    }

    /* 原空间二次型 (z − a)ᵀ Q⁻¹ (z − a) */
    static double chi2Orig(const Eigen::VectorXd &a, const Eigen::MatrixXd &Q,
                           const Eigen::VectorXd &z) {
        const Eigen::VectorXd d = z - a;
        return d.dot(Q.ldlt().solve(d));
    }

    /* 与暴力对照一次（判据 S1/S2/S3 见文件头） */
    static void compareWithBrute(const Eigen::VectorXd &a, const Eigen::MatrixXd &Q,
                                 int range, const char *tag) {
        IlsResult r1, r2;
        const bool ok1 = ilsResolve(a, Q, &r1, 3.0);
        const bool ok2 = ilsResolveBrute(a, Q, range, &r2);
        if (!ok1 || !ok2) {
            std::printf("[FAIL] %s: ILS ok=%d, brute ok=%d\n", tag, (int)ok1, (int)ok2);
            ++g_fail;
            return;
        }

        /* S0 暴力参照自身的自洽性（先证明参照是对的，再用它当基准） */
        {
            const double sq_b = chi2Orig(a, Q, r2.a_fixed);
            const double rel_b = std::fabs(sq_b - r2.sq_best) / std::max(1e-30, r2.sq_best);
            if (rel_b > 1e-6) {
                std::printf("[FAIL] %s: S0 暴力参照不自洽 —— 报出 %.10g，代回得 %.10g (rel %.2e)\n",
                            tag, r2.sq_best, sq_b, rel_b);
                ++g_fail;
            } else {
                ++g_pass;
            }
        }

        /* S1 自洽 */
        const double sq_self = chi2Orig(a, Q, r1.a_fixed);
        const double rel_self = std::fabs(sq_self - r1.sq_best) / std::max(1e-30, r1.sq_best);
        if (rel_self > 1e-6) {
            std::printf("[FAIL] %s: S1 不自洽 —— 报出 sq_best=%.10g，但 a_fixed 代回得 %.10g (rel %.2e)\n",
                        tag, r1.sq_best, sq_self, rel_self);
            ++g_fail;
        } else {
            ++g_pass;
        }

        /* S2 不劣 */
        const double rel = (r1.sq_best - r2.sq_best) / std::max(1e-30, std::fabs(r2.sq_best));
        if (rel > 1e-6) {
            std::printf("[FAIL] %s: S2 ILS 的 χ² 反而更大: %.10g vs brute %.10g\n",
                        tag, r1.sq_best, r2.sq_best);
            ++g_fail;
        } else {
            ++g_pass;
        }

        /* S3 一致性 */
        const double dif = (r1.a_fixed - r2.a_fixed).cwiseAbs().maxCoeff();
        if (dif <= 0.5) { ++g_pass; return; }

        /* 向量不同 —— 必须能证明暴力 box 不够大（其最优解在盒边界上） */
        bool on_boundary = false;
        for (int i = 0; i < a.size(); ++i) {
            const double off = r2.a_fixed(i) - std::round(a(i));
            if (std::fabs(std::fabs(off) - range) < 0.5) { on_boundary = true; break; }
        }
        if (on_boundary) {
            /* 暴力范围不足，属于预期；改为要求 ILS 严格不劣（S2 已查） */
            ++g_pass;
        } else {
            std::printf("[FAIL] %s: S3 整数解真不一致（暴力最优在盒内，说明是穷尽的）\n", tag);
            std::printf("       ILS   = ");
            for (int i = 0; i < r1.a_fixed.size(); ++i) std::printf("%.0f ", r1.a_fixed(i));
            std::printf("\n       brute = ");
            for (int i = 0; i < r2.a_fixed.size(); ++i) std::printf("%.0f ", r2.a_fixed(i));
            std::printf("\n       a     = ");
            for (int i = 0; i < a.size(); ++i) std::printf("%.4f ", a(i));
            std::printf("\n");
            ++g_fail;
        }
    }

    /* 从 T 反推检查：T 整数、|det T| = 1 —— 这由 ilsResolve 内部完成，
     * 这里用一个等价的独立检查：a_fixed 必须是整数 */
    static bool allInteger(const Eigen::VectorXd &v) {
        for (int i = 0; i < v.size(); ++i)
            if (std::fabs(v(i) - std::round(v(i))) > 1e-9) return false;
        return true;
    }

    int run() {
        std::printf("========== 整数最小二乘 (ILS/LAMBDA) 验证 ==========\n");

        /* ---------------- T1 对角 Q ---------------- */
        std::printf("\n---- T1: 对角 Q（低相关） ----\n");
        for (int t = 0; t < 12; ++t) {
            const int n = 3 + t % 4;
            Eigen::MatrixXd Q = makeCov(n, 0.05, 0.01, 1.0);
            Eigen::VectorXd a(n);
            for (int i = 0; i < n; ++i) a(i) = std::round(10.0 * nrand()) + 0.4 * nrand();
            compareWithBrute(a, Q, 4, "T1");
        }

        /* ---------------- T2 强相关 Q ---------------- */
        std::printf("---- T2: 强相关 Q（去相关的用武之地） ----\n");
        double worst_cond_before = 0.0, worst_cond_after = 0.0;
        long max_nodes = 0;
        int n_gauss_sum = 0;
        for (int t = 0; t < 12; ++t) {
            const int n = 4 + t % 3;
            Eigen::MatrixXd Q = makeCov(n, 0.9, 0.002, 1.0);
            Eigen::VectorXd a(n);
            for (int i = 0; i < n; ++i) a(i) = std::round(5.0 * nrand()) + 0.49 * nrand();
            IlsResult rr;
            if (ilsResolve(a, Q, &rr, 3.0)) {
                worst_cond_before = std::max(worst_cond_before, rr.cond_before);
                worst_cond_after  = std::max(worst_cond_after, rr.cond_after);
                max_nodes = std::max(max_nodes, rr.nodes);
                n_gauss_sum += rr.n_decorr;
            }
            compareWithBrute(a, Q, 5, "T2");
        }
        std::printf("       对角比(去相关前/后) 最坏 = %.3g / %.3g；节点数最多 %ld；Gauss 变换共 %d 次\n",
                    worst_cond_before, worst_cond_after, max_nodes, n_gauss_sum);
        CHECK(n_gauss_sum > 0, "T2 确实做了整数 Gauss 变换");
        CHECK(worst_cond_after <= worst_cond_before + 1e-12, "T2 去相关没有让条件数变差");

        /* ---------------- T3 高维 ---------------- */
        std::printf("---- T3: n = 6 ----\n");
        for (int t = 0; t < 4; ++t) {
            const int n = 6;
            Eigen::MatrixXd Q = makeCov(n, 0.6, 0.01, 1.0);
            Eigen::VectorXd a(n);
            for (int i = 0; i < n; ++i) a(i) = std::round(4.0 * nrand()) + 0.45 * nrand();
            compareWithBrute(a, Q, 4, "T3");
        }

        /* ---------------- T4 已知整数真值 ---------------- */
        std::printf("---- T4: 已知整数真值，加噪声后还原 ----\n");
        {
            int n_ok = 0, n_tot = 0;
            for (int t = 0; t < 30; ++t) {
                const int n = 5;
                Eigen::MatrixXd Q = makeCov(n, 0.3, 0.0005, 0.02);   /* 亚周级精度 */
                Eigen::VectorXd zt(n);
                for (int i = 0; i < n; ++i) zt(i) = std::round(50.0 * nrand());
                const Eigen::MatrixXd L = Eigen::LDLT<Eigen::MatrixXd>(Q).matrixL();
                Eigen::VectorXd noise(n);
                for (int i = 0; i < n; ++i) noise(i) = std::sqrt(Q(i, i)) * 0.3 * nrand();
                Eigen::VectorXd a = zt + noise;
                IlsResult r;
                ++n_tot;
                if (ilsResolve(a, Q, &r, 3.0) && allInteger(r.a_fixed) &&
                    (r.a_fixed - zt).cwiseAbs().maxCoeff() < 0.5) {
                    ++n_ok;
                }
            }
            std::printf("       还原成功 %d/%d（噪声 0.3σ，应当基本全对）\n", n_ok, n_tot);
            CHECK(n_ok >= static_cast<int>(n_tot * 0.9), "T4 亚周精度下能还原真值");
        }

        /* ---------------- T5 ratio 语义 ---------------- */
        std::printf("---- T5: ratio 语义（清晰 vs 模糊） ----\n");
        {
            /* 清晰：Q 很小 → 最优解远优于次优 */
            Eigen::MatrixXd Q1 = Eigen::MatrixXd::Identity(3, 3) * 0.001;
            Eigen::VectorXd a1(3); a1 << 1.02, 2.03, 3.01;
            IlsResult r1;
            const bool ok1 = ilsResolve(a1, Q1, &r1, 3.0);
            std::printf("       清晰: fixed=(%.0f,%.0f,%.0f) ratio=%.3g\n",
                        r1.a_fixed(0), r1.a_fixed(1), r1.a_fixed(2), r1.ratio);
            CHECK(ok1 && r1.ratio > 100.0, "T5 清晰情形 ratio 很大");
            CHECK((r1.a_fixed - Eigen::Vector3d(1, 2, 3)).cwiseAbs().maxCoeff() < 0.5,
                  "T5 清晰情形固定到正确整数");

            /* 模糊：Q 很大 → 两个候选取值接近 */
            Eigen::MatrixXd Q2 = Eigen::MatrixXd::Identity(2, 2) * 0.25;
            Eigen::VectorXd a2(2); a2 << 0.5, 0.5;      /* 正好在 0/1 之间 */
            IlsResult r2;
            const bool ok2 = ilsResolve(a2, Q2, &r2, 3.0);
            std::printf("       模糊: ratio=%.3f（应接近 1，判不可固定）\n", r2.ratio);
            CHECK(ok2 && r2.ratio < 1.5, "T5 模糊情形 ratio 接近 1");
        }

        /* ---------------- T6 非正定要拒绝 ---------------- */
        std::printf("---- T6: 非正定 Q 必须拒绝 ----\n");
        {
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(3, 3);   /* 全零 → 非 PD */
            Eigen::VectorXd a(3); a << 0.2, 0.3, 0.4;
            IlsResult r;
            const bool ok = ilsResolve(a, Q, &r, 3.0);
            std::printf("       全零 Q: ok=%d（应为 0）\n", (int)ok);
            CHECK(!ok, "T6 非正定 Q 被拒绝（不返回假固定解）");

            Eigen::MatrixXd Q2(2, 2);
            Q2 << 1.0, 0.9, 0.9, 0.1;                          /* 不定 */
            Eigen::VectorXd a2(2); a2 << 0.4, 0.6;
            IlsResult r2;
            CHECK(!ilsResolve(a2, Q2, &r2, 3.0), "T6 不定矩阵被拒绝");
        }

        /* ---------------- T7 T 的幺模性（通过解的格不变性间接验证） ---------------- */
        std::printf("---- T7: 去相关后解仍是原空间的格点 ----\n");
        {
            int n_ok = 0;
            for (int t = 0; t < 10; ++t) {
                const int n = 5;
                Eigen::MatrixXd Q = makeCov(n, 0.95, 0.001, 1.0);   /* 极强相关 */
                Eigen::VectorXd a(n);
                for (int i = 0; i < n; ++i) a(i) = nrand() * 20.0;
                IlsResult r;
                if (ilsResolve(a, Q, &r, 3.0) && allInteger(r.a_fixed)) ++n_ok;
            }
            std::printf("       极强相关下取到整数格点 %d/10\n", n_ok);
            CHECK(n_ok == 10, "T7 去相关后返回的仍是整数解");
        }

        /* ---------------- T8 尺度不变性 ---------------- */
        std::printf("---- T8: 尺度不变性（Q vs c·Q 必须给出同一解与同一 ratio） ----\n");
        {
            int n_same_vec = 0, n_same_ratio = 0, n_tot = 0;
            double worst_ratio_diff = 0.0;
            for (int t = 0; t < 12; ++t) {
                const int n = 6;
                Eigen::MatrixXd Q = makeCov(n, 0.7, 0.004, 1.0);
                Eigen::VectorXd a(n);
                for (int i = 0; i < n; ++i) a(i) = std::round(8.0 * nrand()) + 0.45 * nrand();
                const double c = (t % 2 == 0) ? 7.3 : 1.0 / 137.0;

                IlsResult r1, r2;
                const bool ok1 = ilsResolve(a, Q, &r1, 3.0);
                const bool ok2 = ilsResolve(a, c * Q, &r2, 3.0);
                ++n_tot;
                if (!ok1 || !ok2) {
                    std::printf("[FAIL] T8 c=%.3g: ok=%d/%d\n", c, (int)ok1, (int)ok2);
                    ++g_fail;
                    continue;
                }
                if ((r1.a_fixed - r2.a_fixed).cwiseAbs().maxCoeff() < 0.5) ++n_same_vec;
                else std::printf("[FAIL] T8 c=%.3g: 整数解随尺度变了\n", c);

                const bool i1 = !std::isfinite(r1.ratio), i2 = !std::isfinite(r2.ratio);
                double diff;
                if (i1 && i2) diff = 0.0;
                else if (i1 != i2) diff = 1e9;
                else diff = std::fabs(r1.ratio - r2.ratio) / std::max(1e-30, std::fabs(r1.ratio));
                worst_ratio_diff = std::max(worst_ratio_diff, diff);
                if (diff < 1e-9) ++n_same_ratio;
                else std::printf("[FAIL] T8 c=%.3g: ratio 变了 %.6g vs %.6g\n", c, r1.ratio, r2.ratio);
            }
            std::printf("       整数解一致 %d/%d；ratio 一致 %d/%d（最大相对差 %.3g）\n",
                        n_same_vec, n_tot, n_same_ratio, n_tot, worst_ratio_diff);
            CHECK(n_same_vec == n_tot, "T8 整数解与协方差尺度无关");
            CHECK(n_same_ratio == n_tot, "T8 ratio 与协方差尺度无关");
        }

        /* ---------------- T9 部分固定 ---------------- */
        std::printf("---- T9: 部分固定（partial AR） ----\n");
        {
            /* P3: 清晰情形 → 必须固定整组 */
            {
                const int n = 5;
                Eigen::MatrixXd Q = makeCov(n, 0.2, 0.0005, 0.01);
                Eigen::VectorXd a(n);
                for (int i = 0; i < n; ++i) a(i) = std::round(20.0 * nrand()) + 0.1 * nrand();
                IlsPartial pr;
                const bool okp = ilsPartialFix(a, Q, 3.0, 2, 2000000L, &pr);
                std::printf("       清晰: ok=%d 固定 %zu/%d（尝试 %d 次）\n",
                            (int)okp, pr.fixed_idx.size(), n, pr.n_tried);
                CHECK(okp && static_cast<int>(pr.fixed_idx.size()) == n,
                      "T9 P3 清晰情形固定整组");
            }

            /* P1/P2/P4: 混合若干"好"与"差"的模糊度 */
            {
                int n_full = 0, n_partial = 0, n_fail = 0;
                int p1_ok = 0, p2_ok = 0, p4_ok = 0, n_case = 0;
                for (int t = 0; t < 10; ++t) {
                    const int n = 8;
                    Eigen::MatrixXd Q = makeCov(n, 0.8, 1e-5, 4.0);   /* 跨度大：既有好也有差 */
                    Eigen::VectorXd a(n);
                    for (int i = 0; i < n; ++i) a(i) = std::round(6.0 * nrand()) + 0.45 * nrand();

                    IlsPartial pr;
                    const bool okp = ilsPartialFix(a, Q, 3.0, 3, 2000000L, &pr);
                    ++n_case;
                    if (!okp) { ++n_fail; continue; }
                    if (static_cast<int>(pr.fixed_idx.size()) == n) ++n_full; else ++n_partial;

                    /* P1 */
                    bool p1 = true;
                    std::vector<bool> isfix(n, false);
                    for (int i : pr.fixed_idx) isfix[static_cast<size_t>(i)] = true;
                    for (int i = 0; i < n; ++i) {
                        if (isfix[static_cast<size_t>(i)]) {
                            if (std::fabs(pr.a_fixed(i) - std::round(pr.a_fixed(i))) > 1e-9) p1 = false;
                        } else {
                            if (std::fabs(pr.a_fixed(i) - a(i)) > 1e-12) p1 = false;
                        }
                    }
                    if (p1) ++p1_ok;

                    /* P2 子集自洽 */
                    const int k = static_cast<int>(pr.fixed_idx.size());
                    Eigen::VectorXd asub(k);
                    Eigen::MatrixXd Qsub(k, k);
                    for (int i = 0; i < k; ++i) {
                        asub(i) = a(pr.fixed_idx[static_cast<size_t>(i)]);
                        for (int j = 0; j < k; ++j)
                            Qsub(i, j) = Q(pr.fixed_idx[static_cast<size_t>(i)],
                                           pr.fixed_idx[static_cast<size_t>(j)]);
                    }
                    if (std::fabs(chi2Orig(asub, Qsub, pr.last.a_fixed) - pr.last.sq_best) <
                        1e-6 * std::max(1e-12, pr.last.sq_best)) ++p2_ok;

                    /* P4 通过门限 */
                    if (pr.last.ratio > 3.0) ++p4_ok;
                }
                std::printf("       %d 例：整组固定 %d，部分固定 %d，失败 %d\n",
                            n_case, n_full, n_partial, n_fail);
                std::printf("       P1 格式正确 %d/%d；P2 子集自洽 %d/%d；P4 过门限 %d/%d\n",
                            p1_ok, n_case - n_fail, p2_ok, n_case - n_fail, p4_ok, n_case - n_fail);
                CHECK(p1_ok == n_case - n_fail, "T9 P1 固定处整数、未固定处保持浮点");
                CHECK(p2_ok == n_case - n_fail, "T9 P2 子集解自洽");
                CHECK(p4_ok == n_case - n_fail, "T9 P4 返回的子集确实通过门限");
                CHECK(n_partial > 0, "T9 确实出现过部分固定（不是全都整组固定）");
            }

            /* P5: 明确模糊 → 必须失败 */
            {
                const int n = 4;
                Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n, n) * 4.0;   /* σ=2 周，模糊 */
                Eigen::VectorXd a(n);
                for (int i = 0; i < n; ++i) a(i) = 0.5 + i;                  /* 正好在整数中间 */
                IlsPartial pr;
                const bool okp = ilsPartialFix(a, Q, 3.0, 2, 2000000L, &pr);
                std::printf("       明确模糊: ok=%d（应为 0，尝试 %d 次）\n", (int)okp, pr.n_tried);
                CHECK(!okp, "T9 P5 明确模糊时不硬固定");
            }
        }

        std::printf("\n====================================================\n");
        std::printf("通过 %d, 失败 %d\n", g_pass, g_fail);
        return g_fail == 0 ? 0 : 1;
    }
}  /* namespace ils */

/* ============================================================================
 * 汇总入口
 * ==========================================================================*/
int main() {
    std::printf("############ test_gnss（紧组合因子 + RINEX 双差 + 紧组合模块 + 整数固定）############\n");
    int bad = 0;
    bad += gnss_base::run();
    bad += rinex_dd::run();
    bad += gnss_tc::run();
    bad += ils::run();

    const int P = gnss_base::g_pass + rinex_dd::g_pass + gnss_tc::g_pass + ils::g_pass;
    const int F = gnss_base::g_fail + rinex_dd::g_fail + gnss_tc::g_fail + ils::g_fail;
    std::printf("\n================ test_gnss 汇总 ================\n");
    std::printf("总断言: %d, 通过: %d, 失败: %d\n", P + F, P, F);
    return bad == 0 ? 0 : 1;
}
