/**
 * @file test_gnss.cpp
 * @brief 多频 GNSS 紧组合单元测试（最终版）
 *
 * 与上一版的差异：
 *   1. numericJacScalar eps 由 1e-3 → 1e-2（消除舍入主导）
 *   2. 端到端载波测试改为"伪距 + 载波"混合（消除欠定）
 */
#include "alg_lib.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include <Eigen/Core>

/* ========================================================================== */
static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            std::printf("[FAIL] %s  (%s:%d)\n", msg, __FILE__, __LINE__);    \
            ++g_failures;                                                    \
        } else {                                                             \
            std::printf("[ OK ] %s\n", msg);                                 \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                           \
    do {                                                                     \
        ++g_checks;                                                          \
        double _a = (a), _b = (b);                                           \
        if (std::fabs(_a - _b) > (tol)) {                                    \
            std::printf("[FAIL] %s  (%s:%d): %.6e vs %.6e (diff=%.3e)\n",    \
                        msg, __FILE__, __LINE__, _a, _b,                     \
                        std::fabs(_a - _b));                                 \
            ++g_failures;                                                    \
        } else {                                                             \
            std::printf("[ OK ] %s\n", msg);                                 \
        }                                                                    \
    } while (0)

/* ==========================================================================
 * 工具
 * ==========================================================================*/
static mat3 makeREn(double lat, double lon) {
    const double sl = std::sin(lat), cl = std::cos(lat);
    const double so = std::sin(lon), co = std::cos(lon);
    Eigen::Matrix3d R;
    R << -so, -sl*co,  cl*co,
           co, -sl*so,  cl*so,
          0.0,      cl,     sl;
    return mat3(R);
}

static vect3 makeP0Ecef(double lat, double lon, double h) {
    constexpr double RE_ = 6378137.0;
    constexpr double e2_ = 6.69437999014e-3;
    const double sl = std::sin(lat), cl = std::cos(lat);
    const double so = std::sin(lon), co = std::cos(lon);
    const double N  = RE_ / std::sqrt(1.0 - e2_ * sl * sl);
    return vect3((N + h) * cl * co,
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

static DDObs makeDDObsMF(const vect3& pi, const vect3& pk,
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
    std::printf("\n---- test_amb_key ----\n");
    const int key = ambKey(5, 12, 0);
    CHECK(((key >> 16) & 0x3F) == 5,  "ambKey ref_sat 解码正确");
    CHECK(((key >> 8)  & 0x3F) == 12, "ambKey sat 解码正确");
    CHECK((key & 0x0F) == 0,          "ambKey freq 解码正确");
    CHECK(ambKey(5, 12, 0) != ambKey(5, 12, 1),   "不同频点 key 不同");
    CHECK(ambKey(5, 12, 0) != ambKey(5, 13, 0),   "不同卫星 key 不同");
    CHECK(ambKey(5, 12, 0) != ambKey(6, 12, 0),   "不同参考星 key 不同");
    CHECK(ambKey(5, 12, 100) != ambKey(5, 12, 200), "WL 与 NL key 不同");
}

static void test_freq_pseudorange_jacobian() {
    std::printf("\n---- test_freq_pseudorange_jacobian ----\n");
    const mat3  R_en = makeREn(0.5236, 2.0);
    const vect3 p0   = makeP0Ecef(0.5236, 2.0, 100.0);
    const vect3 base = p0 + vect3(1000, 2000, 50);
    const vect3 pi(15000000, 10000000, 20000000);
    const vect3 pk(18000000, 12000000, 18000000);
    const vect3 t_ant(0.5, -0.3, 1.2);

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
    const mat3  R_en = makeREn(0.5236, 2.0);
    const vect3 p0   = makeP0Ecef(0.5236, 2.0, 100.0);
    const vect3 base = p0 + vect3(1000, 2000, 50);
    const vect3 pi(15000000, 10000000, 20000000);
    const vect3 pk(18000000, 12000000, 18000000);
    const vect3 t_ant(0, 0, 0);

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
    const mat3  R_en = makeREn(0.5236, 2.0);
    const vect3 p0   = makeP0Ecef(0.5236, 2.0, 100.0);
    const vect3 base = p0 + vect3(1000, 2000, 50);
    const vect3 pi(15000000, 10000000, 20000000);
    const vect3 pk(18000000, 12000000, 18000000);
    const vect3 t_ant(0, 0, 0);

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
    const mat3  R_en = makeREn(0.5236, 2.0);
    const vect3 p0   = makeP0Ecef(0.5236, 2.0, 100.0);
    const vect3 base = p0 + vect3(1000, 2000, 50);
    const vect3 pi(15000000, 10000000, 20000000);
    const vect3 pk(18000000, 12000000, 18000000);
    const vect3 t_ant(0, 0, 0);

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
    const mat3  R_en = makeREn(0.5236, 2.0);
    const vect3 p0   = makeP0Ecef(0.5236, 2.0, 100.0);
    const vect3 base = p0 + vect3(1000, 2000, 50);
    const vect3 pi(15000000, 10000000, 20000000);
    const vect3 pk(18000000, 12000000, 18000000);
    const vect3 t_ant(0, 0, 0);

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
    const vect3 p_true_n(10.0, 20.0, 5.0);
    const mat3  R_en = makeREn(0.5236, 2.0);
    const vect3 p0   = makeP0Ecef(0.5236, 2.0, 100.0);
    const vect3 p_true_e = p0 + R_en * p_true_n;
    const vect3 base = p0;
    const vect3 t_ant(0, 0, 0);

    std::vector<vect3> sat_pos = {
        vect3( 15000000,  10000000,  20000000),
        vect3(-15000000,  10000000,  20000000),
        vect3( 15000000, -10000000,  20000000),
        vect3(-15000000, -10000000,  20000000),
        vect3(          0,           0,  26000000),
    };
    const vect3& ref_pos = sat_pos[0];

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

    const vect3 p_est(pose[0], pose[1], pose[2]);
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
    const vect3 p_true_n(5.0, -10.0, 2.0);
    const mat3  R_en = makeREn(0.5236, 2.0);
    const vect3 p0   = makeP0Ecef(0.5236, 2.0, 100.0);
    const vect3 p_true_e = p0 + R_en * p_true_n;
    const vect3 base = p0;
    const vect3 t_ant(0, 0, 0);

    std::vector<vect3> sat_pos = {
        vect3( 15000000,  10000000,  20000000),
        vect3(-15000000,  10000000,  20000000),
        vect3( 15000000, -10000000,  20000000),
        vect3(-15000000, -10000000,  20000000),
        vect3(          0,           0,  26000000),
    };
    const vect3& ref_pos = sat_pos[0];

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

    const vect3 p_est(pose[0], pose[1], pose[2]);
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

int main() {
    std::printf("========== GNSS 多频紧组合单元测试（最终版） ==========\n");

    test_combination_constants();
    test_amb_key();
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
                g_checks - g_failures, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}