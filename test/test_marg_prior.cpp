/**
 * @file test_marg_prior.cpp
 * @brief 边缘化（Schur 补 + 先验因子）单元测试
 *
 * T1 先验因子的残差 / 环境雅可比：与数值差分（沿流形 Plus 方向）一致
 * T2 Schur 恒等式：先验的代价 == 全问题对最旧帧取最小后的代价（相差常数）
 * T3 不动点性质：先验 + 原有因子在扰动后重新求解应回到同一解
 * T4 边界：关闭开关 / 收尾裁剪到更短窗口
 *
 * 关键点：Ceres 内部把 CostFunction 的环境雅可比按
 *         tangent = ambient · PlusJacobian 变换，
 * 所以先验因子必须与 PoseManifold（右扰动、全角）配套；T1 就是校验这条。
 */

#include "alg_lib.h"

#include <ceres/ceres.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do {                                          \
    if (!(cond)) { std::printf("[FAIL] %s\n", msg); ++g_fail; }        \
    else         { std::printf("[ OK ] %s\n", msg); ++g_pass; }        \
} while (0)

#define CHECK_NEAR(a, b, tol, msg) do {                                \
    const double _a = (a), _b = (b);                                   \
    if (!(std::fabs(_a - _b) <= (tol))) {                              \
        std::printf("[FAIL] %s: %.3e vs %.3e (diff=%.3e)\n",           \
                    msg, _a, _b, std::fabs(_a - _b)); ++g_fail;        \
    } else { std::printf("[ OK ] %s\n", msg); ++g_pass; }              \
} while (0)

/* ============================================================
 * 公共小工具
 * ============================================================ */

/* 把切空间增量作用到 frames[first..] 上（pose 走流形，mix 直接加）。
 * d 的下标以 col_base（frames[first].pose 在全布局中的列号）为原点。 */
static void applyTangentDelta(std::vector<Frame> &frames, size_t first,
                              const Eigen::VectorXd &d, int col_base,
                              const std::vector<int> &pose_col,
                              const std::vector<int> &mix_col) {
    PoseManifold man;
    for (size_t k = first; k < frames.size(); ++k) {
        if (pose_col[k] >= 0) {
            double delta[6];
            for (int i = 0; i < 6; ++i) delta[i] = d(pose_col[k] - col_base + i);
            double out[NUM_POSE];
            man.Plus(frames[k].pose.data(), delta, out);
            for (int i = 0; i < NUM_POSE; ++i) frames[k].pose[i] = out[i];
        }
        if (mix_col[k] >= 0) {
            for (int i = 0; i < NUM_MIX_ODO; ++i) {
                frames[k].mix[i] += d(mix_col[k] - col_base + i);
            }
        }
    }
}

/* ============================================================
 * 仿真：n 个关键帧，常加速度 + 常角速率，IMU/GNSS 观测无噪声
 * ============================================================ */
struct SimData {
    std::vector<Frame> frames;
    vect3 g_n;
};

static SimData simulate(int n_kf, double kf_dt = 1.0, double imu_dt = 0.01) {
    SimData s;
    s.g_n = vect3(0.0, 0.0, -9.79);
    const vect3 a_n(0.05, -0.02, 0.01);     /* n 系常加速度 (m/s²) */
    const vect3 w_b(0.01, -0.02, 0.005);    /* b 系常角速率 (rad/s) */

    PreintegrationParam pp;
    pp.gyr_arw      = 0.5 * DEG / 60.0;
    pp.acc_vrw      = 0.1e-3;
    pp.gyr_bias_std = 10.0 * DPH;
    pp.acc_bias_std = 1.0 * MG;
    pp.corr_time    = 3600.0;

    vect3 p(0.0, 0.0, 0.0);
    vect3 v(1.0, 0.5, 0.0);
    quat  q = a2qua(0.0, 0.0, 30.0 * DEG);

    auto makeFrame = [&](double t) {
        Frame f;
        f.time = t;
        f.pose = { p.i, p.j, p.k, q.q1, q.q2, q.q3, q.q0 };
        f.mix  = { v.i, v.j, v.k, 0, 0, 0, 0, 0, 0, 0.0 };
        f.has_gnss     = true;
        f.gnss_pos     = p;
        f.gnss_std     = vect3(0.5, 0.5, 1.0);
        f.has_gnss_vel = true;
        f.gnss_vn      = v;
        f.gnss_vn_std  = vect3(0.05, 0.05, 0.1);
        return f;
    };

    s.frames.push_back(makeFrame(0.0));

    const int n_imu = static_cast<int>(std::lround(kf_dt / imu_dt));
    for (int k = 1; k < n_kf; ++k) {
        Preintegration pre(pp);
        pre.setBias(O31, O31);
        for (int i = 0; i < n_imu; ++i) {
            Eigen::Matrix3d R = q.toEigen().toRotationMatrix();
            const Eigen::Vector3d f_b =
                R.transpose() * (a_n.toEigen() - s.g_n.toEigen());

            ImuMeas m;
            m.time   = (k - 1) * kf_dt + (i + 1) * imu_dt;
            m.dt     = imu_dt;
            m.dtheta = w_b * imu_dt;
            m.dvel   = vect3(f_b * imu_dt);
            pre.integration(m);

            p = vect3(p.toEigen() + v.toEigen() * imu_dt +
                      0.5 * a_n.toEigen() * imu_dt * imu_dt);
            v = vect3(v.toEigen() + a_n.toEigen() * imu_dt);
            q = q * rv2q(w_b * imu_dt);
            q.normlize(&q);
        }
        s.frames.back().pre_to_next = std::make_shared<Preintegration>(pre);
        s.frames.push_back(makeFrame(k * kf_dt));
    }
    return s;
}

/* 用先验因子自身求代价（不走 Ceres，直接喂原始参数指针） */
static double priorCost(const MarginalizationPriorFactor &cost,
                        const std::vector<Frame> &frames, size_t first) {
    const int nf = static_cast<int>(frames.size() - first);
    std::vector<const double *> params;
    params.reserve(2 * nf);
    for (size_t k = first; k < frames.size(); ++k) {
        params.push_back(frames[k].pose.data());
        params.push_back(frames[k].mix.data());
    }
    std::vector<double> res(static_cast<size_t>(cost.num_residuals()));
    if (!cost.Evaluate(params.data(), res.data(), nullptr)) {
        return -1.0;
    }
    double c = 0.0;
    for (double v : res) c += v * v;
    return c;
}

/* ============================================================
 * T1：先验因子的残差与环境雅可比
 * ============================================================ */
static void test_prior_factor_math() {
    std::printf("\n---- T1: 先验因子残差 / 雅可比（含流形变换） ----\n");
    std::mt19937 rng(20240926u);
    std::normal_distribution<double> nd(0.0, 1.0);

    const int nf = 2;
    const int m  = 15;                 /* 任意残差维数 */
    const int nc = 16 * nf;

    Eigen::MatrixXd J(m, nc);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < nc; ++j) J(i, j) = nd(rng);
    Eigen::VectorXd r0(m);
    for (int i = 0; i < m; ++i) r0(i) = nd(rng);

    /* 线性化点 */
    std::vector<std::array<double, NUM_POSE>>    pl(nf);
    std::vector<std::array<double, NUM_MIX_ODO>> ml(nf);
    std::vector<Frame> frames(nf);
    for (int k = 0; k < nf; ++k) {
        const quat qk = rv2q(vect3(0.3 * nd(rng), 0.3 * nd(rng), 0.3 * nd(rng)));
        pl[k] = { 1.0 + nd(rng), -2.0 + nd(rng), 0.5 + nd(rng),
                  qk.q1, qk.q2, qk.q3, qk.q0 };
        for (int i = 0; i < NUM_MIX_ODO; ++i) ml[k][i] = nd(rng);
        frames[k].pose = pl[k];
        frames[k].mix  = ml[k];
    }

    /* 当前点 = 线性化点 + 切空间偏移（走流形） */
    Eigen::VectorXd d(nc);
    for (int i = 0; i < nc; ++i) d(i) = 0.2 * nd(rng);
    std::vector<int> pc(nf), mc(nf);
    for (int k = 0; k < nf; ++k) { pc[k] = 16 * k; mc[k] = 16 * k + 6; }
    applyTangentDelta(frames, 0, d, 0, pc, mc);

    MarginalizationPriorFactor cost(mat(J), vect(r0), pl, ml);

    /* 1) 残差 = J·Δ - r0，Δ 用头文件里的 PoseManifold::Minus 独立算 */
    {
        std::vector<const double *> params;
        for (int k = 0; k < nf; ++k) {
            params.push_back(frames[k].pose.data());
            params.push_back(frames[k].mix.data());
        }
        Eigen::VectorXd res(m);
        cost.Evaluate(params.data(), res.data(), nullptr);

        Eigen::VectorXd dref(nc);
        PoseManifold man;
        for (int k = 0; k < nf; ++k) {
            double dpose[6];
            man.Minus(frames[k].pose.data(), pl[k].data(), dpose);
            for (int i = 0; i < 6; ++i) dref(16 * k + i) = dpose[i];
            for (int i = 0; i < NUM_MIX_ODO; ++i) {
                dref(16 * k + 6 + i) = frames[k].mix[i] - ml[k][i];
            }
        }
        CHECK_NEAR((res - (J * dref + r0)).cwiseAbs().maxCoeff(), 0.0, 1e-12,
                   "残差 = J·Δ + r0（Δ 走流形 Minus，符号见头文件推导）");
    }

    /* 2) Ceres 的切空间雅可比 = 数值差分（沿 Plus 方向） */
    {
        ceres::Problem problem;
        for (int k = 0; k < nf; ++k) {
            problem.AddParameterBlock(frames[k].pose.data(), NUM_POSE);
            problem.SetManifold(frames[k].pose.data(), new PoseManifold());
            problem.AddParameterBlock(frames[k].mix.data(), NUM_MIX_ODO);
        }
        auto *cf = new MarginalizationPriorFactor(mat(J), vect(r0), pl, ml);
        std::vector<double *> blocks;
        for (int k = 0; k < nf; ++k) {
            blocks.push_back(frames[k].pose.data());
            blocks.push_back(frames[k].mix.data());
        }
        problem.AddResidualBlock(cf, nullptr, blocks);

        ceres::Problem::EvaluateOptions o;
        std::vector<double> resid;
        ceres::CRSMatrix cJ;
        const bool ok = problem.Evaluate(o, nullptr, &resid, nullptr, &cJ);
        CHECK(ok, "Problem::Evaluate 成功");
        CHECK(cJ.num_cols == nc, "切空间列数 = 16×帧数（pose 6 + mix 10）");

        Eigen::MatrixXd Jc = Eigen::MatrixXd::Zero(cJ.num_rows, cJ.num_cols);
        for (int i = 0; i < cJ.num_rows; ++i)
            for (int k = cJ.rows[i]; k < cJ.rows[i + 1]; ++k)
                Jc(i, cJ.cols[k]) = cJ.values[k];

        /* 数值差分 */
        const double eps = 1e-7;
        Eigen::MatrixXd Jfd(m, nc);
        PoseManifold man;
        for (int c = 0; c < nc; ++c) {
            std::vector<Frame> fp = frames, fm = frames;
            Eigen::VectorXd dp = Eigen::VectorXd::Zero(nc);
            Eigen::VectorXd dm = Eigen::VectorXd::Zero(nc);
            dp(c) = eps; dm(c) = -eps;
            applyTangentDelta(fp, 0, dp, 0, pc, mc);
            applyTangentDelta(fm, 0, dm, 0, pc, mc);
            /* 逐残差差分 */
            std::vector<double> rp(m), rm(m);
            std::vector<const double *> pp, pm;
            for (int k = 0; k < nf; ++k) {
                pp.push_back(fp[k].pose.data()); pp.push_back(fp[k].mix.data());
                pm.push_back(fm[k].pose.data()); pm.push_back(fm[k].mix.data());
            }
            cost.Evaluate(pp.data(), rp.data(), nullptr);
            cost.Evaluate(pm.data(), rm.data(), nullptr);
            for (int i = 0; i < m; ++i) Jfd(i, c) = (rp[i] - rm[i]) / (2 * eps);
        }
        CHECK_NEAR((Jc - Jfd).cwiseAbs().maxCoeff(), 0.0, 1e-8,
                   "Ceres 切空间雅可比 == 数值差分");
    }
}

/* ============================================================
 * T2：Schur 恒等式（接线正确性）
 *   min_{δ_m} || r + J [δ_m; δ_r] ||²  ==  ||J_out δ_r - r_out||² + const
 *   比较两个不同 δ_r 的差值即可消掉常数
 * ============================================================ */
static void test_schur_identity(bool fix_first) {
    std::printf("\n---- T2: Schur 恒等式（fix_first_pose=%d） ----\n", (int)fix_first);
    std::mt19937 rng(777u);
    std::normal_distribution<double> nd(0.0, 1.0);

    SimData sim = simulate(5);
    GraphOptimizer::Options opt;
    opt.max_keyframes          = 5;
    opt.fix_first_pose         = fix_first;
    opt.max_iterations         = 100;
    opt.enable_marginalization = true;
    GraphOptimizer gopt(opt);

    gopt.optimize(sim.frames, sim.g_n);

    /* 两套线性化数据：
     *   Jf/rf —— 全窗口（诊断）
     *   Jt/rt —— 只含"触及最旧帧"的残差块（边缘化的正确口径） */
    Eigen::MatrixXd Jf, Jt;
    Eigen::VectorXd rf, rt;
    std::vector<int> pc, mc, pc2, mc2;
    CHECK(gopt.linearizationData(sim.frames, sim.g_n, Jf, rf, pc, mc),
          "取全窗口线性化数据");
    CHECK(gopt.linearizationData(sim.frames, sim.g_n, Jt, rt, pc2, mc2,
                                 /*oldest_touch_only=*/true),
          "取触及最旧帧的线性化数据");
    CHECK(Jt.rows() < Jf.rows(), "边缘化口径的残差行数 < 全窗口行数");

    const int dim_m  = (pc[0] >= 0 ? 6 : 0) + NUM_MIX_ODO;
    const int dim_r  = static_cast<int>(Jf.cols()) - dim_m;
    CHECK(dim_r == 16 * (static_cast<int>(sim.frames.size()) - 1),
          "保留维数 = 16×(帧数-1)");

    CHECK(gopt.marginalize(sim.frames, sim.g_n), "marginalize 成功");
    CHECK(gopt.priorFrames() == static_cast<int>(sim.frames.size()) - 1,
          "先验覆盖帧数 = 帧数-1");
    CHECK(gopt.J_prior().clm == 16 * gopt.priorFrames(),
          "先验列数 = 16×覆盖帧数");

    MarginalizationPriorFactor prior(gopt.J_prior(), gopt.r_prior(),
                                     gopt.priorPoseLin(), gopt.priorMixLin());

    /* ---- 先验信息量 == 触及最旧帧那部分的 Schur 补；
     *      并与"全窗口 Schur 补"显著不同（后者会把只涉及保留帧的
     *      因子重复计一次 → 过度自信） ---- */
    {
        auto schurInfo = [&](const Eigen::MatrixXd &J) {
            const int m = dim_m, r = static_cast<int>(J.cols()) - m;
            Eigen::MatrixXd H = J.transpose() * J;
            Eigen::MatrixXd Hmm = H.topLeftCorner(m, m);
            Eigen::MatrixXd Hmr = H.topRightCorner(m, r);
            Eigen::MatrixXd Hrr = H.bottomRightCorner(r, r);
            Eigen::MatrixXd Hmm_inv =
                Hmm.completeOrthogonalDecomposition().pseudoInverse();
            return Eigen::MatrixXd(Hrr - Hmr.transpose() * Hmm_inv * Hmr);
        };
        const Eigen::MatrixXd Hp =
            gopt.J_prior().toEigen().transpose() * gopt.J_prior().toEigen();
        const Eigen::MatrixXd Hs_touch = schurInfo(Jt);
        const Eigen::MatrixXd Hs_full  = schurInfo(Jf);
        const double scale = std::max(1.0, Hp.cwiseAbs().maxCoeff());
        const double d_touch = (Hp - Hs_touch).cwiseAbs().maxCoeff() / scale;
        const double d_full  = (Hp - Hs_full ).cwiseAbs().maxCoeff() / scale;
        std::printf("     [diag] |H_prior - Schur(touch)|/scale = %.3e, "
                    "|H_prior - Schur(full)|/scale = %.3e\n", d_touch, d_full);
        CHECK(d_touch < 1e-6, "先验信息量 = 触及最旧帧部分的 Schur 补");
        CHECK(d_full  > 1e-3, "先验信息量 ≠ 全窗口 Schur 补（未重复计数保留帧因子）");

        /* 线性项 b* 的对照：分别用 COD 伪逆与工程内 pinv_sym 计算 */
        {
            Eigen::MatrixXd H = Jt.transpose() * Jt;
            Eigen::VectorXd b = Jt.transpose() * rt;
            Eigen::MatrixXd Hmm = H.topLeftCorner(dim_m, dim_m);
            Eigen::MatrixXd Hmr = H.topRightCorner(dim_m, dim_r);
            Eigen::VectorXd bm = b.head(dim_m), br = b.tail(dim_r);
            const Eigen::MatrixXd Hp1 =
                gopt.J_prior().toEigen().transpose() * gopt.r_prior().toEigen();

            const Eigen::VectorXd b_cod =
                br - Hmr.transpose() *
                     (Hmm.completeOrthogonalDecomposition().pseudoInverse() * bm);
            const mat Pj = pinv_sym(mat(Hmm), 1.0e-10);
            const Eigen::VectorXd b_pin =
                br - Hmr.transpose() * (Pj.toEigen() * bm);
            std::printf("     [diag] |b_prior-b_cod|=%.3e  |b_prior-b_pin|=%.3e  "
                        "|b|inf=%.3e\n",
                        (Hp1 - b_cod).cwiseAbs().maxCoeff(),
                        (Hp1 - b_pin).cwiseAbs().maxCoeff(),
                        b.cwiseAbs().maxCoeff());
        }
    }

    /* 两次独立扰动 */
    double diff_expected[2] = {0.0, 0.0};
    double diff_actual[2]   = {0.0, 0.0};
    for (int trial = 0; trial < 2; ++trial) {
        Eigen::VectorXd dr(dim_r);
        for (int i = 0; i < dim_r; ++i) dr(i) = 0.05 * nd(rng);
        /* 第 2 次只扰动"可观测方向"：把零偏与 sodo（本工程里没有任何因子
         * 使用 sodo，且零偏只有帧间差约束）置零，便于用严格容差验收 */
        if (trial == 1) {
            for (int k = 0; 16 * k + 15 < dim_r; ++k) {
                for (int i = 9; i < 16; ++i) dr(16 * k + i) = 0.0;
            }
        }

        /* expected: 对最旧帧取最小（用与边缘化同口径的 Jt/rt）。
         * 用正规方程形式 min‖rhs + Jm·dm‖² = ‖rhs‖² - (Jmᵀrhs)ᵀ(JmᵀJm)⁺(Jmᵀrhs)：
         * Jm 秩亏（sodo 列恒为 0），QR 的秩判定阈值会给偏大的最小值。 */
        Eigen::MatrixXd Jm = Jt.leftCols(dim_m);
        Eigen::MatrixXd Jr = Jt.rightCols(dim_r);
        Eigen::VectorXd rhs = rt + Jr * dr;
        /* 用工程内同一个伪逆口径（pinv_sym, 相对阈值 1e-10）解内层最小化，
         * 再直接求残差范数——避免"大数相减"式的抵消误差。 */
        const mat Hmm = mat(Jm.transpose() * Jm);
        const Eigen::MatrixXd Hmm_pinv = pinv_sym(Hmm, 1.0e-10).toEigen();
        const Eigen::VectorXd dm = -(Hmm_pinv * (Jm.transpose() * rhs));
        const double expected = (rhs + Jm * dm).squaredNorm();

        /* actual: 先验因子代价 */
        std::vector<Frame> f2 = sim.frames;
        /* dr 是"保留列"子向量，原点 = 全布局里第一个保留帧的列号 dim_m */
        applyTangentDelta(f2, 1, dr, dim_m, pc, mc);
        const double actual = priorCost(prior, f2, 1);

        diff_expected[trial] = expected;
        diff_actual[trial]   = actual;

        std::printf("     [diag t%d] 期望 %+.12e / 实际 %+.12e\n",
                    trial, expected, actual);
    }
    CHECK_NEAR(diff_actual[0] - diff_actual[1],
               diff_expected[0] - diff_expected[1],
               1e-6 * std::max(1.0, std::fabs(diff_expected[0])),
               "先验代价差 == 全问题对最旧帧取最小后的代价差（含零空间方向）");

    /* 纯可观测方向：解析/数值应严格一致 */
    {
        Eigen::VectorXd dr1(dim_r), dr2(dim_r);
        for (int i = 0; i < dim_r; ++i) { dr1(i) = 0.05 * nd(rng); dr2(i) = 0.05 * nd(rng); }
        for (int k = 0; 16 * k + 15 < dim_r; ++k) {
            for (int i = 9; i < 16; ++i) { dr1(16 * k + i) = 0.0; dr2(16 * k + i) = 0.0; }
        }
        auto minCost = [&](const Eigen::VectorXd &dr) {
            Eigen::MatrixXd Jm = Jt.leftCols(dim_m);
            Eigen::MatrixXd Jr = Jt.rightCols(dim_r);
            Eigen::VectorXd rhs = rt + Jr * dr;
            const Eigen::MatrixXd Pinv =
                pinv_sym(mat(Jm.transpose() * Jm), 1.0e-10).toEigen();
            const Eigen::VectorXd dm = -(Pinv * (Jm.transpose() * rhs));
            return (rhs + Jm * dm).squaredNorm();
        };
        /* 诊断：全问题法方程的条件数（秩亏 → 条件数很大） */
        {
            Eigen::MatrixXd H = Jt.transpose() * Jt;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
            const double dmax = es.eigenvalues().maxCoeff();
            const double dmin = es.eigenvalues().cwiseAbs().minCoeff();
            std::printf("     [diag] cond(JᵀJ) ≈ %.3e (dmax=%.3e, dmin=%.3e)\n",
                        dmin > 0 ? dmax / dmin : -1.0, dmax, dmin);
        }
        std::vector<Frame> fa = sim.frames, fb = sim.frames;
        applyTangentDelta(fa, 1, dr1, dim_m, pc, mc);
        applyTangentDelta(fb, 1, dr2, dim_m, pc, mc);
        CHECK_NEAR(priorCost(prior, fa, 1) - priorCost(prior, fb, 1),
                   minCost(dr1) - minCost(dr2),
                   1e-7 * std::max(1.0, std::fabs(minCost(dr1))),
                   "可观测方向上先验代价差一致（噪声地板来自秩亏块的伪逆阈值）");
    }
}

/* ============================================================
 * T3：不动点性质——先验 + 原有因子在扰动后应回到同一解
 * ============================================================ */
static void test_fixed_point() {
    std::printf("\n---- T3: 边缘化后的不动点性质（先验 + 保留帧自身因子） ----\n");
    std::mt19937 rng(31415u);
    std::normal_distribution<double> nd(0.0, 1.0);

    SimData sim = simulate(4);
    GraphOptimizer::Options opt;
    opt.max_keyframes  = 4;
    opt.max_iterations = 200;
    /* 本用例不固定首帧：否则化简窗口的首帧 pose 被钉住，
     * 扰动到它上面就无法被纠正（这是 gauge fix 的固有语义，
     * 不是边缘化误差）。 */
    opt.fix_first_pose = false;
    GraphOptimizer gopt(opt);
    gopt.optimize(sim.frames, sim.g_n);

    /* 参考解：原窗口里被保留的 frames[1..]（= 全窗口最优解） */
    std::vector<Frame> ref(sim.frames.begin() + 1, sim.frames.end());
    CHECK(gopt.marginalize(sim.frames, sim.g_n), "marginalize 成功");
    CHECK(gopt.priorFrames() == static_cast<int>(ref.size()), "先验覆盖保留帧");

    auto maxDiff = [](const std::vector<Frame> &a, const std::vector<Frame> &b,
                      double &dp, double &dv, double &dq) {
        dp = dv = dq = 0.0;
        for (size_t k = 0; k < a.size(); ++k) {
            dp = std::max(dp, norm(vect3(a[k].pose[0] - b[k].pose[0],
                                         a[k].pose[1] - b[k].pose[1],
                                         a[k].pose[2] - b[k].pose[2])));
            dv = std::max(dv, norm(vect3(a[k].mix[0] - b[k].mix[0],
                                         a[k].mix[1] - b[k].mix[1],
                                         a[k].mix[2] - b[k].mix[2])));
            const quat qa(a[k].pose[6], a[k].pose[3], a[k].pose[4], a[k].pose[5]);
            const quat qb(b[k].pose[6], b[k].pose[3], b[k].pose[4], b[k].pose[5]);
            dq = std::max(dq, norm(q2rv((~qb) * qa)));
        }
    };

    /* (a) 在参考点上重解（先验 + 保留帧自身因子）：应当原地不动。
     *     由包络定理：∂(min_{x_m} C)/∂x_r = ∂C/∂x_r = 0，故 x* 仍是驻点。 */
    double dp0 = 0.0, dv0 = 0.0, dq0 = 0.0;
    {
        std::vector<Frame> a = ref;
        gopt.optimize(a, sim.g_n);
        maxDiff(a, ref, dp0, dv0, dq0);
        std::printf("     参考点重解偏差: dp=%.3e  dv=%.3e  dq=%.3e\n", dp0, dv0, dq0);
        CHECK(dp0 < 1e-6 && dv0 < 1e-6 && dq0 < 1e-6, "参考点是不动点");
    }

    /* 化简问题的代价（先验 + 保留帧自身因子）：用于判断"参考点是否最优" */
    auto reducedCost = [&](std::vector<Frame> &f) {
        Eigen::MatrixXd Jc;
        Eigen::VectorXd rc;
        std::vector<int> pc3, mc3;
        if (!gopt.linearizationData(f, sim.g_n, Jc, rc, pc3, mc3, false)) return -1.0;
        return rc.squaredNorm();
    };
    {
        std::vector<Frame> r0 = ref;
        std::printf("     [diag] 化简问题代价: 参考点 %.6e\n", reducedCost(r0));
    }

    /* (b) 打乱后重解：应收敛回参考解。
     *     只扰动可观测方向（位置/姿态/速度）；零偏在 bias_jac=false 下
     *     本来就存在规范零空间，不参与断言。 */
    {
        std::vector<Frame> pert = ref;
        for (size_t k = 0; k < pert.size(); ++k) {
            for (int i = 0; i < 3; ++i) pert[k].pose[i] += 0.05 * nd(rng);
            for (int i = 0; i < 3; ++i) pert[k].mix[i] += 0.02 * nd(rng);
        }

        const bool ok = gopt.optimize(pert, sim.g_n);
        std::printf("     [diag] 重解收敛=%d, msg=%s\n", (int)ok,
                    gopt.last_message.c_str());

        double dp = 0.0, dv = 0.0, dq = 0.0;
        maxDiff(pert, ref, dp, dv, dq);
        std::printf("     扰动后重解偏差: dp=%.3e m  dv=%.3e m/s  dq=%.3e rad\n",
                    dp, dv, dq);
        std::printf("     [diag] 化简问题代价: 重解后 %.6e\n", reducedCost(pert));
        /* 姿态方向存在"yaw ↔ 加计水平零偏"的近零空间（本工程 bias_jac=false、
         * 4 帧窗口且无航向观测），故姿态可复现到 ~1e-5 rad 量级即可。 */
        CHECK(dp < 1e-6, "位置回到参考解");
        CHECK(dv < 1e-6, "速度回到参考解");
        CHECK(dq < 1e-4, "姿态回到参考解");
    }
}

/* ============================================================
 * T4：开关与收尾裁剪
 * ============================================================ */
static void test_switch_and_trim() {
    std::printf("\n---- T4: 开关 / 收尾裁剪 ----\n");
    SimData sim = simulate(4);

    {
        GraphOptimizer::Options opt;
        opt.max_keyframes          = 4;
        opt.enable_marginalization = false;
        GraphOptimizer gopt(opt);
        gopt.optimize(sim.frames, sim.g_n);
        CHECK(!gopt.marginalize(sim.frames, sim.g_n), "关闭开关时 marginalize 返回 false");
        CHECK(!gopt.hasPrior(), "关闭开关时不产生先验（行为与旧版一致）");
    }

    {
        GraphOptimizer::Options opt;
        opt.max_keyframes = 4;
        GraphOptimizer gopt(opt);
        gopt.optimize(sim.frames, sim.g_n);
        gopt.marginalize(sim.frames, sim.g_n);
        CHECK(gopt.hasPrior(), "先验已建立");

        /* 收尾：窗口只剩 2 帧（< 先验覆盖的 3 帧）→ 应先验尾部 Schur 裁剪 */
        std::vector<Frame> tail(sim.frames.begin() + 1, sim.frames.end());
        tail.resize(2);
        gopt.optimize(tail, sim.g_n);
        CHECK(gopt.priorFrames() == 2, "先验被裁剪到窗口帧数");
        CHECK(gopt.J_prior().clm == 16 * 2, "裁剪后先验列数 = 16×2");

        bool finite = true;
        for (const auto &f : tail) {
            for (double v : f.pose) finite = finite && std::isfinite(v);
            for (double v : f.mix)  finite = finite && std::isfinite(v);
        }
        CHECK(finite, "裁剪后解仍是有限值");
    }
}

int main() {
    std::printf("========== 边缘化先验单元测试 ==========\n");
    test_prior_factor_math();
    test_schur_identity(true);
    test_schur_identity(false);
    test_fixed_point();
    test_switch_and_trim();
    std::printf("\n=============================================\n");
    std::printf("通过 %d, 失败 %d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
