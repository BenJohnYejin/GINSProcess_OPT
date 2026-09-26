/**
 * @file graph_optimization_test.cpp
 * @brief 图优化问题流程端到端测试
 *
 * 仿真-估计闭环：
 *   1) 生成真值轨迹（匀速直线）
 *   2) 由真值反推 IMU 测量与 GNSS 观测
 *   3) 滑窗图优化估计状态
 *   4) 对比估计与真值
 *
 * 依赖：alg_lib.h, state_manifold.h, marginalization.h, Ceres 2.1+
 */

#include "alg_lib.h"
#include <ceres/ceres.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <array>

/* ============================================================================
 * PoseManifold：pose = [p(3), qx,qy,qz,qw]，右扰动，切空间 6 维
 * ==========================================================================*/
class PoseManifold : public ceres::Manifold {
public:
    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }

    bool Plus(const double *x, const double *delta, double *x_plus) const override {
        x_plus[0] = x[0] + delta[0];
        x_plus[1] = x[1] + delta[1];
        x_plus[2] = x[2] + delta[2];

        quat q(x[6], x[3], x[4], x[5]);
        quat dq = rv2q(vect3(delta[3], delta[4], delta[5]));
        quat q_new = q * dq;
        normlize(&q_new);

        x_plus[3] = q_new.q1;
        x_plus[4] = q_new.q2;
        x_plus[5] = q_new.q3;
        x_plus[6] = q_new.q0;
        return true;
    }

    bool PlusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

        const double qx = x[3], qy = x[4], qz = x[5], qw = x[6];
        /* ∂q/∂δφ = 0.5 · [ -q_v  q_w I + [q_v]× ]  转置到 [x,y,z,w] 顺序 */
        Eigen::Matrix<double, 4, 3> dq_dphi;
        dq_dphi << -qx, -qy, -qz,
                    qw, -qz,  qy,
                    qz,  qw, -qx,
                   -qy,  qx,  qw;
        dq_dphi *= 0.5;

        /* pose 的四元数顺序为 [x,y,z,w]，与 Eigen 一致 */
        J.block<1, 3>(3, 3) = dq_dphi.row(1);   // pose[3] = qx
        J.block<1, 3>(4, 3) = dq_dphi.row(2);   // pose[4] = qy
        J.block<1, 3>(5, 3) = dq_dphi.row(3);   // pose[5] = qz
        J.block<1, 3>(6, 3) = dq_dphi.row(0);   // pose[6] = qw
        return true;
    }

    bool Minus(const double *y, const double *x, double *y_minus_x) const override {
        y_minus_x[0] = y[0] - x[0];
        y_minus_x[1] = y[1] - x[1];
        y_minus_x[2] = y[2] - x[2];

        quat qx(x[6], x[3], x[4], x[5]);
        quat qy(y[6], y[3], y[4], y[5]);
        vect3 dphi = q2rv((~qx) * qy);
        y_minus_x[3] = dphi.i;
        y_minus_x[4] = dphi.j;
        y_minus_x[5] = dphi.k;
        return true;
    }

    /* ---- 新增：MinusJacobian ---- */
    bool MinusJacobian(const double *x, double *jacobian) const override {
        /* 形状 6×7（行主序）
         * ∂(y ⊖ x)/∂y|_{y=x}
         *   = [ I_3          0_{3×4} ]
         *     [ 0_{3×3}      J_q     ]
         * 其中 J_q 与 Ceres 内置 QuaternionManifold 保持一致 */
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobian);
        J.setZero();

        /* 位置部分 */
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

        /* 旋转部分：pose 中四元数顺序为 [qx, qy, qz, qw] */
        const double qx = x[3], qy = x[4], qz = x[5], qw = x[6];

        Eigen::Matrix<double, 3, 4> Jq;
        Jq << -qy,  qx, -qw,  qz,
              -qz,  qw,  qx, -qy,
              -qw, -qz,  qy,  qx;
        Jq *= 2.0;

        J.block<3, 4>(3, 3) = Jq;
        return true;
    }
};

/* ============================================================================
 * 辅助：把标量类型从 double 提升到 AutoDiff 的 Jet<T>
 * ==========================================================================*/
template <typename T>
inline Eigen::Matrix<T, 3, 1> toEig3(const vect3 &v) {
    return Eigen::Matrix<T, 3, 1>(T(v.i), T(v.j), T(v.k));
}
template <typename T>
inline Eigen::Quaternion<T> toEigQ(const quat &q) {
    return Eigen::Quaternion<T>(T(q.q0), T(q.q1), T(q.q2), T(q.q3));
}

/* ============================================================================
 * IMU 预积分因子（15 维）
 *   r_p = R_i^T (p_j - p_i - v_i dt - 0.5 g dt²) - Δp
 *   r_v = R_i^T (v_j - v_i - g dt)               - Δv
 *   r_q = 2 [ (Δq)^-1 ⊗ (q_i^-1 ⊗ q_j) ]_xyz
 *   r_bg = bg_j - bg_i
 *   r_ba = ba_j - ba_i
 * ==========================================================================*/
struct PreintResidual {
    PreintResidual(const Preintegration &p, const vect3 &g)
        : preint_(p), g_n_(g) {
        /* ---- 计算白化矩阵 W = L⁻¹，其中 Σ = L·Lᵀ ---- */
        const mat L = llt_L(mat(p.cov()));
        if (L.row == 15) {
            Eigen::MatrixXd Le = L.toEigen();
            Eigen::MatrixXd Linv = Le.triangularView<Eigen::Lower>().solve(
                Eigen::MatrixXd::Identity(15, 15));
            whitening_ = mat(Eigen::MatrixXd(Linv));
        } else {
            whitening_ = mat(Eigen::MatrixXd::Identity(15, 15));
        }
    }

    template <typename T>
    bool operator()(const T *const pose_i, const T *const mix_i,
                    const T *const pose_j, const T *const mix_j,
                    T *residual) const {
        /* ============================================================
         * 1) 读入状态
         * ============================================================ */
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_i(pose_i);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_j(pose_j);
        Eigen::Quaternion<T> q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        Eigen::Quaternion<T> q_j(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_i (mix_i);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_j (mix_j);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg_i(mix_i + 3);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg_j(mix_j + 3);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba_i(mix_i + 6);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba_j(mix_j + 6);

        /* ============================================================
         * 2) 预积分量 & 重力
         * ============================================================ */
        const T dt = T(preint_.dt());
        const Eigen::Matrix<T, 3, 1>  dp  = toEig3<T>(preint_.p());
        const Eigen::Matrix<T, 3, 1>  dv  = toEig3<T>(preint_.v());
        const Eigen::Quaternion<T>    dq  = toEigQ<T>(preint_.q());
        const Eigen::Matrix<T, 3, 1> g_n = toEig3<T>(g_n_);

        const Eigen::Matrix<T, 3, 3> R_i = q_i.toRotationMatrix();

        Eigen::Map<Eigen::Matrix<T, 15, 1>> r(residual);

        /* ============================================================
         * 3) 残差计算（这一步不能省！）
         * ============================================================ */
        /* r_p = R_iᵀ · (p_j - p_i - v_i·dt - 0.5·g·dt²) - Δp */
        r.template segment<3>(0) = R_i.transpose() *
            (p_j - p_i - v_i * dt - T(0.5) * g_n * dt * dt) - dp;

        /* r_v = R_iᵀ · (v_j - v_i - g·dt) - Δv */
        r.template segment<3>(3) = R_i.transpose() *
            (v_j - v_i - g_n * dt) - dv;

        /* r_q = 2 · vec( (Δq)⁻¹ ⊗ (q_i⁻¹ ⊗ q_j) ) */
        const Eigen::Quaternion<T> q_rel = q_i.conjugate() * q_j;
        const Eigen::Quaternion<T> q_err = dq.conjugate() * q_rel;
        r.template segment<3>(6) = T(2.0) * q_err.vec();

        /* r_bg = bg_j - bg_i */
        r.template segment<3>(9)  = bg_j - bg_i;

        /* r_ba = ba_j - ba_i */
        r.template segment<3>(12) = ba_j - ba_i;

        /* ============================================================
         * 4) 白化：r ← W · r
         * ============================================================ */
        if (whitening_.row == 15) {
            Eigen::Matrix<T, 15, 15> W = whitening_.toEigen().template cast<T>();
            r = W * r;
        }
        return true;
    }

    Preintegration preint_;
    vect3         g_n_;
    mat           whitening_;
};

/* ============================================================================
 * GNSS 位置因子（3 维，白化）
 * ==========================================================================*/
struct GnssPosResidual {
    GnssPosResidual(const vect3 &pos, const vect3 &std_dev)
        : pos_(pos), std_(std_dev) {}

    template <typename T>
    bool operator()(const T *const pose, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(pose);
        Eigen::Map<Eigen::Matrix<T, 3, 1>>       r(residual);
        const Eigen::Matrix<T, 3, 1> pos = toEig3<T>(pos_);
        const Eigen::Matrix<T, 3, 1> sd  = toEig3<T>(std_);
        for (int i = 0; i < 3; ++i) r(i) = (p(i) - pos(i)) / sd(i);
        return true;
    }

    vect3 pos_, std_;
};

/* ============================================================================
 * 仿真
 * ==========================================================================*/
struct TruthSample {
    double t;
    vect3  p;
    quat   q;
    vect3  v;
    vect3  bg;
    vect3  ba;
};

struct SimConfig {
    double dt_imu        = 0.005;   /*< IMU 采样周期 200Hz  */
    int    N_imu_per_kf  = 200;     /*< 每关键帧 1s          */
    int    N_kf          = 5;       /*< 关键帧数             */
    vect3  v0            = vect3(1.0, 0.5, 0.0);  /*< 真值速度 */
    vect3  g_n           = vect3(0.0, 0.0, -9.8);
    double sigma_imu     = 0.0;     /*< IMU 噪声（每步）     */
    double sigma_gnss    = 0.0;     /*< GNSS 位置噪声        */
};

struct SimData {
    std::vector<TruthSample> truth;
    std::vector<Frame>       frames;   /*< 已含 pre_to_next 与 gnss 观测 */
};

SimData simulate(const SimConfig &cfg, unsigned seed = 20240918u) {
    SimData out;
    out.truth.resize(cfg.N_kf);
    out.frames.resize(cfg.N_kf);

    /* ---- 1) 真值轨迹（匀速直线） ---- */
    for (int k = 0; k < cfg.N_kf; ++k) {
        const double t = k * cfg.N_imu_per_kf * cfg.dt_imu;
        out.truth[k].t  = t;
        out.truth[k].p  = cfg.v0 * t;
        out.truth[k].q  = qI;
        out.truth[k].v  = cfg.v0;
        out.truth[k].bg = O31;
        out.truth[k].ba = O31;
    }

    /* ---- 2) 由真值反推 IMU 测量 ---- */
    /* 恒定姿态、匀速运动：
     *   比力 f_body = R^T (a_n - g_n) = R^T (-g_n)
     *   R = I 时 f_body = -g_n = [0, 0, +9.8]            */
    const vect3 f_body = -cfg.g_n;

    std::mt19937 rng(seed);
    std::normal_distribution<double> nd_imu (0.0, cfg.sigma_imu);
    std::normal_distribution<double> nd_gnss(0.0, 1.0);



    for (int k = 0; k < cfg.N_kf; ++k) {
        Frame &f = out.frames[k];
        f.time = out.truth[k].t;

        /* 初值：位置给真值（视作无先验），速度/零偏给 0 */
        f.pose = { out.truth[k].p.i, out.truth[k].p.j, out.truth[k].p.k,
                   out.truth[k].q.q1, out.truth[k].q.q2,
                   out.truth[k].q.q3, out.truth[k].q.q0 };
        f.mix  = { 0, 0, 0,   0, 0, 0,   0, 0, 0,   0 };

        /* GNSS 位置观测（带噪） */
        f.has_gnss  = true;
        vect3 noise(nd_gnss(rng) * cfg.sigma_gnss,
                    nd_gnss(rng) * cfg.sigma_gnss,
                    nd_gnss(rng) * cfg.sigma_gnss);
        f.gnss_pos  = out.truth[k].p + noise;
        f.gnss_std  = vect3(0.5, 0.5, 1.0);   /* 观测标准差 */

        /* 预积分器（末帧无 pre_to_next） */
        if (k + 1 < cfg.N_kf) {
            PreintegrationParam pp;
            pp.gyr_arw      = 1.0e-3;
            pp.acc_vrw      = 1.0e-3;
            pp.gyr_bias_std = 1.0e-4;
            pp.acc_bias_std = 1.0e-3;
            pp.corr_time    = 3600.0;

            f.pre_to_next = std::make_shared<Preintegration>(pp);
            f.pre_to_next->setBias(O31, O31);
        }
    }

    /* ---- 3) 累积预积分 ---- */
    for (int k = 0; k + 1 < cfg.N_kf; ++k) {
        auto &preint = *out.frames[k].pre_to_next;
        for (int i = 0; i < cfg.N_imu_per_kf; ++i) {
            ImuMeas m;
            m.dt     = cfg.dt_imu;
            m.dtheta = O31;
            m.dvel   = f_body * cfg.dt_imu;

            if (cfg.sigma_imu > 0.0) {
                m.dtheta += vect3(nd_imu(rng), nd_imu(rng), nd_imu(rng));
                m.dvel   += vect3(nd_imu(rng), nd_imu(rng), nd_imu(rng));
            }
            preint.integration(m);
        }
    }
    return out;
}

/* ============================================================================
 * 组图 + 求解
 * ==========================================================================*/
struct SolveResult {
    ceres::Solver::Summary summary;
    std::vector<TruthSample> est;    /*< 估计状态（从 frames 提取） */
};

SolveResult optimize(std::vector<Frame> &frames, const vect3 &g_n) {
    ceres::Problem problem;

    /* ---- 参数块 ---- */
    for (auto &f : frames) {
        problem.AddParameterBlock(f.pose.data(), NUM_POSE);
        problem.SetManifold(f.pose.data(), new PoseManifold());
        problem.AddParameterBlock(f.mix.data(), NUM_MIX_ODO);
    }

    /* 固定第一帧 pose（gauge fix） */
    problem.SetParameterBlockConstant(frames.front().pose.data());

    /* ---- 预积分因子 ---- */
    for (size_t k = 0; k + 1 < frames.size(); ++k) {
        if (!frames[k].pre_to_next) continue;
        auto *cost = new ceres::AutoDiffCostFunction<PreintResidual,
                                                     15, 7, 10, 7, 10>(
            new PreintResidual(*frames[k].pre_to_next, g_n));
        problem.AddResidualBlock(cost, nullptr,
                                 frames[k].pose.data(),   frames[k].mix.data(),
                                 frames[k+1].pose.data(), frames[k+1].mix.data());
    }

    /* ---- GNSS 位置因子 ---- */
    for (auto &f : frames) {
        if (!f.has_gnss) continue;
        auto *cost = new ceres::AutoDiffCostFunction<GnssPosResidual, 3, 7>(
            new GnssPosResidual(f.gnss_pos, f.gnss_std));
        problem.AddResidualBlock(cost, nullptr, f.pose.data());
    }

    /* ---- 求解 ---- */
    ceres::Solver::Options options;
    options.linear_solver_type           = ceres::DENSE_QR;
    options.trust_region_strategy_type   = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations           = 200;
    options.minimizer_progress_to_stdout = false;
    options.logging_type                 = ceres::SILENT;

    options.function_tolerance         = 1e-10;    // ← 默认 1e-6
    options.gradient_tolerance         = 1e-12;    // ← 默认 1e-10
    options.parameter_tolerance        = 1e-12;    // ← 默认 1e-8

    SolveResult res;
    ceres::Solve(options, &problem, &res.summary);

    std::printf("  [solver] iter=%d/%d  term=%d  cost=%.3e\n",
                (int)res.summary.iterations.size(),
                options.max_num_iterations,
                (int)res.summary.termination_type,
                res.summary.final_cost);
    std::printf("  [solver] msg: %s\n", res.summary.message.c_str());
    std::printf("  [solver] init_cost=%.3e  final_cost=%.3e \n",
                res.summary.initial_cost, res.summary.final_cost);     

    res.est.resize(frames.size());
    for (size_t k = 0; k < frames.size(); ++k) {
        res.est[k].t  = frames[k].time;
        res.est[k].p  = vect3(frames[k].pose[0], frames[k].pose[1], frames[k].pose[2]);
        res.est[k].q  = quat(frames[k].pose[6], frames[k].pose[3],
                             frames[k].pose[4], frames[k].pose[5]);
        res.est[k].v  = vect3(frames[k].mix[0], frames[k].mix[1], frames[k].mix[2]);
        res.est[k].bg = vect3(frames[k].mix[3], frames[k].mix[4], frames[k].mix[5]);
        res.est[k].ba = vect3(frames[k].mix[6], frames[k].mix[7], frames[k].mix[8]);
    }
    return res;
}

/* ============================================================================
 * 验证
 * ==========================================================================*/
struct EvalStats {
    double max_p_err  = 0.0;
    double max_v_err  = 0.0;
    double max_q_err  = 0.0;
    double max_bg_err = 0.0;
    double max_ba_err = 0.0;
    double final_cost = 0.0;
};

EvalStats evaluate(const SimData &sim, const SolveResult &sol) {
    EvalStats s;
    s.final_cost = sol.summary.final_cost;
    for (size_t k = 0; k < sim.truth.size(); ++k) {
        const auto &T = sim.truth[k];
        const auto &E = sol.est[k];

        s.max_p_err  = std::max(s.max_p_err, norm(E.p  - T.p));
        s.max_v_err  = std::max(s.max_v_err, norm(E.v  - T.v));
        s.max_bg_err = std::max(s.max_bg_err, norm(E.bg - T.bg));
        s.max_ba_err = std::max(s.max_ba_err, norm(E.ba - T.ba));

        vect3 dphi = q2rv((~T.q) * E.q);
        s.max_q_err = std::max(s.max_q_err, norm(dphi));
    }
    return s;
}

/* ============================================================================
 * 测试入口
 * ==========================================================================*/
static int g_pass = 0, g_fail = 0;

static void check(bool cond, const char *msg) {
    if (cond) { std::printf("[ OK ] %s\n", msg); ++g_pass; }
    else      { std::printf("[FAIL] %s\n", msg); ++g_fail; }
}

/* ---------------------------------------------------------------------- */
static void test_static_scene() {
    std::printf("\n===== T1: 静止场景（无噪声） =====\n");

    SimConfig cfg;
    cfg.N_kf       = 5;
    cfg.v0         = vect3(0.0, 0.0, 0.0);   /* 静止 */
    cfg.sigma_imu  = 0.0;
    cfg.sigma_gnss = 0.0;

    SimData sim = simulate(cfg);
    SolveResult sol = optimize(sim.frames, cfg.g_n);
    EvalStats st = evaluate(sim, sol);

    std::printf("  final_cost=%.3e\n", st.final_cost);
    std::printf("  max_p_err=%.3e  max_v_err=%.3e  max_q_err=%.3e\n",
                st.max_p_err, st.max_v_err, st.max_q_err);

    check(sol.summary.termination_type == ceres::CONVERGENCE, "Ceres 收敛");
    check(st.max_p_err < 1e-8, "位置误差 < 1e-8");
    check(st.max_v_err < 1e-8, "速度误差 < 1e-8");
    check(st.max_q_err < 1e-8, "姿态误差 < 1e-8");
}

/* ---------------------------------------------------------------------- */
static void test_uniform_motion_noiseless() {
    std::printf("\n===== T2: 匀速直线（无噪声） =====\n");

    SimConfig cfg;
    cfg.v0         = vect3(1.0, 0.5, 0.0);
    cfg.sigma_imu  = 0.0;
    cfg.sigma_gnss = 0.0;

    SimData sim = simulate(cfg);
    SolveResult sol = optimize(sim.frames, cfg.g_n);
    EvalStats st = evaluate(sim, sol);

    std::printf("  final_cost=%.3e\n", st.final_cost);
    std::printf("  max_p_err=%.3e  max_v_err=%.3e\n", st.max_p_err, st.max_v_err);

    for (size_t k = 0; k < sol.est.size(); ++k) {
        std::printf("  k=%zu  p_est=(%8.4f,%8.4f,%8.4f)  v_est=(%8.4f,%8.4f,%8.4f)\n",
                    k, sol.est[k].p.i, sol.est[k].p.j, sol.est[k].p.k,
                       sol.est[k].v.i, sol.est[k].v.j, sol.est[k].v.k);
    }

    check(sol.summary.termination_type == ceres::CONVERGENCE, "Ceres 收敛");
    check(st.max_p_err < 1e-8, "位置误差 < 1e-8");
    check(st.max_v_err < 1e-8, "速度误差 < 1e-8");
}

/* ---------------------------------------------------------------------- */
static void test_noisy_scene() {
    std::printf("\n===== T3: 匀速直线（含噪声） =====\n");

    SimConfig cfg;
    cfg.v0         = vect3(1.0, 0.5, 0.0);
    cfg.sigma_imu  = 1.0e-3;   /* 每步 IMU 噪声 */
    cfg.sigma_gnss = 0.5;      /* GNSS 位置噪声 */

    SimData sim = simulate(cfg);
    SolveResult sol = optimize(sim.frames, cfg.g_n);
    EvalStats st = evaluate(sim, sol);

    std::printf("  final_cost=%.3e\n", st.final_cost);
    std::printf("  max_p_err=%.3e  max_v_err=%.3e\n", st.max_p_err, st.max_v_err);

    for (size_t k = 0; k < sol.est.size(); ++k) {
        std::printf("  k=%zu  p_est=(%8.4f,%8.4f,%8.4f)  v_est=(%8.4f,%8.4f,%8.4f)\n",
                    k, sol.est[k].p.i, sol.est[k].p.j, sol.est[k].p.k,
                       sol.est[k].v.i, sol.est[k].v.j, sol.est[k].v.k);
    }

    check(sol.summary.termination_type == ceres::CONVERGENCE, "Ceres 收敛");
    /* 有噪声时定位误差应与 GNSS 噪声同量级 */
    check(st.max_p_err < 2.0, "位置误差有界（< 2 m）");
    check(st.max_v_err < 1.0, "速度误差有界（< 1 m/s）");
}

/* ---------------------------------------------------------------------- */
static void test_gnss_anchor_recovery() {
    std::printf("\n===== T4: GNSS 锚定恢复 =====\n");
    /* 情景：初值 p 全部设为 0，看 GNSS 因子能否把位置拉回真值 */

    SimConfig cfg;
    cfg.v0         = vect3(1.0, 0.0, 0.0);
    cfg.sigma_imu  = 0.0;
    cfg.sigma_gnss = 0.0;

    SimData sim = simulate(cfg);
    /* 破坏初值：把所有帧位置置零 */
    for (auto &f : sim.frames) {
        f.pose[0] = 0.0;
        f.pose[1] = 0.0;
        f.pose[2] = 0.0;
    }

    SolveResult sol = optimize(sim.frames, cfg.g_n);
    EvalStats st = evaluate(sim, sol);

    std::printf("  final_cost=%.3e  max_p_err=%.3e\n",
                st.final_cost, st.max_p_err);
    check(st.max_p_err < 1e-8, "从零初值恢复位置");
}

/* ---------------------------------------------------------------------- */
static void test_attitude_perturbation() {
    std::printf("\n===== T5: 姿态扰动恢复 =====\n");
    /* 只破坏中间帧 k=2 的姿态：其他帧保持真值姿态 I，
     * 使相邻预积分的相对姿态不一致，从而可观测 */

    SimConfig cfg;
    cfg.v0         = vect3(1.0, 0.5, 0.0);
    cfg.sigma_imu  = 0.0;
    cfg.sigma_gnss = 0.0;

    SimData sim = simulate(cfg);

    /* 只破坏中间帧姿态：绕 z 轴偏 10° */
    const size_t k_bad = sim.frames.size() / 2;   // k = 2
    quat q_bias = rv2q(vect3(0.0, 0.0, 10.0 * DEG));
    {
        auto &f = sim.frames[k_bad];
        quat q(f.pose[6], f.pose[3], f.pose[4], f.pose[5]);
        quat q_new = q * q_bias;
        normlize(&q_new);
        f.pose[3] = q_new.q1;
        f.pose[4] = q_new.q2;
        f.pose[5] = q_new.q3;
        f.pose[6] = q_new.q0;
    }

    SolveResult sol = optimize(sim.frames, cfg.g_n);
    EvalStats st = evaluate(sim, sol);

    std::printf("  final_cost=%.3e  max_q_err=%.3e (rad)  max_p_err=%.3e\n",
                st.final_cost, st.max_q_err, st.max_p_err);
    std::printf("  破坏帧 = k=%zu，姿态偏差 = %.1f°\n", k_bad, 10.0);
    check(sol.summary.termination_type == ceres::CONVERGENCE, "Ceres 收敛");
    check(st.max_q_err < 1e-8, "中间帧姿态从 10° 偏差恢复");
    check(st.max_p_err < 1e-8, "位置不受影响");
}

/* ---------------------------------------------------------------------- */
static void test_acc_bias_observable() {
    std::printf("\n===== T6: 加计零偏可观性 =====\n");
    /* 真值含 ba_z = 0.05 m/s²，在 IMU 测量中叠加 ba·dt，
     * 优化应能从轨迹中恢复 ba */

    SimConfig cfg;
    cfg.v0         = vect3(1.0, 0.0, 0.0);
    cfg.sigma_imu  = 0.0;
    cfg.sigma_gnss = 0.0;
    const vect3 ba_true(0.0, 0.0, 0.05);

    SimData sim = simulate(cfg);
    /* 重放 IMU：把 ba 叠加进 dvel */
    for (int k = 0; k + 1 < cfg.N_kf; ++k) {
        sim.frames[k].pre_to_next = std::make_shared<Preintegration>();
        sim.frames[k].pre_to_next->setBias(O31, O31);
        const vect3 f_body = -cfg.g_n;   /* [0,0,+9.8] */
        for (int i = 0; i < cfg.N_imu_per_kf; ++i) {
            ImuMeas m;
            m.dt     = cfg.dt_imu;
            m.dtheta = O31;
            m.dvel   = (f_body + ba_true) * cfg.dt_imu;
            sim.frames[k].pre_to_next->integration(m);
        }
    }

    SolveResult sol = optimize(sim.frames, cfg.g_n);
    EvalStats st = evaluate(sim, sol);

    std::printf("  final_cost=%.3e  max_ba_err=%.3e  max_p_err=%.3e\n",
                st.final_cost, st.max_ba_err, st.max_p_err);
    check(sol.summary.termination_type == ceres::CONVERGENCE, "Ceres 收敛");
    check(st.max_ba_err < 1e-8, "加计零偏完全恢复");
}

/* ---------------------------------------------------------------------- */
// static void test_gnss_outlier_robustness() {
//     std::printf("\n===== T7: GNSS 异常值鲁棒性 =====\n");
//     /* 在 k=2 帧注入 100m 的 GNSS 粗差，其余帧正常；
//      * 用 Huber 核函数抑制异常值影响 */

//     SimConfig cfg;
//     cfg.v0         = vect3(1.0, 0.0, 0.0);
//     cfg.sigma_imu  = 0.0;
//     cfg.sigma_gnss = 0.0;

//     SimData sim = simulate(cfg);

//     /* 注入 100m 粗差 */
//     sim.frames[2].gnss_pos.i += 100.0;

//     SolveResult sol = optimizeWithHuber(sim.frames, cfg.g_n, 1.0);
//     EvalStats st = evaluate(sim, sol);

//     std::printf("  final_cost=%.3e  max_p_err=%.3e\n",
//                 st.final_cost, st.max_p_err);
//     check(sol.summary.termination_type == ceres::CONVERGENCE, "Ceres 收敛");
//     /* Huber 应抑制粗差，位置误差不应被拉到 100m 量级 */
//     check(st.max_p_err < 5.0, "Huber 抑制 GNSS 粗差（< 5m）");
// }

/* ---------------------------------------------------------------------- */
static void test_long_trajectory() {
    std::printf("\n===== T8: 长轨迹数值稳定性 =====\n");

    SimConfig cfg;
    cfg.N_kf        = 50;         /* 5 帧 → 50 帧 */
    cfg.v0          = vect3(1.0, 0.5, 0.0);
    cfg.sigma_imu   = 0.0;
    cfg.sigma_gnss  = 0.0;

    SimData sim = simulate(cfg);
    SolveResult sol = optimize(sim.frames, cfg.g_n);
    EvalStats st = evaluate(sim, sol);

    std::printf("  final_cost=%.3e  max_p_err=%.3e  max_v_err=%.3e\n",
                st.final_cost, st.max_p_err, st.max_v_err);
    check(sol.summary.termination_type == ceres::CONVERGENCE, "50 帧收敛");
    check(st.max_p_err < 1e-8, "长轨迹位置精度保持");
}

/* ---------------------------------------------------------------------- */
// static void test_gnss_yaw_factor() {
//     std::printf("\n===== T9: 双天线航向因子 =====\n");
//     /* 真值姿态绕 z 轴偏 30°，但初值给 I；
//      * 加入 yaw = 30° 的观测，看姿态能否被拉回 */

//     SimConfig cfg;
//     cfg.v0         = vect3(1.0, 0.0, 0.0);
//     cfg.sigma_imu  = 0.0;
//     cfg.sigma_gnss = 0.0;

//     SimData sim = simulate(cfg);

//     const double yaw_true = 30.0 * DEG;
//     for (auto &f : sim.frames) {
//         /* 真值姿态含 30° yaw，但 pose 参数块仍给 I（初值破坏） */
//         f.has_gnss_yaw  = true;
//         f.gnss_yaw      = yaw_true;
//         f.gnss_yaw_std  = 1.0 * DEG;   /* 观测标准差 1° */
//     }

//     /* 重放 IMU：真值姿态下比力投影会变，但简化为静止姿态下的常值 */
//     SolveResult sol = optimizeWithYaw(sim.frames, cfg.g_n);
//     EvalStats st = evaluate(sim, sol);

//     std::printf("  final_cost=%.3e  max_q_err=%.3e\n",
//                 st.final_cost, st.max_q_err);
//     check(sol.summary.termination_type == ceres::CONVERGENCE, "Ceres 收敛");
//     /* 仅靠 yaw 因子 + GNSS 位置，姿态 z 分量应被拉到 30° 附近 */
//     check(st.max_q_err < 20.0 * DEG, "yaw 因子有效约束姿态");
// }

/* ---------------------------------------------------------------------- */
// static void test_static_zero_velocity() {
//     std::printf("\n===== T10: 静止零速因子 =====\n");
//     /* 真值静止，但初值速度给 0.5 m/s；is_static=true 时零速因子应拉回 */

//     SimConfig cfg;
//     cfg.v0         = vect3(0.0, 0.0, 0.0);
//     cfg.sigma_imu  = 0.0;
//     cfg.sigma_gnss = 0.0;

//     SimData sim = simulate(cfg);
//     for (auto &f : sim.frames) {
//         f.mix[0] = 0.5;   /* 破坏速度初值 */
//         f.is_static = true;
//     }

//     SolveResult sol = optimizeWithStatic(sim.frames, cfg.g_n);
//     EvalStats st = evaluate(sim, sol);

//     std::printf("  final_cost=%.3e  max_v_err=%.3e\n",
//                 st.final_cost, st.max_v_err);
//     check(st.max_v_err < 1e-8, "零速因子拉回静止");
// }



int main() {
    std::printf("========== Graph Optimization 流程测试 ==========\n");

    test_static_scene();
    test_uniform_motion_noiseless();
    test_noisy_scene();
    test_gnss_anchor_recovery();
    test_attitude_perturbation();     // T5
    test_acc_bias_observable();       // T6
    // test_gnss_outlier_robustness();   // T7
    test_long_trajectory();           // T8
    // test_gnss_yaw_factor();           // T9
    // test_static_zero_velocity();      // T10

    std::printf("\n=============================================\n");
    std::printf("通过 %d, 失败 %d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}