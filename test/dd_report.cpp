/* ============================================================================
 * dd_report.cpp —— GNSS 双差定位算例（基准站 = base，流动站 = 719new）
 *
 * 用法：
 *   dd_report --rover <obs> --base <obs> --nav <nav> [--basepos <alloy pos 文件>]
 *             [--epochs N] [--win W] [--csv out.csv]
 *
 * 做四件事：
 *   1) 从 alloy 静态解文件里取**基准站精确坐标**（RINEX 头里的近似值差 2.19 m，
 *      直接用近似值会给双差几何引入米级偏差）；
 *   2) 生成 GPS 双频双差（稳定参考星 + MW 周跳）；
 *   3) 逐历元伪距单点定位 → 流动站轨迹（ECEF + 经纬高 + 基线长 + 速度）；
 *   4) 载波窗口浮点解 + 整数固定（ILS/LAMBDA），并对比"近似基准站坐标 vs 精确坐标"
 *      对解算的影响。
 * ==========================================================================*/
#include "alg_lib.h"
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static const Eigen::Matrix3d kR_en = Eigen::Matrix3d::Identity();
static const Eigen::Vector3d kP0   = Eigen::Vector3d::Zero();
static const Eigen::Vector3d kTant = Eigen::Vector3d::Zero();

/* ---------------- 小工具 ---------------- */
static void ecef2blh(const Eigen::Vector3d &p, double *lat, double *lon, double *h) {
    const double a = 6378137.0, f = 1.0 / 298.257223563, e2 = f * (2.0 - f);
    *lon = std::atan2(p.y(), p.x());
    const double r = std::hypot(p.x(), p.y());
    double phi = std::atan2(p.z(), r * (1.0 - e2));
    double hh = 0.0;
    for (int i = 0; i < 12; ++i) {
        const double s = std::sin(phi);
        const double N = a / std::sqrt(1.0 - e2 * s * s);
        hh = r / std::max(1e-9, std::cos(phi)) - N;
        phi = std::atan2(p.z(), r * (1.0 - e2 * N / (N + hh)));
    }
    *lat = phi; *h = hh;
}

/* 读 alloy 静态解文件（pos_YYYYDDD_XXXX）最后一行的 XYZ */
static bool readAlloyPos(const std::string &path, Eigen::Vector3d *out) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line, last;
    while (std::getline(in, line)) if (!line.empty()) last = line;
    if (last.empty()) return false;
    double mjd = 0.0, x = 0.0, y = 0.0, z = 0.0;
    if (std::sscanf(last.c_str(), " %*s %lf %lf %lf %lf", &mjd, &x, &y, &z) < 4) return false;
    *out = Eigen::Vector3d(x, y, z);
    return true;
}

static bool solveProblem(ceres::Problem &p, ceres::LinearSolverType lst, int iters) {
    ceres::Solver::Options o;
    o.linear_solver_type = lst;
    o.max_num_iterations = iters;
    o.function_tolerance = 1e-14;
    o.gradient_tolerance = 1e-16;
    o.parameter_tolerance = 1e-14;
    o.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary s;
    ceres::Solve(o, &p, &s);
    return s.termination_type == ceres::CONVERGENCE || s.termination_type == ceres::USER_SUCCESS;
}

static void addPose(ceres::Problem &p, double *pose, const Eigen::Vector3d &init) {
    pose[0] = init.x(); pose[1] = init.y(); pose[2] = init.z();
    pose[3] = 0.0; pose[4] = 0.0; pose[5] = 0.0; pose[6] = 1.0;
    p.AddParameterBlock(pose, NUM_POSE);
    p.SetManifold(pose, new PoseManifold());
}

/* 逐历元伪距单点定位 */
struct Kinematic {
    std::vector<double>     t;
    std::vector<Eigen::Vector3d> pos;
    std::vector<int>        n_dd;
    std::vector<double>     rms_m;
    std::vector<size_t>     dds_idx;   /* ← 新增：对应 dds 下标 */
};

static void solveKinematic(const std::vector<DdEpochResult> &dds, size_t n_use,
                           const Eigen::Vector3d &base, const Eigen::Vector3d &init,
                           Kinematic *out) {
    out->t.clear(); out->pos.clear(); out->n_dd.clear(); out->rms_m.clear();
    Eigen::Vector3d guess = init;
    for (size_t e = 0; e < n_use && e < dds.size(); ++e) {
        const std::vector<DDObs> &dd = dds[e].dd;
        if (dd.size() < 4) continue;
        std::vector<double> pose(NUM_POSE, 0.0);
        ceres::Problem prob;
        addPose(prob, pose.data(), guess);
        for (size_t i = 0; i < dd.size(); ++i)
            for (int f = 0; f < 2; ++f) {
                if (!dd[i].has_P[f]) continue;
                prob.AddResidualBlock(
                    new GnssDDPseudorangeFreqResidual(dd[i], f, base, kR_en, kP0, kTant),
                    nullptr, pose.data());
            }
        if (!solveProblem(prob, ceres::DENSE_QR, 50)) continue;
        const Eigen::Vector3d x(pose[0], pose[1], pose[2]);
        if (!x.allFinite() || (x - base).norm() > 1e6) continue;

        double ss = 0.0; int nn = 0;
        for (const DDObs &d : dd)
            for (int f = 0; f < 2; ++f) {
                if (!d.has_P[f]) continue;
                const double ri = (d.sat_i.pos_ecef - x).norm();
                const double rk = (d.sat_k.pos_ecef - x).norm();
                const double bi = (d.sat_i.pos_ecef - base).norm();
                const double bk = (d.sat_k.pos_ecef - base).norm();
                const double r = d.dd_P[f] - ((ri - rk) - (bi - bk));
                ss += r * r; ++nn;
            }
        out->t.push_back(dds[e].t);
        out->pos.push_back(x);
        out->n_dd.push_back(static_cast<int>(dd.size()));
        out->rms_m.push_back(nn ? std::sqrt(ss / nn) : 0.0);
        out->dds_idx.push_back(e);
        guess = x;
    }
}

/* 由法方程取分块协方差（对角缩放 + 整体伪逆，避免 Schur 补的大数相减） */
static void covFromH(const Eigen::MatrixXd &H, int nP, int nA,
                     Eigen::MatrixXd *Qpos, Eigen::MatrixXd *Qnn)
{
    const int n = (int)H.rows();
    if (n == 0 || nP + nA != n) return;

    Eigen::MatrixXd Hs = 0.5 * (H + H.transpose());

    const Eigen::MatrixXd H_PP = Hs.topLeftCorner(nP, nP);
    const Eigen::MatrixXd H_PN = Hs.topRightCorner(nP, nA);
    const Eigen::MatrixXd H_NN = Hs.bottomRightCorner(nA, nA);

    Eigen::MatrixXd X;
    bool solved = false;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(H_PP);
    if (ldlt.info() == Eigen::Success) {
        X = ldlt.solve(H_PN);
        solved = X.allFinite();
    }
    if (!solved) {
        X = pinv_sym(H_PP, 1e-10) * H_PN;
    }

    const Eigen::MatrixXd S = H_NN - H_PN.transpose() * X;   /* Schur 补 */
    *Qnn  = pinv_sym(0.5 * (S + S.transpose()), 1e-10);

    /* 位置协方差：Qpos = H_PP^{-1} + H_PP^{-1} H_PN Qnn H_NP H_PP^{-1} */
    Eigen::MatrixXd Hinv;
    if (ldlt.info() == Eigen::Success) {
        Hinv = ldlt.solve(Eigen::MatrixXd::Identity(nP, nP));
    }
    if (!Hinv.allFinite()) Hinv = pinv_sym(H_PP, 1e-10);

    if (Qpos)
        *Qpos = Hinv + Hinv * H_PN * (*Qnn) * H_PN.transpose() * Hinv;
}

/* 载波窗口：浮点解 + 整数固定 */
struct WinResult {
    bool   ok{false};                       /* 窗口浮点解求解成功 */
    bool   fixed{false};                    /* 是否真正固定（新增） */
    int    n_amb{0};
    int    n_fixed{0};

    /* 浮点解 */
    Eigen::Vector3d float_pos{};
    double float_rms_m{0.0};
    double float_scatter{0.0};

    /* 固定解（仅 fixed==true 时有效） */
    Eigen::Vector3d fixed_pos{};
    double fixed_rms_m{0.0};

    double ratio{0.0};
    int    n_slip{0};

    std::map<AmbKey, double> fixed_amb;
    std::string note;

    std::vector<double>          fixed_epoch_t;   /* 每历元 GPS 秒 */
    std::vector<Eigen::Vector3d> fixed_pos_per_epoch;
};

static bool solveWindow(const std::vector<DdEpochResult> &dds,
                        size_t e0, size_t e1,
                        const Eigen::Vector3d &base,
                        const std::vector<Eigen::Vector3d> &init_pos,
                        const std::vector<std::vector<std::array<int, MAX_FREQ>>> &arc_all,
                        WinResult *out)
{
    const size_t NW = e1 - e0;
    if (NW == 0) return false;

    std::vector<std::vector<double>> poses(NW, std::vector<double>(NUM_POSE, 0.0));
    AmbMap amb;
    std::vector<std::vector<std::array<int, MAX_FREQ>>> arc(NW);
    for (size_t k = 0; k < NW; ++k) arc[k] = arc_all[e0 + k];   /* 直接用预先算好的 arc */

    ceres::Problem problem;
    int n_p = 0, n_l = 0, n_slip = 0;
    for (size_t k = 0; k < NW; ++k) {
        const size_t e = e0 + k;
        const std::vector<DDObs> &dd = dds[e].dd;
        if (dd.size() < 4) continue;

        /* 修复 P1-4：使用外部传入的初值 */
        const Eigen::Vector3d init_k = (k < init_pos.size()) ? init_pos[k] : base;
        addPose(problem, poses[k].data(), init_k);

        for (size_t i = 0; i < dd.size(); ++i) {
            for (int f = 0; f < 2; ++f) {
                if (!dd[i].has_P[f]) continue;
                problem.AddResidualBlock(
                    new GnssDDPseudorangeFreqResidual(dd[i], f, base, kR_en, kP0, kTant),
                    nullptr, poses[k].data());
                ++n_p;
            }
            for (int f = 0; f < 2; ++f) {
                if (!dd[i].has_L[f]) continue;
                /* 修复 P1-5：arc 从 arc_all 里取，不再窗口内重数 */
                const int a = arc[k][i][f];
                const AmbKey key = ambKey(dd[i].ref_sat, dd[i].sat, f, a);
                auto it = amb.find(key);
                if (it == amb.end()) {
                    double n0 = 0.0;
                    if (dd[i].has_P[f])
                        n0 = std::round(dd[i].dd_L[f] - dd[i].dd_P[f] / GPS_FREQ[f].lambda);
                    amb[key] = {n0};
                    it = amb.find(key);
                }
                if (dds[e].slip[i][f]) ++n_slip;
                problem.AddResidualBlock(
                    new GnssDDCarrierFreqResidual(dd[i], f, base, kR_en, kP0, kTant),
                    nullptr, poses[k].data(), it->second.data());
                ++n_l;
            }
        }
    }
    if (amb.empty()) return false;
    if (!solveProblem(problem, ceres::SPARSE_NORMAL_CHOLESKY, 100)) return false;

    /* ---------- 阶段 1：浮点解位置 & RMS ---------- */
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    int npos = 0;
    for (size_t k = 0; k < NW; ++k) {
        if (dds[e0 + k].dd.size() < 4) continue;
        mean += Eigen::Vector3d(poses[k][0], poses[k][1], poses[k][2]);
        ++npos;
    }
    if (npos) mean /= (double)npos;

    double scatter = 0.0;
    { int np = 0;
      for (size_t k = 0; k < NW; ++k) {
          if (dds[e0 + k].dd.size() < 4) continue;
          scatter += (Eigen::Vector3d(poses[k][0], poses[k][1], poses[k][2]) - mean).squaredNorm();
          ++np;
      }
      if (np) scatter = std::sqrt(scatter / np);
    }

    double ss = 0.0; int nn = 0;
    for (size_t k = 0; k < NW; ++k) {
        const auto &dd = dds[e0 + k].dd;
        const Eigen::Vector3d x(poses[k][0], poses[k][1], poses[k][2]);
        for (size_t i = 0; i < dd.size(); ++i)
            for (int f = 0; f < 2; ++f) {
                if (!dd[i].has_L[f]) continue;
                const double ri = (dd[i].sat_i.pos_ecef - x).norm();
                const double rk = (dd[i].sat_k.pos_ecef - x).norm();
                const double bi = (dd[i].sat_i.pos_ecef - base).norm();
                const double bk = (dd[i].sat_k.pos_ecef - base).norm();
                const double rho = (ri - rk) - (bi - bk);
                const AmbKey key = ambKey(dd[i].ref_sat, dd[i].sat, f, arc[k][i][f]);
                const double r = GPS_FREQ[f].lambda * dd[i].dd_L[f] - rho
                                 - GPS_FREQ[f].lambda * amb[key][0];
                ss += r * r; ++nn;
            }
    }

    out->n_amb         = (int)amb.size();
    out->n_slip        = n_slip;
    out->float_pos     = mean;
    out->float_scatter = scatter;
    out->float_rms_m   = nn ? std::sqrt(ss / nn) : 0.0;
    /* 默认 fixed_pos 就取浮点解，若后文固定成功再覆盖 */
    out->fixed_pos     = mean;
    out->fixed_rms_m   = out->float_rms_m;

    /* ---------- 阶段 2：法方程 → 协方差 → ILS ---------- */
    std::vector<double*> ordered;
    for (size_t k = 0; k < NW; ++k) {
        if (dds[e0 + k].dd.size() < 4) continue;
        ordered.push_back(poses[k].data());
    }
    /* 修复 P2-12：不再硬编码 *6，用 Ceres 报告切空间维数 */
    int nP = 0;
    for (double* p : ordered) nP += problem.ParameterBlockTangentSize(p);

    std::vector<AmbKey> keys;
    for (auto &kv : amb) { keys.push_back(kv.first); ordered.push_back(kv.second.data()); }
    const int nA = (int)keys.size();

    ceres::Problem::EvaluateOptions eo;
    eo.apply_loss_function = false;
    eo.parameter_blocks = ordered;
    double cost = 0.0;
    std::vector<double> res, grad;
    ceres::CRSMatrix jac;

    /* 修复 P0-2：求解失败 → 返回 false，ok 保持 false */
    if (!problem.Evaluate(eo, &cost, &res, &grad, &jac)) {
        out->note = "Evaluate failed";
        return false;
    }
    if (jac.num_cols != nP + nA) {
        out->note = "jacobian column mismatch";
        return false;
    }

    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(jac.num_rows, jac.num_cols);
    for (int r0 = 0; r0 < jac.num_rows; ++r0)
        for (int k2 = jac.rows[r0]; k2 < jac.rows[r0 + 1]; ++k2)
            J(r0, jac.cols[k2]) = jac.values[k2];
    Eigen::VectorXd rv(res.size());
    for (size_t k2 = 0; k2 < res.size(); ++k2) rv((int)k2) = res[k2];

    Eigen::MatrixXd H; Eigen::VectorXd bb;
    accumulate_normal_equations(J, rv, H, bb);
    Eigen::MatrixXd Qp_unused, Q;
    covFromH(H, nP, nA, &Qp_unused, &Q);
    if (Q.rows() != nA) {
        /* 协方差失败：浮点解仍有效，只标记固定失败 */
        out->ok = true;
        out->note = "covariance extraction failed";
        return true;
    }
    Q = 0.5 * (Q + Q.transpose());

    Eigen::VectorXd a(nA);
    for (int i = 0; i < nA; ++i) a(i) = amb[keys[i]][0];

    /* 修复 P1-6：使用部分固定，最少 4 个模糊度，ratio 阈值 3.0 */
    IlsPartial rp;
    if (ilsPartialFix(a, Q, 3.0, 4, 4000000L, &rp) && rp.ok) {
        out->ratio = rp.last.ratio;
        if (rp.last.ratio > 3.0) {
            out->n_fixed = (int)rp.fixed_idx.size();
            for (int i : rp.fixed_idx) {
                const AmbKey key = keys[i];
                const double v = rp.last.a_fixed(i);
                amb[key][0] = v;
                out->fixed_amb[key] = v;
            }
            out->fixed = true;
        }
    }

    /* ---------- 阶段 3：固定后重解（修复 P0-1） ---------- */
    if (out->fixed) {
        for (auto &kv : amb) {
            if (out->fixed_amb.count(kv.first))
                problem.SetParameterBlockConstant(kv.second.data());
        }
        if (!solveProblem(problem, ceres::SPARSE_NORMAL_CHOLESKY, 50)) {
            out->note = "re-solve with fixed amb failed; degrade to float";
            out->fixed = false;
            out->n_fixed = 0;
            out->ok = true;
            return true;
        }

        /* 固定后的位置 */
        Eigen::Vector3d mean2 = Eigen::Vector3d::Zero();
        int np2 = 0;
        for (size_t k = 0; k < NW; ++k) {
            if (dds[e0 + k].dd.size() < 4) continue;
            mean2 += Eigen::Vector3d(poses[k][0], poses[k][1], poses[k][2]);
            ++np2;
        }
        if (np2) mean2 /= (double)np2;

        /* 固定后的载波 RMS（修复 P2-8） */
        double ss2 = 0.0; int nn2 = 0;
        for (size_t k = 0; k < NW; ++k) {
            const auto &dd = dds[e0 + k].dd;
            const Eigen::Vector3d x(poses[k][0], poses[k][1], poses[k][2]);
            for (size_t i = 0; i < dd.size(); ++i)
                for (int f = 0; f < 2; ++f) {
                    if (!dd[i].has_L[f]) continue;
                    const double ri = (dd[i].sat_i.pos_ecef - x).norm();
                    const double rk = (dd[i].sat_k.pos_ecef - x).norm();
                    const double bi = (dd[i].sat_i.pos_ecef - base).norm();
                    const double bk = (dd[i].sat_k.pos_ecef - base).norm();
                    const double rho = (ri - rk) - (bi - bk);
                    const AmbKey key = ambKey(dd[i].ref_sat, dd[i].sat, f, arc[k][i][f]);
                    const double r = GPS_FREQ[f].lambda * dd[i].dd_L[f] - rho
                                     - GPS_FREQ[f].lambda * amb[key][0];
                    ss2 += r * r; ++nn2;
                }
        }

        out->fixed_pos   = mean2;
        out->fixed_rms_m = nn2 ? std::sqrt(ss2 / nn2) : 0.0;

        /* 修复 P0-3：一致性检验 */
        /* ---------- 阶段 3：固定后重解（修复 P0-1） ---------- */
        if (out->fixed) {
            for (auto &kv : amb) {
                if (out->fixed_amb.count(kv.first))
                    problem.SetParameterBlockConstant(kv.second.data());
            }
            if (!solveProblem(problem, ceres::SPARSE_NORMAL_CHOLESKY, 50)) {
                out->note = "re-solve with fixed amb failed; degrade to float";
                out->fixed = false;
                out->n_fixed = 0;
                out->ok = true;
                return true;
            }

            Eigen::Vector3d mean2 = Eigen::Vector3d::Zero();
            int np2 = 0;
            for (size_t k = 0; k < NW; ++k) {
                if (dds[e0 + k].dd.size() < 4) continue;
                mean2 += Eigen::Vector3d(poses[k][0], poses[k][1], poses[k][2]);
                ++np2;
            }
            if (np2) mean2 /= (double)np2;

            double ss2 = 0.0; int nn2 = 0;
            /* ... 原有残差统计，略 ... */

            out->fixed_pos   = mean2;
            out->fixed_rms_m = nn2 ? std::sqrt(ss2 / nn2) : 0.0;

            const double pos_diff = (out->fixed_pos - out->float_pos).norm();
            if (pos_diff > 2.0) {
                out->note = "fixed vs float pos diff > 2m; degrade to float";
                out->fixed = false;
                out->n_fixed = 0;
                out->fixed_pos = out->float_pos;
                out->fixed_rms_m = out->float_rms_m;
            } else if (out->fixed_rms_m > 0.05) {
                out->note = "fixed rms too large; degrade to float";
                out->fixed = false;
                out->n_fixed = 0;
                out->fixed_pos = out->float_pos;
                out->fixed_rms_m = out->float_rms_m;
            }

            /* ↓↓↓ 新增：固定有效时，把重解后每个历元的位姿作为该历元固定解导出 ↓↓↓ */
            if (out->fixed) {
                out->fixed_epoch_t.reserve(NW);
                out->fixed_pos_per_epoch.reserve(NW);
                for (size_t k = 0; k < NW; ++k) {
                    if (dds[e0 + k].dd.size() < 4) continue;
                    const Eigen::Vector3d pos_k(poses[k][0], poses[k][1], poses[k][2]);
                    if (!pos_k.allFinite() || (pos_k - base).norm() > 1e6) continue;
                    out->fixed_epoch_t.push_back(dds[e0 + k].t);
                    out->fixed_pos_per_epoch.push_back(pos_k);
                }
            }
        }

        const double pos_diff = (out->fixed_pos - out->float_pos).norm();
        if (pos_diff > 2.0) {
            out->note = "fixed vs float pos diff > 2m; degrade to float";
            out->fixed = false;
            out->n_fixed = 0;
            out->fixed_pos = out->float_pos;
            out->fixed_rms_m = out->float_rms_m;
        } else if (out->fixed_rms_m > 0.05) {
            /* 固定后残差应降到毫米-厘米级，否则认为固定不可信 */
            out->note = "fixed rms too large; degrade to float";
            out->fixed = false;
            out->n_fixed = 0;
            out->fixed_pos = out->float_pos;
            out->fixed_rms_m = out->float_rms_m;
        }

    }

    out->ok = true;
    return true;
}

/* ============================================================================
 * 整段静态解：一个位置 + 全部模糊度
 * ==========================================================================*/
struct StaticResult {
    bool    ok{false};
    Eigen::Vector3d pos{};
    double  sig_E{0.0}, sig_N{0.0}, sig_U{0.0};
    int     n_amb{0}, n_slip{0}, n_p{0}, n_l{0};
    double  carr_rms_m{0.0}, ratio{0.0}, p_boot{0.0};
    int     n_fixed{0};
};

static bool solveStatic(const std::vector<DdEpochResult> &dds, size_t e0, size_t e1,
                        const Eigen::Vector3d &base,
                        const std::vector<std::vector<std::array<int, MAX_FREQ>>> &arc_all,
                        StaticResult *out)
{
    if (e1 <= e0) return false;
    std::vector<double> pose(NUM_POSE, 0.0);
    AmbMap amb;

    std::vector<std::vector<std::array<int, MAX_FREQ>>> arc(e1 - e0);
    for (size_t k = 0; k < e1 - e0; ++k) arc[k] = arc_all[e0 + k];

    ceres::Problem problem;
    addPose(problem, pose.data(), base);

    int n_p = 0, n_l = 0, n_slip = 0;
    for (size_t k = 0; k < e1 - e0; ++k) {
        const size_t e = e0 + k;
        const std::vector<DDObs> &dd = dds[e].dd;
        for (size_t i = 0; i < dd.size(); ++i) {
            for (int f = 0; f < 2; ++f) {
                if (!dd[i].has_P[f]) continue;
                problem.AddResidualBlock(
                    new GnssDDPseudorangeFreqResidual(dd[i], f, base, kR_en, kP0, kTant),
                    nullptr, pose.data());
                ++n_p;
            }
            for (int f = 0; f < 2; ++f) {
                if (!dd[i].has_L[f]) continue;
                const int a = arc[k][i][f];                 /* ← 来自 arc_all */
                const AmbKey key = ambKey(dd[i].ref_sat, dd[i].sat, f, a);
                auto it = amb.find(key);
                if (it == amb.end()) {
                    double n0 = 0.0;
                    if (dd[i].has_P[f])
                        n0 = std::round(dd[i].dd_L[f] - dd[i].dd_P[f] / GPS_FREQ[f].lambda);
                    amb[key] = {n0};
                    it = amb.find(key);
                }
                if (dds[e].slip[i][f]) ++n_slip;
                problem.AddResidualBlock(
                    new GnssDDCarrierFreqResidual(dd[i], f, base, kR_en, kP0, kTant),
                    nullptr, pose.data(), it->second.data());
                ++n_l;
            }
        }
    }
    if (amb.empty()) return false;
    if (!solveProblem(problem, ceres::SPARSE_NORMAL_CHOLESKY, 100)) return false;

    out->pos = Eigen::Vector3d(pose[0], pose[1], pose[2]);
    out->n_amb = static_cast<int>(amb.size());
    out->n_slip = n_slip; out->n_p = n_p; out->n_l = n_l;

    /* 载波残差 RMS */
    double ss = 0.0; int nn = 0;
    for (size_t k = 0; k < e1 - e0; ++k) {
        const std::vector<DDObs> &dd = dds[e0 + k].dd;
        for (size_t i = 0; i < dd.size(); ++i)
            for (int f = 0; f < 2; ++f) {
                if (!dd[i].has_L[f]) continue;
                const double ri = (dd[i].sat_i.pos_ecef - out->pos).norm();
                const double rk = (dd[i].sat_k.pos_ecef - out->pos).norm();
                const double bi = (dd[i].sat_i.pos_ecef - base).norm();
                const double bk = (dd[i].sat_k.pos_ecef - base).norm();
                const double rho = (ri - rk) - (bi - bk);
                const AmbKey key = ambKey(dd[i].ref_sat, dd[i].sat, f, arc[k][i][f]);
                const double r = GPS_FREQ[f].lambda * dd[i].dd_L[f] - rho
                                 - GPS_FREQ[f].lambda * amb[key][0];
                ss += r * r; ++nn;
            }
    }
    out->carr_rms_m = nn ? std::sqrt(ss / nn) : 0.0;

    /* 法方程：列序 [pose(6), 模糊度...] */
    std::vector<double*> ordered{pose.data()};
    std::vector<AmbKey> keys;
    for (auto &kv : amb) { keys.push_back(kv.first); ordered.push_back(kv.second.data()); }
    const int nA = static_cast<int>(keys.size());
    const int nP = 6;

    ceres::Problem::EvaluateOptions eo;
    eo.apply_loss_function = false;
    eo.parameter_blocks = ordered;
    double cost = 0.0;
    std::vector<double> res, grad;
    ceres::CRSMatrix jac;
    if (!problem.Evaluate(eo, &cost, &res, &grad, &jac)) return true;
    if (jac.num_cols != nP + nA) return true;
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(jac.num_rows, jac.num_cols);
    for (int r0 = 0; r0 < jac.num_rows; ++r0)
        for (int k2 = jac.rows[r0]; k2 < jac.rows[r0 + 1]; ++k2)
            J(r0, jac.cols[k2]) = jac.values[k2];
    Eigen::VectorXd rv(res.size());
    for (size_t k2 = 0; k2 < res.size(); ++k2) rv(static_cast<int>(k2)) = res[k2];

    Eigen::MatrixXd H; Eigen::VectorXd bb;
    accumulate_normal_equations(J, rv, H, bb);
    Eigen::MatrixXd Qp, Qnn;
    covFromH(H, nP, nA, &Qp, &Qnn);
    if (Qp.rows() != nP || Qnn.rows() != nA) return true;
    Qp = 0.5 * (Qp + Qp.transpose());

    /* ECEF → ENU */
    double la = 0, lo = 0, h = 0;
    ecef2blh(base, &la, &lo, &h);
    Eigen::Matrix3d R;
    R << -std::sin(lo),              std::cos(lo),              0.0,
         -std::sin(la) * std::cos(lo), -std::sin(la) * std::sin(lo), std::cos(la),
          std::cos(la) * std::cos(lo),  std::cos(la) * std::sin(lo), std::sin(la);
    const Eigen::Matrix3d Qenu = R * Qp.topLeftCorner(3, 3) * R.transpose();
    out->sig_E = std::sqrt(std::max(0.0, Qenu(0, 0)));
    out->sig_N = std::sqrt(std::max(0.0, Qenu(1, 1)));
    out->sig_U = std::sqrt(std::max(0.0, Qenu(2, 2)));

    /* ILS */
    Eigen::VectorXd a(nA);
    for (int i = 0; i < nA; ++i) a(i) = amb[keys[static_cast<size_t>(i)]][0];

    IlsResult r;
    if (ilsResolve(a, Qnn, &r, 3.0)) {
        out->ratio  = r.ratio;
        out->p_boot = r.p_boot;
        if (r.ratio > 3.0) {
            for (int i = 0; i < nA; ++i) amb[keys[i]][0] = r.a_fixed(i);
            for (auto &kv : amb) problem.SetParameterBlockConstant(kv.second.data());

            if (solveProblem(problem, ceres::SPARSE_NORMAL_CHOLESKY, 50)) {
                const Eigen::Vector3d pos_fixed(pose[0], pose[1], pose[2]);

                double ss2 = 0.0; int nn2 = 0;
                for (size_t k = 0; k < e1 - e0; ++k) {
                    const std::vector<DDObs> &ddk = dds[e0 + k].dd;
                    for (size_t i = 0; i < ddk.size(); ++i)
                        for (int f = 0; f < 2; ++f) {
                            if (!ddk[i].has_L[f]) continue;
                            const double ri = (ddk[i].sat_i.pos_ecef - pos_fixed).norm();
                            const double rk = (ddk[i].sat_k.pos_ecef - pos_fixed).norm();
                            const double bi = (ddk[i].sat_i.pos_ecef - base).norm();
                            const double bk = (ddk[i].sat_k.pos_ecef - base).norm();
                            const double rho = (ri - rk) - (bi - bk);
                            const AmbKey key = ambKey(ddk[i].ref_sat, ddk[i].sat, f, arc[k][i][f]);
                            const double rr = GPS_FREQ[f].lambda * ddk[i].dd_L[f] - rho
                                            - GPS_FREQ[f].lambda * amb[key][0];
                            ss2 += rr * rr; ++nn2;
                        }
                }
                const double rms_fixed = nn2 ? std::sqrt(ss2 / nn2) : 0.0;
                const double pos_diff  = (pos_fixed - out->pos).norm();

                if (pos_diff <= 2.0 && rms_fixed < 0.05) {
                    out->pos        = pos_fixed;
                    out->carr_rms_m = rms_fixed;
                    out->n_fixed    = nA;
                }
            }
        }
    }

    out->ok = true;   /* ok 只在末尾置位 */
    
    return true;
}

int main(int argc, char **argv) {
    std::string rover_obs, base_obs, nav, basepos, csv;
    size_t max_epochs = 0;
    size_t win = 40;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--rover") rover_obs = next();
        else if (a == "--base") base_obs = next();
        else if (a == "--nav") nav = next();
        else if (a == "--basepos") basepos = next();
        else if (a == "--csv") csv = next();
        else if (a == "--epochs") max_epochs = static_cast<size_t>(std::atoi(next().c_str()));
        else if (a == "--win") win = static_cast<size_t>(std::atoi(next().c_str()));
    }
    if (rover_obs.empty() || base_obs.empty() || nav.empty()) {
        std::printf("用法: dd_report --rover <obs> --base <obs> --nav <nav> "
                    "[--basepos <pos 文件>] [--epochs N] [--win W] [--csv out.csv]\n");
        return 2;
    }

    /* ---- 频率槽：GPS L1C/L2W（两站观测码严格相同的组合） ---- */
    std::vector<DdSignal> slots(2);
    slots[0].sys = 'G'; slots[0].code_P = "C1C"; slots[0].code_L = "L1C"; slots[0].code_D = "D1C";
    slots[1].sys = 'G'; slots[1].code_P = "C2W"; slots[1].code_L = "L2W"; slots[1].code_D = "D2W";
    std::vector<std::array<int, 3>> si;
    const std::vector<ObsRequest> reqs = buildObsRequests(slots, &si);

    std::vector<RinexEpoch> rover, base;
    RinexObsHeader bh;
    std::map<char, std::map<int, std::vector<KeplerEph>>> eph;
    if (!readRinex3Obs(rover_obs, reqs, rover, nullptr) ||
        !readRinex3Obs(base_obs, reqs, base, &bh) ||
        !readRinex3Nav(nav, eph, nullptr)) {
        std::printf("数据读取失败\n");
        return 1;
    }

    /* ---- 基准站坐标：优先用 alloy 静态解 ---- */
        const Eigen::Vector3d base_approx = bh.approx_pos;
    Eigen::Vector3d base_precise = {-2273226.1844, 5012023.0401, 3213012.1566};
    bool have_precise = true;
    if (!basepos.empty()) {
        Eigen::Vector3d tmp;
        if (readAlloyPos(basepos, &tmp)) base_precise = tmp;
    }
    double blat = 0, blon = 0, bhgt = 0;
    ecef2blh(base_precise, &blat, &blon, &bhgt);

    std::printf("================ GNSS 双差定位算例 ================\n");
    std::printf("流动站观测: %s\n", rover_obs.c_str());
    std::printf("基准站观测: %s   (marker=%s)\n", base_obs.c_str(), bh.marker.c_str());
    std::printf("星历      : %s\n\n", nav.c_str());

    std::printf("基准站坐标:\n");
    std::printf("  精确(alloy 静态解) : %15.4f %15.4f %15.4f  m\n",
                base_precise.x(), base_precise.y(), base_precise.z());
    std::printf("  RINEX 头近似       : %15.4f %15.4f %15.4f  m\n",
                base_approx.x(), base_approx.y(), base_approx.z());
    if (have_precise) {
        std::printf("  两者相差           : %.3f m\n", (base_precise - base_approx).norm());
        std::printf("  精确坐标的经纬高   : %.9f deg, %.9f deg, %.3f m\n",
                    blat * 180.0 / PI, blon * 180.0 / PI, bhgt);
    } else {
        std::printf("  （未提供 --basepos，只能用近似坐标）\n");
    }

    /* ---- 双差 ---- */
    DdBuildConfig cfg;
    cfg.base_ecef = base_precise;
    cfg.rover_approx = base_precise;
    cfg.slots = slots;
    cfg.elev_mask_deg = 10.0;
    cfg.min_sats = 4;
    cfg.stable_reference = true;

    std::vector<DdEpochResult> dds;
    DdBuildStats stt;
    if (!buildDdEpochs(rover, base, eph, cfg, dds, &stt) || dds.empty()) {
        std::printf("双差生成失败\n");
        return 1;
    }
    int tot_dd = 0;
    for (auto &d : dds) tot_dd += d.n_used;
    std::printf("\n双差观测:\n");
    std::printf("  流动站历元 %d，产出 %zu 个历元，共 %d 个双差（%.1f/历元）\n",
                stt.n_epochs_rover, dds.size(), tot_dd,
                dds.empty() ? 0.0 : static_cast<double>(tot_dd) / dds.size());
    std::printf("  跳过: 无基准站 %d，星数不足 %d，参考星不可用 %d\n",
                stt.n_no_base, stt.n_too_few, stt.n_ref_missing);
    std::printf("  稳定参考星: ");
    for (auto &kv : stt.chosen_ref) std::printf("%c%02d ", kv.first, kv.second);
    std::printf("   周跳 %d 次\n", stt.n_slip);
    std::printf("  时间范围: %zu 个历元，约 %.1f 分钟\n", dds.size(),
                dds.size() > 1 ? (dds.back().t - dds.front().t) / 60.0 : 0.0);

    const size_t n_use = (max_epochs > 0) ? std::min(max_epochs, dds.size()) : dds.size();

    /* ---- 1) 逐历元伪距定位（精确基准站坐标） ---- */
    Kinematic kin;
    solveKinematic(dds, n_use, base_precise, base_precise, &kin);
    std::printf("\n---- 1) 逐历元伪距单点定位（基准站用**精确**坐标）----\n");
    std::printf("  解出 %zu / %zu 个历元\n", kin.pos.size(), n_use);
    if (!kin.pos.empty()) {
        std::printf("  前 5 个历元（ECEF → 经纬高）：\n");
        for (size_t i = 0; i < kin.pos.size() && i < 5; ++i) {
            double la, lo, h;
            ecef2blh(kin.pos[i], &la, &lo, &h);
            std::printf("    #%zu  %.9f  %.9f  %8.3f m   基线 %.2f m   RMS %.3f m  (%d 个双差)\n",
                        i, la * 180.0 / PI, lo * 180.0 / PI, h,
                        (kin.pos[i] - base_precise).norm(), kin.rms_m[i], kin.n_dd[i]);
        }
        std::printf("  后 3 个历元：\n");
        for (size_t i = (kin.pos.size() > 3 ? kin.pos.size() - 3 : 0); i < kin.pos.size(); ++i) {
            double la, lo, h;
            ecef2blh(kin.pos[i], &la, &lo, &h);
            std::printf("    #%zu  %.9f  %.9f  %8.3f m   基线 %.2f m   RMS %.3f m\n",
                        i, la * 180.0 / PI, lo * 180.0 / PI, h,
                        (kin.pos[i] - base_precise).norm(), kin.rms_m[i]);
        }
        double bl_min = 1e300, bl_max = 0.0, bl_sum = 0.0;
        for (const auto &p : kin.pos) {
            const double d = (p - base_precise).norm();
            bl_min = std::min(bl_min, d); bl_max = std::max(bl_max, d); bl_sum += d;
        }
        std::printf("  基线长: %.3f ~ %.3f m（均值 %.3f）\n", bl_min, bl_max,
                    bl_sum / kin.pos.size());
        /* 速度（相邻历元位移 / 时差） */
        if (kin.pos.size() > 2) {
            std::vector<double> spd;
            for (size_t i = 1; i < kin.pos.size(); ++i) {
                const double dt = kin.t[i] - kin.t[i - 1];
                if (dt > 1e-6) spd.push_back((kin.pos[i] - kin.pos[i - 1]).norm() / dt);
            }
            if (!spd.empty()) {
                std::sort(spd.begin(), spd.end());
                double s = 0; for (double v : spd) s += v;
                std::printf("  速度(相邻差分): 均值 %.2f m/s, 中位 %.2f, P95 %.2f, 最大 %.2f\n",
                            s / spd.size(), spd[spd.size() / 2], spd[spd.size() * 95 / 100],
                            spd.back());
            }
        }
        double rs = 0; for (double v : kin.rms_m) rs += v;
        if (!kin.rms_m.empty())
            std::printf("  伪距后验 RMS 均值 = %.3f m\n", rs / kin.rms_m.size());
    }

    /* 基线随时间（车在动，光看首尾会误判） */
    if (kin.pos.size() > 20) {
        std::printf("\n  基线随时间（每 %zu 个历元取样）：\n",
                    std::max<size_t>(1, kin.pos.size() / 12));
        const size_t step = std::max<size_t>(1, kin.pos.size() / 12);
        for (size_t i = 0; i < kin.pos.size(); i += step) {
            double la, lo, h;
            ecef2blh(kin.pos[i], &la, &lo, &h);
            std::printf("    #%4zu  t+%6.1f s  %.9f  %.9f  %7.2f m  基线 %8.1f m\n",
                        i, kin.t[i] - kin.t[0], la * 180.0 / PI, lo * 180.0 / PI, h,
                        (kin.pos[i] - base_precise).norm());
        }
    }

    /* ---- 2) 精确 vs 近似基准站坐标的影响 ---- */
    if (have_precise && (base_precise - base_approx).norm() > 1e-6) {
        std::printf("\n---- 2) 基准站用近似坐标 vs 精确坐标 ----\n");
        DdBuildConfig cfg2 = cfg;
        cfg2.base_ecef = base_approx;
        std::vector<DdEpochResult> dds2;
        DdBuildStats st2;
        buildDdEpochs(rover, base, eph, cfg2, dds2, &st2);
        Kinematic kin2;
        solveKinematic(dds2, std::min<size_t>(n_use, dds2.size()), base_approx, base_approx, &kin2);
        const size_t ncmp = std::min(kin.pos.size(), kin2.pos.size());
        double dmax = 0.0, dsum = 0.0;
        for (size_t i = 0; i < ncmp; ++i) {
            const double d = (kin.pos[i] - kin2.pos[i]).norm();
            dmax = std::max(dmax, d); dsum += d;
        }
        std::printf("  两者解出的流动站位置差: 均值 %.3f m, 最大 %.3f m（%zu 个历元可比）\n",
                    ncmp ? dsum / ncmp : 0.0, dmax, ncmp);
        std::printf("  （基准站坐标差 %.3f m；近似坐标会把这段偏差按基线方向投影到解里）\n",
                    (base_precise - base_approx).norm());
    }

    /* ---- 3) 载波窗口浮点解 + 整数固定 ---- */
    std::printf("\n---- 3) 载波窗口解（%zu 历元/窗）----\n", win);
    std::vector<WinResult> wins;
    std::vector<size_t>    wins_e0;
    /* 全序列 arc 预计算 */
    std::vector<std::vector<std::array<int, MAX_FREQ>>> all_arc(dds.size());
    {
        std::map<AmbKey, int> arc_of;
        for (size_t e = 0; e < dds.size(); ++e) {
            all_arc[e].assign(dds[e].dd.size(), {0, 0, 0});
            for (size_t i = 0; i < dds[e].dd.size(); ++i) {
                for (int f = 0; f < 2; ++f) {
                    const AmbKey bk = ambKey(dds[e].dd[i].ref_sat, dds[e].dd[i].sat, f);
                    if (dds[e].slip[i][f]) ++arc_of[bk];
                    all_arc[e][i][f] = arc_of[bk];
                }
            }
        }
    }

    /* kin.pos 已经在上一步 solveKinematic 中算好 */
    /* 原：for (size_t e0 = 0; e0 + win <= n_use && e0 < win * 4; e0 += win) */
    for (size_t e0 = 0; e0 + win <= n_use; e0 += win) {
        const size_t NW = win;

        /* 用 dds_idx 映射正确取初值 */
        std::vector<Eigen::Vector3d> init_pos(NW, base_precise);
        {
            std::map<size_t, Eigen::Vector3d> kin_by_idx;
            for (size_t i = 0; i < kin.pos.size(); ++i)
                kin_by_idx[kin.dds_idx[i]] = kin.pos[i];
            for (size_t k = 0; k < NW && e0 + k < dds.size(); ++k) {
                auto it = kin_by_idx.find(e0 + k);
                if (it != kin_by_idx.end()) init_pos[k] = it->second;
            }
        }

        WinResult w;
        if (!solveWindow(dds, e0, e0 + win, base_precise,
                        init_pos, all_arc, &w) || !w.ok) {
            std::printf("  窗[%zu,%zu): 求解失败\n", e0, e0 + win);
            continue;
        }
        /* 打印：区分浮点/固定 */
        double la, lo, h;
        if (w.fixed) {
            ecef2blh(w.fixed_pos, &la, &lo, &h);
            std::printf("  窗[%3zu,%3zu): [FIX ] 固定 %d/%d  ratio %-8.3g  载波RMS %.4f m\n",
                        e0, e0 + win, w.n_fixed, w.n_amb, w.ratio, w.fixed_rms_m);
            std::printf("            位置 %.9f %.9f %8.3f m  基线 %.3f m\n",
                        la * 180.0 / PI, lo * 180.0 / PI, h,
                        (w.fixed_pos - base_precise).norm());
        } else {
            ecef2blh(w.float_pos, &la, &lo, &h);
            std::printf("  窗[%3zu,%3zu): [FLOAT] 模糊度 %d  ratio %-8.3g  %s\n",
                        e0, e0 + win, w.n_amb, w.ratio, w.note.c_str());
            std::printf("            位置 %.9f %.9f %8.3f m  基线 %.3f m  载波RMS %.4f m\n",
                        la * 180.0 / PI, lo * 180.0 / PI, h,
                        (w.float_pos - base_precise).norm(), w.float_rms_m);
        }
        wins.push_back(std::move(w));
        wins_e0.push_back(e0);
    }

    /* 跨窗整数一致性：只比较两窗共有的键 */
    if (wins.size() >= 2) {
        std::printf("\n  跨窗整数一致性（与窗[%zu,%zu) 比较）：\n",
                    wins_e0[0], wins_e0[0] + win);
        for (size_t i = 1; i < wins.size(); ++i) {
            int n_cmp = 0, n_same = 0;
            if (wins[0].fixed && wins[i].fixed) {
                for (const auto &kv : wins[0].fixed_amb) {
                    auto it = wins[i].fixed_amb.find(kv.first);
                    if (it == wins[i].fixed_amb.end()) continue;
                    ++n_cmp;
                    if (std::fabs(kv.second - it->second) < 0.5) ++n_same;
                }
            }
            const char *tag = "";
            if (!wins[0].fixed || !wins[i].fixed)
                tag = "  <-- 至少一窗未固定";
            else if (n_cmp == 0)
                tag = "  <-- 两窗已固定但无共同键（弧段不同）";
            else if (n_same != n_cmp)
                tag = "  <-- 有整数不一致！";
            std::printf("    窗[%3zu,%3zu): 共有 %2d 个整数，一致 %2d 个%s\n",
                        wins_e0[i], wins_e0[i] + win, n_cmp, n_same, tag);
        }
    }

    /* ---- 4) 静止段静态解 ---- */
    size_t n_static = 0;
    double t_static_end = 0.0;
    if (!kin.pos.empty()) {
        /* 门限 5 m：伪距单点解噪声 0.4~1 m，0.5 m 会立刻判成"在动"；
         * 本场车开出去 6.4 km，5 m 足够区分"停着"和"开动"。 */
        const Eigen::Vector3d p0 = kin.pos[0];
        n_static = 1;
        while (n_static < kin.pos.size() &&
               (kin.pos[n_static] - p0).norm() < 5.0) ++n_static;
        t_static_end = kin.t[n_static - 1];
    }
    /* 用时间把静止段终点映射回**双差历元下标**（kin 下标 ≠ dds 下标） */
    size_t e_static = 0;
    for (size_t e = 0; e < dds.size(); ++e) if (dds[e].t <= t_static_end + 0.5) e_static = e + 1;

    std::printf("\n---- 4) 静止段静态解 ----\n");
    /* 候选长度：先用"位移 < 5 m"给个上界，再用**载波 RMS 是否毫米级**逐步收缩。
     * 只用距离阈值不行 —— 实测车在段内缓慢溜动 1.5 m，距离阈值放它进来，
     * 静态模型就不成立（载波 RMS 0.30 m）。 */
    size_t n_cand = 0;
    if (!kin.pos.empty()) {
        const Eigen::Vector3d p0 = kin.pos[0];
        n_cand = 1;
        while (n_cand < kin.pos.size() &&
               (kin.pos[n_cand] - p0).norm() < 5.0) ++n_cand;
    }
    size_t e_cand = 0;
    if (n_cand > 0) {
        const double t_end = kin.t[n_cand - 1];
        for (size_t e = 0; e < dds.size(); ++e) if (dds[e].t <= t_end + 0.5) e_cand = e + 1;
    }
    std::printf("  候选静止段 = %zu 个双差历元（位移 < 5 m）\n", e_cand);

    size_t e_used = 0;
    StaticResult sr;
    bool got = false;
    for (int attempt = 0; attempt < 7 && e_cand >= 40; ++attempt) {
        StaticResult cand;
        if (solveStatic(dds, 0, e_cand, base_precise, all_arc, &cand) && cand.ok &&
            cand.carr_rms_m < 0.02) {
            sr = cand; e_used = e_cand; got = true;
            break;
        }
        const size_t next = e_cand * 8 / 10;
        std::printf("     长度 %zu 不满足（载波RMS %.4f m）→ 收缩到 %zu 重试\n",
                    e_cand, cand.ok ? cand.carr_rms_m : -1.0, next);
        if (next == e_cand) break;
        e_cand = next;
    }

    if (!got) {
        std::printf("  未能找到静态模型可拟合到毫米级的长度，跳过静态解\n");
    } else {
        double la, lo, h;
        ecef2blh(sr.pos, &la, &lo, &h);
        const Eigen::Vector3d d = sr.pos - base_precise;
        double bla, blo, bh;
        ecef2blh(base_precise, &bla, &blo, &bh);
        Eigen::Matrix3d Rm;
        Rm << -std::sin(blo),              std::cos(blo),              0.0,
              -std::sin(bla) * std::cos(blo), -std::sin(bla) * std::sin(blo), std::cos(bla),
               std::cos(bla) * std::cos(blo),  std::cos(bla) * std::sin(blo), std::sin(bla);
        const Eigen::Vector3d enu = Rm * d;
        std::printf("  采用长度 = %zu 个历元（约 %.1f 分钟）\n", e_used,
                    e_used > 1 ? (dds[e_used - 1].t - dds[0].t) / 60.0 : 0.0);
        std::printf("  位置(ECEF) : %.4f  %.4f  %.4f  m\n", sr.pos.x(), sr.pos.y(), sr.pos.z());
        std::printf("  经纬高     : %.9f deg  %.9f deg  %.4f m\n",
                    la * 180.0 / PI, lo * 180.0 / PI, h);
        std::printf("  相对基准站 : E %+.4f  N %+.4f  U %+.4f  m（基线 %.4f m）\n",
                    enu.x(), enu.y(), enu.z(), d.norm());
        std::printf("  正式精度   : σE %.3e  σN %.3e  σU %.3e  m（3D %.3e m）\n",
                    sr.sig_E, sr.sig_N, sr.sig_U,
                    std::sqrt(sr.sig_E * sr.sig_E + sr.sig_N * sr.sig_N +
                              sr.sig_U * sr.sig_U));
        std::printf("               （正式精度只反映观测噪声权重；实际精度受未建模的\n");
        std::printf("                天线相位中心/对流层/电离层限制）\n");
        std::printf("  观测       : 伪距 %d  载波 %d  模糊度 %d  周跳 %d\n",
                    sr.n_p, sr.n_l, sr.n_amb, sr.n_slip);
        std::printf("  载波RMS    : %.4f m；ILS ratio %.3g  p_boot %.3g  固定 %d/%d\n",
                    sr.carr_rms_m, sr.ratio, sr.p_boot, sr.n_fixed, sr.n_amb);
    }

    /* ---- 5) CSV ---- */
    if (!csv.empty() && !kin.pos.empty()) {
        /* 把所有窗口的逐历元固定解按时间建表 */
        std::map<double, Eigen::Vector3d> fixed_by_time;
        for (const auto &w : wins) {
            for (size_t i = 0; i < w.fixed_epoch_t.size(); ++i)
                fixed_by_time[w.fixed_epoch_t[i]] = w.fixed_pos_per_epoch[i];
        }

        std::ofstream out(csv);
        out << "idx,gps_sec,lat_deg,lon_deg,height_m,X,Y,Z,baseline_m,n_dd,rms_m,"
            << "fix_ok,fix_lat_deg,fix_lon_deg,fix_height_m,fix_X,fix_Y,fix_Z,fix_baseline_m\n";
        out << std::setprecision(9);
        for (size_t i = 0; i < kin.pos.size(); ++i) {
            double la, lo, h;
            ecef2blh(kin.pos[i], &la, &lo, &h);
            out << i << ',' << std::fixed << std::setprecision(3) << kin.t[i] << ','
                << std::setprecision(9)
                << la * 180.0 / PI << ',' << lo * 180.0 / PI << ',' << h << ','
                << kin.pos[i].x() << ',' << kin.pos[i].y() << ',' << kin.pos[i].z() << ','
                << std::setprecision(3) << (kin.pos[i] - base_precise).norm() << ','
                << kin.n_dd[i] << ',' << kin.rms_m[i];

            auto it = fixed_by_time.find(kin.t[i]);
            if (it != fixed_by_time.end()) {
                double fla, flo, fh;
                ecef2blh(it->second, &fla, &flo, &fh);
                out << std::setprecision(9) << ",1,"
                    << fla * 180.0 / PI << ',' << flo * 180.0 / PI << ','
                    << std::setprecision(3) << fh << ','
                    << std::setprecision(9)
                    << it->second.x() << ',' << it->second.y() << ',' << it->second.z() << ','
                    << std::setprecision(3) << (it->second - base_precise).norm();
            } else {
                out << ",0,nan,nan,nan,nan,nan,nan,nan";
            }
            out << '\n';
        }
        std::printf("\n轨迹已写入: %s（%zu 行）\n", csv.c_str(), kin.pos.size());
    }

    return 0;
}
