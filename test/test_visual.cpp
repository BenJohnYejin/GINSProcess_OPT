// test_visual.cpp
// 编译（与 test_math.cpp 分开跑）：
//   g++ -std=c++17 -O2 test_visual.cpp alg_lib.cpp \
//       -I. -I/usr/include/eigen3 \
//       $(pkg-config --cflags --libs ceres) \
//       $(pkg-config --cflags --libs opencv4) \
//       -o test_visual
//
// 只测视觉模块：CameraModel / CalibManifold / VisualReprojResidual /
//              VisualFrontend / preprocessImages / 端到端 Ceres 收敛

#include "alg_lib.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

/* ============================== 极简测试框架 ============================== */
static int g_pass = 0, g_fail = 0;

#define EXPECT_TRUE(cond)                                                    \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
            g_fail++;                                                        \
        } else { g_pass++; }                                                 \
    } while (0)

#define EXPECT_NEAR(a, b, eps)                                               \
    do {                                                                     \
        double _a = (a), _b = (b);                                           \
        if (!(std::fabs(_a - _b) <= (eps))) {                                \
            std::printf("    FAIL %s:%d: %s=%.12g vs %s=%.12g (|d|=%.3g, eps=%.3g)\n", \
                        __FILE__, __LINE__, #a, _a, #b, _b,                   \
                        std::fabs(_a - _b), (double)(eps));                   \
            g_fail++;                                                        \
        } else { g_pass++; }                                                 \
    } while (0)

#define EXPECT_EQ(a, b)                                                      \
    do {                                                                     \
        if (!((a) == (b))) {                                                 \
            std::printf("    FAIL %s:%d: %s == %s\n",                        \
                        __FILE__, __LINE__, #a, #b);                         \
            g_fail++;                                                        \
        } else { g_pass++; }                                                 \
    } while (0)

#define RUN_TEST(fn)                                                         \
    do {                                                                     \
        std::printf("[ RUN  ] %s\n", #fn);                                   \
        int _before = g_fail;                                                \
        fn();                                                                \
        std::printf(g_fail == _before ? "[  OK  ] %s\n" : "[ FAIL ] %s\n",   \
                    #fn);                                                    \
    } while (0)

/* ============================================================================
 * 视觉测试辅助
 * ==========================================================================*/
/* 从 4 个参数块调用 VisualReprojResidual::Evaluate
 * 若不关心某个雅可比，传 nullptr；若四个都不关心，内部不分配雅可比缓冲
 *
 * 注意：
 *   - 对外接口统一用默认 ColMajor（与调用方声明的 Eigen::Matrix<double, 2, 7> 一致）
 *   - 内部缓冲用 RowMajor（Ceres 的 Jacobian 缓冲区约定）
 *   - 2×1 列向量不能 RowMajor，只能用 ColMajor */
#if 0
static void evalVisFactor(const VisualReprojResidual& f,
                          const double* pose_i,
                          const double* pose_j,
                          const double* calib,
                          const double* inv_d,
                          double r[2],
                          Eigen::Matrix<double, 2, 7>*  Ji,
                          Eigen::Matrix<double, 2, 7>*  Jj,
                          Eigen::Matrix<double, 2, 14>* Jc,
                          Eigen::Matrix<double, 2, 1>*  Jd) {
    const double* params[4] = { pose_i, pose_j, calib, inv_d };

    /* 内部缓冲：与 Ceres Jacobian 缓冲区布局一致（RowMajor）
     * 但 2×1 列向量无法 RowMajor，用默认 ColMajor */
    Eigen::Matrix<double, 2, 7,  Eigen::RowMajor> j_i;
    Eigen::Matrix<double, 2, 7,  Eigen::RowMajor> j_j;
    Eigen::Matrix<double, 2, 14, Eigen::RowMajor> j_c;
    Eigen::Matrix<double, 2, 1>                   j_d;

    const bool need_jac = (Ji || Jj || Jc || Jd);
    double* jacs[4] = { j_i.data(), j_j.data(), j_c.data(), j_d.data() };
    f.Evaluate(params, r, need_jac ? jacs : nullptr);

    /* 赋给调用方：Eigen 自动完成 RowMajor -> ColMajor 的布局转换 */
    if (Ji) *Ji = j_i;
    if (Jj) *Jj = j_j;
    if (Jc) *Jc = j_c;
    if (Jd) *Jd = j_d;
}
#endif
static void evalVisFactor(const VisualReprojResidual& f,
                          const double* pose_i,
                          const double* pose_j,
                          const double* calib,
                          const double* inv_d,
                          double r[2],
                          Eigen::Matrix<double, 2, 7>*  Ji,
                          Eigen::Matrix<double, 2, 7>*  Jj,
                          Eigen::Matrix<double, 2, 14>* Jc,
                          Eigen::Matrix<double, 2, 1>*  Jd) {
    const double* params[4] = { pose_i, pose_j, calib, inv_d };
    const bool need_jac = (Ji || Jj || Jc || Jd);

    /* 用普通 C 数组，按 RowMajor 布局自行管理：
     *   VisualReprojResidual::Evaluate 内部按 jacobians[i][row*N+col] 写，
     *   所以缓冲区必须是行主序的连续 2*N 个 double。
     *   不要用 Eigen::Matrix<double,2,N,RowMajor>：Eigen 对小矩阵会做
     *   内联优化，局部对象的存储分配不受我们控制，Evaluate 的裸指针写入
     *   可能落到未预留的空间里。 */
    double raw_i[2 * 7]  = {0};
    double raw_j[2 * 7]  = {0};
    double raw_c[2 * 14] = {0};
    double raw_d[2 * 1]  = {0};

    double* jacs[4] = { raw_i, raw_j, raw_c, raw_d };
    f.Evaluate(params, r, need_jac ? jacs : nullptr);

    if (Ji) {
        for (int rr = 0; rr < 2; ++rr)
            for (int cc = 0; cc < 7; ++cc)
                (*Ji)(rr, cc) = raw_i[rr * 7 + cc];
    }
    if (Jj) {
        for (int rr = 0; rr < 2; ++rr)
            for (int cc = 0; cc < 7; ++cc)
                (*Jj)(rr, cc) = raw_j[rr * 7 + cc];
    }
    if (Jc) {
        for (int rr = 0; rr < 2; ++rr)
            for (int cc = 0; cc < 14; ++cc)
                (*Jc)(rr, cc) = raw_c[rr * 14 + cc];
    }
    if (Jd) {
        for (int rr = 0; rr < 2; ++rr)
            (*Jd)(rr, 0) = raw_d[rr];
    }
}

/* pose 的 7×6 PlusJacobian（与 Ceres 内部一致） */
static Eigen::Matrix<double, 7, 6> posePlusJacobian(const double* pose) {
    PoseManifold pm;
    Eigen::Matrix<double, 7, 6, Eigen::RowMajor> J;
    pm.PlusJacobian(pose, J.data());
    return J;
}

/* calib 的 14×13 PlusJacobian */
static Eigen::Matrix<double, 14, 13> calibPlusJacobian(const double* calib) {
    CalibManifold cm;
    Eigen::Matrix<double, 14, 13, Eigen::RowMajor> J;
    cm.PlusJacobian(calib, J.data());
    return J;
}

/* 生成一幅合成网格图，供前端测试用 */
static cv::Mat makeSyntheticGrid(int w = 640, int h = 480,
                                 int step = 40, int border = 40) {
    cv::Mat img(h, w, CV_8UC1, cv::Scalar(255));
    for (int i = border; i < h - border; i += step)
        cv::line(img, cv::Point(border, i), cv::Point(w - border, i),
                 cv::Scalar(0), 2);
    for (int i = border; i < w - border; i += step)
        cv::line(img, cv::Point(i, border), cv::Point(i, h - border),
                 cv::Scalar(0), 2);
    return img;
}

/* ============================================================================
 * CameraModel
 * ==========================================================================*/

static void test_camera_model() {
    CameraModel cam;
    cam.fx = 460.0; cam.fy = 461.0;
    cam.cx = 320.0; cam.cy = 240.0;
    cam.k1 = -0.29; cam.k2 = 0.08;
    cam.p1 = 0.001; cam.p2 = -0.002;

    cv::Mat K = cam.K();
    EXPECT_NEAR(K.at<double>(0, 0), 460.0, 1e-12);
    EXPECT_NEAR(K.at<double>(1, 1), 461.0, 1e-12);
    EXPECT_NEAR(K.at<double>(0, 2), 320.0, 1e-12);
    EXPECT_NEAR(K.at<double>(1, 2), 240.0, 1e-12);
    EXPECT_NEAR(K.at<double>(2, 2), 1.0,   1e-12);
    EXPECT_NEAR(K.at<double>(0, 1), 0.0,   1e-12);

    cv::Mat D = cam.D();
    EXPECT_NEAR(D.at<double>(0, 0), -0.29,  1e-12);
    EXPECT_NEAR(D.at<double>(0, 1),  0.08,  1e-12);
    EXPECT_NEAR(D.at<double>(0, 2),  0.001, 1e-12);
    EXPECT_NEAR(D.at<double>(0, 3), -0.002, 1e-12);
}

/* ============================================================================
 * CalibManifold
 * ==========================================================================*/

static void test_calib_manifold_basic() {
    CalibManifold cm;
    EXPECT_EQ(cm.AmbientSize(), NUM_CALIB);      /* 14 */
    EXPECT_EQ(cm.TangentSize(), NUM_CALIB - 1);  /* 13 */

    double x[14] = { 0.1, 0.02, -0.03, 0.5, -0.2, 0.1, 0.05,
                     0.1, 0.05, -0.02,
                     0.0, 0.0, 0.0, 1.0 };

    double x_plus[14];
    double zero[13] = {0};
    cm.Plus(x, zero, x_plus);
    for (int i = 0; i < 14; i++)
        EXPECT_NEAR(x_plus[i], x[i], 1e-14);

    /* 前 10 维加性扰动 */
    double delta[13] = { 0.01, 0.02, 0.03, -0.1, 0.2, -0.3, 0.4,
                         0.5, -0.6, 0.7,
                         0, 0, 0 };
    cm.Plus(x, delta, x_plus);
    for (int i = 0; i < 10; i++)
        EXPECT_NEAR(x_plus[i], x[i] + delta[i], 1e-14);

    /* 四元数扰动：绕 z 轴 0.1 rad */
    double dq[13] = { 0,0,0, 0,0,0, 0, 0,0,0, 0, 0, 0.1 };
    cm.Plus(x, dq, x_plus);
    double nq = std::sqrt(x_plus[10]*x_plus[10] + x_plus[11]*x_plus[11] +
                          x_plus[12]*x_plus[12] + x_plus[13]*x_plus[13]);
    EXPECT_NEAR(nq, 1.0, 1e-13);
    EXPECT_NEAR(x_plus[10], 0.0,           1e-13);
    EXPECT_NEAR(x_plus[11], 0.0,           1e-13);
    EXPECT_NEAR(x_plus[12], std::sin(0.05), 1e-13);
    EXPECT_NEAR(x_plus[13], std::cos(0.05), 1e-13);
}

static void test_calib_manifold_plus_minus_inverse() {
    CalibManifold cm;
    double x[14] = { 0.1, 0.02, -0.03, 0.5, -0.2, 0.1, 0.05,
                     0.1, 0.05, -0.02,
                     0.01, -0.02, 0.03, 0.998 };
    /* 归一化 x 的四元数部分 */
    {
        quat q(x[13], x[10], x[11], x[12]);
        normlize(&q);
        x[10] = q.q1; x[11] = q.q2; x[12] = q.q3; x[13] = q.q0;
    }

    double delta[13] = { 0.01, -0.02, 0.03, 0.1, -0.2, 0.3, -0.4,
                         0.05, 0.06, -0.07,
                         0.001, -0.002, 0.003 };
    double y[14];
    cm.Plus(x, delta, y);

    double dy[13];
    cm.Minus(y, x, dy);
    for (int i = 0; i < 13; i++)
        EXPECT_NEAR(dy[i], delta[i], 1e-12);
}

static void test_calib_manifold_plus_jacobian() {
    CalibManifold cm;
    double x[14] = { 0.1, 0.02, -0.03, 0.5, -0.2, 0.1, 0.05,
                     0.1, 0.05, -0.02,
                     0.0, 0.0, 0.0, 1.0 };

    Eigen::Matrix<double, 14, 13, Eigen::RowMajor> J;
    cm.PlusJacobian(x, J.data());

    const double eps = 1e-6;
    for (int k = 0; k < 13; k++) {
        double dp[13] = {0}, dm[13] = {0};
        dp[k] = eps;  dm[k] = -eps;
        double xp[14], xm[14];
        cm.Plus(x, dp, xp);
        cm.Plus(x, dm, xm);
        for (int i = 0; i < 14; i++) {
            double num = (xp[i] - xm[i]) / (2.0 * eps);
            EXPECT_NEAR(J(i, k), num, 1e-6);
        }
    }
}

static void test_calib_manifold_minus_jacobian() {
    CalibManifold cm;
    double x[14] = { 0.1, 0.02, -0.03, 0.5, -0.2, 0.1, 0.05,
                     0.1, 0.05, -0.02,
                     0.0, 0.0, 0.0, 1.0 };

    Eigen::Matrix<double, 13, 14, Eigen::RowMajor> J;
    cm.MinusJacobian(x, J.data());

    const double eps = 1e-6;
    for (int k = 0; k < 14; k++) {
        double yp[14], ym[14];
        std::memcpy(yp, x, sizeof(yp));
        std::memcpy(ym, x, sizeof(ym));
        yp[k] += eps;  ym[k] -= eps;
        double dp[13], dm[13];
        cm.Minus(yp, x, dp);
        cm.Minus(ym, x, dm);
        for (int i = 0; i < 13; i++) {
            double num = (dp[i] - dm[i]) / (2.0 * eps);
            EXPECT_NEAR(J(i, k), num, 1e-5);
        }
    }
}

/* ============================================================================
 * VisualReprojResidual：残差正确性
 * ==========================================================================*/

static void test_visual_residual_zero_same_frame() {
    /* 两帧位姿相同、外参单位、观测相同 → 残差必为 0 */
    vect3 obs(0.1, -0.2, 1.0);
    VisualReprojResidual f(obs, obs, 1.0);

    double pose_i[7] = { 0, 0, 0, 0, 0, 0, 1 };
    double pose_j[7] = { 0, 0, 0, 0, 0, 0, 1 };
    double calib[14] = { 0,0,0, 0,0,0, 0, 0,0,0, 0,0,0,1 };
    double inv_d[1]  = { 0.5 };

    double r[2];
    evalVisFactor(f, pose_i, pose_j, calib, inv_d, r,
                  nullptr, nullptr, nullptr, nullptr);
    EXPECT_NEAR(r[0], 0.0, 1e-13);
    EXPECT_NEAR(r[1], 0.0, 1e-13);
}

static void test_visual_residual_reproj() {
    /* 帧 i 在原点朝向 z+，帧 j 在 z=1m 处
     * 特征在帧 i 的 z=2m 处：obs_i = (0,0,1)，ρ = 0.5
     * 投影到帧 j：Z_c2 = 2-1 = 1 → obs_j = (0,0,1)
     * 残差应为 0 */
    vect3 obs_i(0.0, 0.0, 1.0);
    vect3 obs_j(0.0, 0.0, 1.0);
    VisualReprojResidual f(obs_i, obs_j, 1.0);

    double pose_i[7] = { 0, 0, 0, 0, 0, 0, 1 };
    double pose_j[7] = { 0, 0, 1, 0, 0, 0, 1 };
    double calib[14] = { 0,0,0, 0,0,0, 0, 0,0,0, 0,0,0,1 };
    double inv_d[1]  = { 0.5 };

    double r[2];
    evalVisFactor(f, pose_i, pose_j, calib, inv_d, r,
                  nullptr, nullptr, nullptr, nullptr);
    EXPECT_NEAR(r[0], 0.0, 1e-13);
    EXPECT_NEAR(r[1], 0.0, 1e-13);
}

static void test_visual_residual_off_center() {
    /* 特征在帧 i 的 (1, 2, 2)：obs_i = (0.5, 1.0, 1)
     * 帧 j 平移 +z 1m：Z_c2 = 1，obs_j 仍为 (1.0, 2.0, 1.0)，残差 0 */
    vect3 obs_i(0.5, 1.0, 1.0);
    vect3 obs_j(1.0, 2.0, 1.0);
    VisualReprojResidual f(obs_i, obs_j, 1.0);

    double pose_i[7] = { 0, 0, 0, 0, 0, 0, 1 };
    double pose_j[7] = { 0, 0, 1, 0, 0, 0, 1 };
    double calib[14] = { 0,0,0, 0,0,0, 0, 0,0,0, 0,0,0,1 };
    double inv_d[1]  = { 0.5 };

    double r[2];
    evalVisFactor(f, pose_i, pose_j, calib, inv_d, r,
                  nullptr, nullptr, nullptr, nullptr);
    EXPECT_NEAR(r[0], 0.0, 1e-13);
    EXPECT_NEAR(r[1], 0.0, 1e-13);
}

/* ============================================================================
 * VisualReprojResidual：雅可比数值校验
 * ==========================================================================*/

/* 一组非平凡的位姿 / 标定 / 逆深度，供 4 个雅可比测试复用 */
struct VisJacFixture {
    double pose_i[7];
    double pose_j[7];
    double calib[NUM_CALIB];
    double inv_d[1];
    vect3  obs_i, obs_j;
    double sigma;

    VisJacFixture() {
        obs_i = vect3(0.12, -0.20, 1.0);
        obs_j = vect3(0.18, -0.16, 1.0);
        sigma = 1.0;

        quat qi = a2qua(vect3(0.05, -0.10, 0.30));
        quat qj = a2qua(vect3(0.08,  0.02, 0.25));
        pose_i[0] = 1.0;  pose_i[1] = -0.5;  pose_i[2] = 0.2;
        pose_i[3] = qi.q1; pose_i[4] = qi.q2; pose_i[5] = qi.q3; pose_i[6] = qi.q0;
        pose_j[0] = 2.0;  pose_j[1] = 0.3;   pose_j[2] = -0.1;
        pose_j[3] = qj.q1; pose_j[4] = qj.q2; pose_j[5] = qj.q3; pose_j[6] = qj.q0;

        for (int k = 0; k < NUM_CALIB; k++) calib[k] = 0.0;
        calib[7]  = 0.05;  calib[8] = -0.02;  calib[9] = 0.01;
        quat qb = rv2q(vect3(0.02, -0.01, 0.03));
        calib[10] = qb.q1; calib[11] = qb.q2;
        calib[12] = qb.q3; calib[13] = qb.q0;

        inv_d[0] = 0.4;
    }
};

static void test_visual_jacobian_pose_i() {
    VisJacFixture fx;
    VisualReprojResidual f(fx.obs_i, fx.obs_j, fx.sigma);

    Eigen::Matrix<double, 2, 7> Ji_amb;
    double r[2];
    evalVisFactor(f, fx.pose_i, fx.pose_j, fx.calib, fx.inv_d,
                  r, &Ji_amb, nullptr, nullptr, nullptr);

    Eigen::Matrix<double, 2, 6> Ji_tan = Ji_amb * posePlusJacobian(fx.pose_i);

    const double eps = 1e-6;
    for (int k = 0; k < 6; k++) {
        double dp[6] = {0}, dm[6] = {0};
        dp[k] = eps;  dm[k] = -eps;
        double pi_p[7], pi_m[7];
        posePlus(fx.pose_i, dp, pi_p);
        posePlus(fx.pose_i, dm, pi_m);

        double rp[2], rm[2];
        evalVisFactor(f, pi_p, fx.pose_j, fx.calib, fx.inv_d,
                      rp, nullptr, nullptr, nullptr, nullptr);
        evalVisFactor(f, pi_m, fx.pose_j, fx.calib, fx.inv_d,
                      rm, nullptr, nullptr, nullptr, nullptr);

        for (int m = 0; m < 2; m++) {
            double num = (rp[m] - rm[m]) / (2.0 * eps);
            EXPECT_NEAR(Ji_tan(m, k), num, 1e-5);
        }
    }
}

static void test_visual_jacobian_pose_j() {
    VisJacFixture fx;
    VisualReprojResidual f(fx.obs_i, fx.obs_j, fx.sigma);

    Eigen::Matrix<double, 2, 7> Jj_amb;
    double r[2];
    evalVisFactor(f, fx.pose_i, fx.pose_j, fx.calib, fx.inv_d,
                  r, nullptr, &Jj_amb, nullptr, nullptr);

    Eigen::Matrix<double, 2, 6> Jj_tan = Jj_amb * posePlusJacobian(fx.pose_j);

    const double eps = 1e-6;
    for (int k = 0; k < 6; k++) {
        double dp[6] = {0}, dm[6] = {0};
        dp[k] = eps;  dm[k] = -eps;
        double pj_p[7], pj_m[7];
        posePlus(fx.pose_j, dp, pj_p);
        posePlus(fx.pose_j, dm, pj_m);

        double rp[2], rm[2];
        evalVisFactor(f, fx.pose_i, pj_p, fx.calib, fx.inv_d,
                      rp, nullptr, nullptr, nullptr, nullptr);
        evalVisFactor(f, fx.pose_i, pj_m, fx.calib, fx.inv_d,
                      rm, nullptr, nullptr, nullptr, nullptr);

        for (int m = 0; m < 2; m++) {
            double num = (rp[m] - rm[m]) / (2.0 * eps);
            EXPECT_NEAR(Jj_tan(m, k), num, 1e-5);
        }
    }
}

static void test_visual_jacobian_calib() {
    VisJacFixture fx;
    VisualReprojResidual f(fx.obs_i, fx.obs_j, fx.sigma);

    Eigen::Matrix<double, 2, 14> Jc_amb;
    double r[2];
    evalVisFactor(f, fx.pose_i, fx.pose_j, fx.calib, fx.inv_d,
                  r, nullptr, nullptr, &Jc_amb, nullptr);

    Eigen::Matrix<double, 2, 13> Jc_tan = Jc_amb * calibPlusJacobian(fx.calib);

    CalibManifold cm;
    const double eps = 1e-6;
    for (int k = 0; k < 13; k++) {
        double dp[13] = {0}, dm[13] = {0};
        dp[k] = eps;  dm[k] = -eps;
        double cp[14], cm14[14];
        cm.Plus(fx.calib, dp, cp);
        cm.Plus(fx.calib, dm, cm14);

        double rp[2], rm[2];
        evalVisFactor(f, fx.pose_i, fx.pose_j, cp,   fx.inv_d,
                      rp, nullptr, nullptr, nullptr, nullptr);
        evalVisFactor(f, fx.pose_i, fx.pose_j, cm14, fx.inv_d,
                      rm, nullptr, nullptr, nullptr, nullptr);

        for (int m = 0; m < 2; m++) {
            double num = (rp[m] - rm[m]) / (2.0 * eps);
            EXPECT_NEAR(Jc_tan(m, k), num, 1e-5);
        }
    }
}

static void test_visual_jacobian_inv_depth() {
    VisJacFixture fx;
    VisualReprojResidual f(fx.obs_i, fx.obs_j, fx.sigma);

    Eigen::Matrix<double, 2, 1> Jd_amb;
    double r[2];
    evalVisFactor(f, fx.pose_i, fx.pose_j, fx.calib, fx.inv_d,
                  r, nullptr, nullptr, nullptr, &Jd_amb);

    const double eps = 1e-7;
    double d_p[1] = { fx.inv_d[0] + eps };
    double d_m[1] = { fx.inv_d[0] - eps };
    double rp[2], rm[2];
    evalVisFactor(f, fx.pose_i, fx.pose_j, fx.calib, d_p,
                  rp, nullptr, nullptr, nullptr, nullptr);
    evalVisFactor(f, fx.pose_i, fx.pose_j, fx.calib, d_m,
                  rm, nullptr, nullptr, nullptr, nullptr);

    for (int m = 0; m < 2; m++) {
        double num = (rp[m] - rm[m]) / (2.0 * eps);
        EXPECT_NEAR(Jd_amb(m, 0), num, 1e-5);
    }
}

/* ============================================================================
 * VisualFrontend
 * ==========================================================================*/

static void test_visual_frontend_first_frame() {
    CameraModel cam;
    cam.fx = 460.0; cam.fy = 460.0;
    cam.cx = 320.0; cam.cy = 240.0;

    VisualFrontend vf(cam, 60, 15);
    cv::Mat img = makeSyntheticGrid();
    auto feats = vf.process(0.0, img);

    EXPECT_TRUE(feats.size() >= 20);

    for (const auto& f : feats) {
        EXPECT_NEAR(f.xyz_c(2), 1.0, 1e-12);
        EXPECT_TRUE(f.track_cnt >= 1);
        EXPECT_TRUE(f.feat_id > 0);
    }

    /* 特征 id 应互不相同 */
    std::set<int> ids;
    for (const auto& f : feats) ids.insert(f.feat_id);
    EXPECT_EQ((int)ids.size(), (int)feats.size());
}

static void test_visual_frontend_tracking() {
    CameraModel cam;
    cam.fx = 460.0; cam.fy = 460.0;
    cam.cx = 320.0; cam.cy = 240.0;

    VisualFrontend vf(cam, 60, 15);

    cv::Mat img1 = makeSyntheticGrid();
    auto f1 = vf.process(0.0, img1);
    const int n1 = (int)f1.size();
    EXPECT_TRUE(n1 >= 20);

    /* 帧 2：整体平移 (5, 3) 像素 */
    cv::Mat img2;
    cv::Mat M = (cv::Mat_<double>(2,3) << 1, 0, 5, 0, 1, 3);
    cv::warpAffine(img1, img2, M, img1.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(255));

    auto f2 = vf.process(1.0, img2);

    /* 至少一半点被追踪 */
    int tracked = 0;
    for (const auto& f : f2) if (f.track_cnt >= 2) tracked++;
    EXPECT_TRUE(tracked >= n1 / 2);

    /* 追踪到的点保留原 id，位移接近 (5, 3) */
    for (const auto& a : f1) {
        for (const auto& b : f2) {
            if (b.feat_id == a.feat_id && b.track_cnt >= 2) {
                EXPECT_NEAR(b.pt.x - a.pt.x, 5.0, 1.0);
                EXPECT_NEAR(b.pt.y - a.pt.y, 3.0, 1.0);
                break;
            }
        }
    }
}

static void test_visual_frontend_mask_prevents_clustering() {
    CameraModel cam;
    cam.fx = 460.0; cam.fy = 460.0;
    cam.cx = 320.0; cam.cy = 240.0;

    VisualFrontend vf(cam, 60, 30);
    cv::Mat img = makeSyntheticGrid();
    auto feats = vf.process(0.0, img);

    /* 任意两特征间的像素距离不应过近 */
    const double min_sq = (30.0 * 0.5) * (30.0 * 0.5);
    for (size_t i = 0; i < feats.size(); i++) {
        for (size_t j = i + 1; j < feats.size(); j++) {
            const double dx = feats[i].pt.x - feats[j].pt.x;
            const double dy = feats[i].pt.y - feats[j].pt.y;
            EXPECT_TRUE(dx * dx + dy * dy >= min_sq);
        }
    }
}

/* ============================================================================
 * preprocessImages
 * ==========================================================================*/

static void test_preprocess_images_from_files() {
    fs::path tmp = fs::temp_directory_path() / "ipos_vis_ut";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    CameraModel cam;
    cam.fx = 460.0; cam.fy = 460.0;
    cam.cx = 320.0; cam.cy = 240.0;

    const long long t0_ns = 1403636579000000000LL;
    for (int k = 0; k < 4; k++) {
        cv::Mat img = makeSyntheticGrid();
        cv::Mat M = (cv::Mat_<double>(2,3) << 1, 0, k, 0, 1, k);
        cv::Mat shifted;
        cv::warpAffine(img, shifted, M, img.size(), cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, cv::Scalar(255));

        char name[64];
        std::snprintf(name, sizeof(name), "%lld.png", t0_ns + k * 50000000LL);
        EXPECT_TRUE(cv::imwrite((tmp / name).string(), shifted));
    }

    auto table = preprocessImages(tmp.string(), cam, 0.003);
    EXPECT_EQ((int)table.size(), 4);

    int n_nonempty = 0;
    for (const auto& kv : table) {
        if (!kv.second.empty()) n_nonempty++;
        for (const auto& obs : kv.second) {
            EXPECT_NEAR(obs.xyz_c.k, 1.0, 1e-12);
            EXPECT_TRUE(obs.sigma > 0.0);
        }
    }
    /* 首帧必然空（track_cnt 都=1），后续帧至少有一部分非空 */
    EXPECT_TRUE(n_nonempty >= 1);

    /* 时间戳应是纳秒转秒 */
    EXPECT_NEAR(table.begin()->first, t0_ns * 1e-9, 1e-9);

    fs::remove_all(tmp);
}

static void test_preprocess_images_empty_dir() {
    fs::path tmp = fs::temp_directory_path() / "ipos_vis_empty";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    CameraModel cam;
    auto table = preprocessImages(tmp.string(), cam, 0.003);
    EXPECT_EQ((int)table.size(), 0);

    fs::remove_all(tmp);
}
#if 1
static void test_visual_factor_ceres_convergence() {
    /* 1) 真值：任意选 rho_gt、obs_i、pose/calib；
     * 2) 由几何关系反算 obs_j_gt；
     * 3) 给错初值，看 Ceres 是否收敛回 rho_gt。 */
    VisJacFixture fx;
    const double rho_gt = 0.4;

    /* 复现 Evaluate 内部的投影链，算 obs_j 应为什么 */
    quat qi(fx.pose_i[6], fx.pose_i[3], fx.pose_i[4], fx.pose_i[5]);
    quat qj(fx.pose_j[6], fx.pose_j[3], fx.pose_j[4], fx.pose_j[5]);
    quat qb(fx.calib[13], fx.calib[10], fx.calib[11], fx.calib[12]);
    vect3 ti(fx.pose_i[0], fx.pose_i[1], fx.pose_i[2]);
    vect3 tj(fx.pose_j[0], fx.pose_j[1], fx.pose_j[2]);
    vect3 t_bc(fx.calib[7], fx.calib[8], fx.calib[9]);

    quat  q_wc_i = qi * qb;
    vect3 p_wc_i = ti + qi * t_bc;
    quat  q_wc_j = qj * qb;
    vect3 p_wc_j = tj + qj * t_bc;

    const double inv_rho = 1.0 / rho_gt;
    vect3 P_c_i(fx.obs_i.i * inv_rho, fx.obs_i.j * inv_rho, inv_rho);
    vect3 P_w   = p_wc_i + q_wc_i * P_c_i;
    vect3 P_c_j = (~q_wc_j) * (P_w - p_wc_j);
    vect3 obs_j_gt(P_c_j.i / P_c_j.k, P_c_j.j / P_c_j.k, 1.0);

    VisualReprojResidual* cost =
        new VisualReprojResidual(fx.obs_i, obs_j_gt, fx.sigma);

    double rho = rho_gt + 0.5;   /* 故意给错 */

    ceres::Problem problem;
    problem.AddParameterBlock(&rho, 1);
    problem.SetParameterLowerBound(&rho, 0, 1e-3);
    problem.SetParameterUpperBound(&rho, 0, 10.0);
    problem.AddResidualBlock(cost, nullptr,
                             fx.pose_i, fx.pose_j, fx.calib, &rho);
    problem.SetParameterBlockConstant(fx.pose_i);
    problem.SetParameterBlockConstant(fx.pose_j);
    problem.SetParameterBlockConstant(fx.calib);

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = 50;
    opts.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    /* 逆深度应收敛到 rho_gt */
    EXPECT_NEAR(rho, rho_gt, 1e-4);
}
#endif

#if 0
/* ============================================================================
 * 端到端 Ceres 收敛
 * ==========================================================================*/
static void test_visual_factor_ceres_convergence() {
    /* 位姿固定，仅优化逆深度：初值给错，求解应恢复真值 */
    VisJacFixture fx;
    const double rho_gt = fx.inv_d[0];

    VisualReprojResidual* cost =
        new VisualReprojResidual(fx.obs_i, fx.obs_j, fx.sigma);

    double rho = rho_gt + 0.5;   /* 初值故意给错 */

    ceres::Problem problem;
    problem.AddParameterBlock(&rho, 1);
    problem.SetParameterLowerBound(&rho, 0, 1e-3);
    problem.SetParameterUpperBound(&rho, 0, 10.0);
    problem.AddResidualBlock(cost, nullptr,
                             fx.pose_i, fx.pose_j, fx.calib, &rho);

    problem.SetParameterBlockConstant(fx.pose_i);
    problem.SetParameterBlockConstant(fx.pose_j);
    problem.SetParameterBlockConstant(fx.calib);

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = 50;
    opts.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    /* 残差接近 0（因为观测就是按 rho_gt 生成的） */
    double r[2];
    evalVisFactor(VisualReprojResidual(fx.obs_i, fx.obs_j, fx.sigma),
                  fx.pose_i, fx.pose_j, fx.calib, &rho,
                  r, nullptr, nullptr, nullptr, nullptr);
    EXPECT_NEAR(r[0], 0.0, 1e-6);
    EXPECT_NEAR(r[1], 0.0, 1e-6);
}
#endif

/* ============================================================================
 * main
 * ==========================================================================*/

int main() {
    std::printf("========== 视觉模块单元测试 ==========\n");

    /* 相机模型 */
    RUN_TEST(test_camera_model);

    /* 标定流形 */
    RUN_TEST(test_calib_manifold_basic);
    RUN_TEST(test_calib_manifold_plus_minus_inverse);
    RUN_TEST(test_calib_manifold_plus_jacobian);
    RUN_TEST(test_calib_manifold_minus_jacobian);

    /* 视觉因子残差 */
    RUN_TEST(test_visual_residual_zero_same_frame);
    RUN_TEST(test_visual_residual_reproj);
    RUN_TEST(test_visual_residual_off_center);

    /* 视觉因子雅可比（核心） */
    RUN_TEST(test_visual_jacobian_pose_i);
    RUN_TEST(test_visual_jacobian_pose_j);
    RUN_TEST(test_visual_jacobian_calib);
    RUN_TEST(test_visual_jacobian_inv_depth);

    /* 视觉前端 */
    RUN_TEST(test_visual_frontend_first_frame);
    RUN_TEST(test_visual_frontend_tracking);
    RUN_TEST(test_visual_frontend_mask_prevents_clustering);

    /* 图像预处理 */
    RUN_TEST(test_preprocess_images_from_files);
    RUN_TEST(test_preprocess_images_empty_dir);

    /* 端到端 */
    RUN_TEST(test_visual_factor_ceres_convergence);

    std::printf("=============================================\n");
    std::printf("总断言: %d, 通过: %d, 失败: %d\n",
                g_pass + g_fail, g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}