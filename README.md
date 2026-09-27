# 第 X 章 GNSS/INS 滑窗图优化松组合导航理论

本章建立 GNSS/INS 松组合滑窗图优化的完整理论框架。首先定义坐标系与状态量；其次推导 IMU 预积分的连续与离散形式及其误差传播；然后在流形上定义状态扰动与切空间运算；随后推导各观测因子（GNSS 位置、速度、航向、静止零速、里程计速度）的残差模型与雅可比；最后给出边缘化的 Schur 补推导与整体优化问题的求解。

---

## X.1 坐标系与符号约定

### X.1.1 坐标系定义

定义如下三个正交坐标系：

- **惯性系 $i$**：地心惯性坐标系。
- **地球系 $e$**：地心地固坐标系。
- **导航系 $n$**：东-北-天（ENU）局部水平坐标系，原点取第一个有效 GNSS 观测点。
- **机体系 $b$**：右-前-上（RFU），原点取 IMU 中心。
- **里程计系 $m$**：里程计安装方向，理想情况下 $x_m$ 沿前进方向。

### X.1.2 符号约定

- 四元数 $q=[q_0,q_1,q_2,q_3]=[w,x,y,z]$ 表示 $b\to n$ 旋转，旋转矩阵记为 $R(q)\in SO(3)$。
- 欧拉角 $\text{att}=[\text{pitch},\text{roll},\text{yaw}]$，姿态解算约定为

$$
\text{pitch}=\arcsin(R_{21}),\quad
\text{roll}=\operatorname{atan2}(-R_{20},R_{22}),\quad
\text{yaw}=\operatorname{atan2}(-R_{01},R_{11})
\tag{X.1}
$$

- 向量叉乘矩阵（反对称矩阵）：

$$
[a]_\times=\begin{bmatrix}
0&-a_z&a_y\\
a_z&0&-a_x\\
-a_y&a_x&0
\end{bmatrix}
\tag{X.2}
$$

- 四元数指数映射与对数映射：

$$
\mathrm{Exp}(\phi)=\begin{bmatrix}\cos(\|\phi\|/2)\\[2pt]\dfrac{\phi}{\|\phi\|}\sin(\|\phi\|/2)\end{bmatrix},\qquad
\mathrm{Log}(q)=2\arccos(q_0)\frac{q_v}{\|q_v\|}
\tag{X.3}
$$

### X.1.3 状态量定义

滑窗内第 $k$ 个关键帧的状态定义为

$$
x_k=(p_k,q_k,v_k,b_{g,k},b_{a,k},s_{\text{odo},k})
\tag{X.4}
$$

其中 $p_k\in\mathbb R^3$ 为局部 ENU 位置，$q_k\in S^3$ 为 $b\to n$ 姿态，$v_k\in\mathbb R^3$ 为 ENU 速度，$b_{g,k},b_{a,k}\in\mathbb R^3$ 为陀螺与加表零偏，$s_{\text{odo},k}\in\mathbb R$ 为里程计刻度因子误差。整体状态 $x=\{x_k\}_{k=0}^{N-1}$，其中 $N$ 为滑窗内关键帧数。

---

## X.2 IMU 预积分理论

### X.2.1 连续时间运动学

IMU 的陀螺与加表测量模型为

$$
\omega_m=\omega+b_g+n_g,\qquad a_m=a+b_a+n_a
\tag{X.5}
$$

其中 $\omega,a$ 为真实角速度与比力，$n_g,n_a$ 为白噪声。导航方程在 $n$ 系下为

$$
\dot p=v,\qquad
\dot v=R(q)a+g,\qquad
\dot q=\tfrac12 q\otimes(\omega_m-b_g-n_g)
\tag{X.6}
$$

零偏随机游走：

$$
\dot b_g=n_{bg},\qquad \dot b_a=n_{ba}
\tag{X.7}
$$

**注 X.1** 式 (X.6) 中 $\dot q$ 的右乘形式与四元数约定一致：$q\otimes\delta q$ 表示在机体系施加扰动。

### X.2.2 预积分量的定义

在起始帧 $b_i$ 系中定义预积分量

$$
\Delta p_{ij}=R_i^T\left(p_j-p_i-v_i\Delta t-\tfrac12 g\Delta t^2\right)
\tag{X.8}
$$

$$
\Delta v_{ij}=R_i^T(v_j-v_i-g\Delta t)
\tag{X.9}
$$

$$
\Delta q_{ij}=q_i^{-1}\otimes q_j
\tag{X.10}
$$

其中 $\Delta t=t_j-t_i$，$g$ 为局部 ENU 下的重力矢量（指向下）。由式 (X.6) 直接可得预积分量的微分方程

$$
\dot{\Delta p}=\Delta v,\qquad
\dot{\Delta v}=R(\Delta q)(a_m-b_a-n_a),\qquad
\dot{\Delta q}=\tfrac12\Delta q\otimes(\omega_m-b_g-n_g)
\tag{X.11}
$$

### X.2.3 中值离散积分

设第 $k$ 个 IMU 区间时长为 $\Delta t_k$，去偏后测量为

$$
\bar\omega_k=\omega_{m,k}-b_g,\qquad \bar a_k=a_{m,k}-b_a
\tag{X.12}
$$

中值积分离散形式为

$$
\Delta q_{k+1}=\Delta q_k\otimes\mathrm{Exp}(\bar\omega_k\Delta t_k)
\tag{X.13}
$$

$$
\Delta q_{\mathrm{mid}}=\Delta q_k\otimes\mathrm{Exp}(\tfrac12\bar\omega_k\Delta t_k)
\tag{X.14}
$$

$$
\Delta v_{k+1}=\Delta v_k+R(\Delta q_{\mathrm{mid}})\bar a_k\Delta t_k
\tag{X.15}
$$

$$
\Delta p_{k+1}=\Delta p_k+\Delta v_k\Delta t_k+\tfrac12R(\Delta q_{\mathrm{mid}})\bar a_k\Delta t_k^2
\tag{X.16}
$$

**定理 X.1**（中值积分精度）当 $\bar\omega_k\Delta t_k$ 与 $\bar a_k\Delta t_k$ 为小量时，式 (X.13)–(X.16) 的截断误差为 $O(\Delta t_k^3)$。

**证明** 将真实积分在区间中点展开，与式 (X.13)–(X.16) 比较，梯形公式（中点）的局部截断误差为 $O(\Delta t_k^3)$。$\square$

### X.2.4 误差状态定义

定义误差状态

$$
\Delta p=\Delta\hat p+\delta p,\quad
\Delta v=\Delta\hat v+\delta v,\quad
\Delta q=\Delta\hat q\otimes\mathrm{Exp}(\delta\phi)
\tag{X.17}
$$

$$
b_g=\hat b_g+\delta b_g,\quad
b_a=\hat b_a+\delta b_a
\tag{X.18}
$$

其中 $\delta\phi$ 为机体系下的右扰动。误差状态向量按如下顺序排列：

$$
\delta x=[\delta p,\delta v,\delta\phi,\delta b_g,\delta b_a]\in\mathbb R^{15}
\tag{X.19}
$$

### X.2.5 连续时间误差方程

对式 (X.11) 在名义值处线性化，得到误差微分方程

$$
\dot{\delta p}=\delta v
\tag{X.20}
$$

$$
\dot{\delta v}=-R(\Delta q)[a]_\times\delta\phi-R(\Delta q)\delta b_a-R(\Delta q)n_a
\tag{X.21}
$$

$$
\dot{\delta\phi}=-[\omega]_\times\delta\phi-\delta b_g-n_g
\tag{X.22}
$$

$$
\dot{\delta b_g}=n_{bg},\qquad \dot{\delta b_a}=n_{ba}
\tag{X.23}
$$

**证明** 以 $\dot{\delta v}$ 为例。由式 (X.11) 第二式，名义量为

$$
\dot{\Delta\hat v}=R(\Delta\hat q)\bar a
$$

真实量为

$$
\dot{\Delta v}=R(\Delta\hat q\otimes\mathrm{Exp}(\delta\phi))(\bar a-\delta b_a-n_a)
$$

利用 $R(\mathrm{Exp}(\delta\phi))\approx I+[\delta\phi]_\times$，得

$$
R(\Delta q)\approx R(\Delta\hat q)(I+[\delta\phi]_\times)
$$

代入并减去名义式，忽略二阶小量：

$$
\dot{\delta v}=R(\Delta\hat q)\big([\delta\phi]_\times\bar a-\delta b_a-n_a\big)
=-R(\Delta\hat q)[\bar a]_\times\delta\phi-R(\Delta\hat q)\delta b_a-R(\Delta\hat q)n_a
$$

即式 (X.21)。其余类似。$\square$

### X.2.6 离散时间误差传播

将式 (X.20)–(X.23) 离散化，得到一阶形式

$$
\delta x_{k+1}=F_k\delta x_k+G_k w_k,\qquad
w_k=[n_g,n_a,n_{bg},n_{ba}]^T
\tag{X.24}
$$

其中

$$
F_k=I+A_k\Delta t_k,\qquad
A_k=\begin{bmatrix}
0&I&0&0&0\\
0&0&-R[a]_\times&0&-R\\
0&0&-[\omega]_\times&-I&0\\
0&0&0&0&0\\
0&0&0&0&0
\end{bmatrix}
\tag{X.25}
$$

$$
G_k=\begin{bmatrix}
0&0&0&0\\
0&-R&0&0\\
-I&0&0&0\\
0&0&I&0\\
0&0&0&I
\end{bmatrix}
\tag{X.26}
$$

过程噪声协方差

$$
Q_k=\mathrm{diag}\big(\sigma_g^2\Delta t_k I,\ \sigma_a^2\Delta t_k I,\ \tfrac{2\sigma_{bg}^2}{\tau}\Delta t_k I,\ \tfrac{2\sigma_{ba}^2}{\tau}\Delta t_k I\big)
\tag{X.27}
$$

协方差传播与状态转移矩阵累积：

$$
P_{k+1}=F_kP_kF_k^T+G_kQ_kG_k^T,\qquad
\Phi_{k+1}=F_k\Phi_k
\tag{X.28}
$$

其中 $\Phi_k=\partial\delta x_k/\partial\delta x_i$。

### X.2.7 零偏一阶补偿

当零偏估计偏离线性化点时，预积分量可用一阶展开补偿。设

$$
\delta b_g=b_{g,i}-b_g^{lin},\qquad \delta b_a=b_{a,i}-b_a^{lin}
\tag{X.29}
$$

则

$$
\Delta p_{\mathrm{corr}}=\Delta p+J_{p,b_g}\delta b_g+J_{p,b_a}\delta b_a
\tag{X.30}
$$

$$
\Delta v_{\mathrm{corr}}=\Delta v+J_{v,b_g}\delta b_g+J_{v,b_a}\delta b_a
\tag{X.31}
$$

$$
\Delta q_{\mathrm{corr}}=\Delta q\otimes\mathrm{Exp}(J_{q,b_g}\delta b_g)
\tag{X.32}
$$

其中 $J_{\cdot,b}=\Phi$ 的对应分块：

$$
J_{p,b_g}=\Phi_{0,9},\ J_{p,b_a}=\Phi_{0,12},\ J_{v,b_g}=\Phi_{3,9},\ J_{v,b_a}=\Phi_{3,12},\ J_{q,b_g}=\Phi_{6,9}
\tag{X.33}
$$

### X.2.8 预积分残差

定义残差向量 $r=[r_p,r_v,r_\phi,r_{bg},r_{ba}]\in\mathbb R^{15}$：

$$
r_p=R_i^T\left(p_j-p_i-v_i\Delta t-\tfrac12g\Delta t^2\right)-\Delta p_{\mathrm{corr}}
\tag{X.34}
$$

$$
r_v=R_i^T(v_j-v_i-g\Delta t)-\Delta v_{\mathrm{corr}}
\tag{X.35}
$$

$$
r_\phi=2\left[\Delta q_{\mathrm{corr}}^{-1}\otimes q_i^{-1}\otimes q_j\right]_{\mathrm{vec}}
\tag{X.36}
$$

$$
r_{bg}=b_{g,j}-b_{g,i},\qquad r_{ba}=b_{a,j}-b_{a,i}
\tag{X.37}
$$

若按预积分协方差白化，则乘 $S=P_{ij}^{-1/2}$，得 $r\leftarrow Sr$。

---

## X.3 状态流形与切空间

### X.3.1 流形的定义

状态 $x$ 所在流形为

$$
\mathcal M=\mathbb R^3\times S^3\times\mathbb R^3\times\mathbb R^3\times\mathbb R^3\ (\times\mathbb R)
\tag{X.38}
$$

其在单位元处的切空间维数为

$$
\dim\mathcal T_x\mathcal M=3+3+3+3+3=15\quad(\text{含里程计为 }16)
\tag{X.39}
$$

### X.3.2 右扰动与切空间运算

在 $q$ 处定义右扰动：

$$
q'=q\otimes\mathrm{Exp}(\delta\phi)
\tag{X.40}
$$

对应旋转矩阵近似：

$$
R(q')\approx R(q)(I+[\delta\phi]_\times)
\tag{X.41}
$$

流形加法 $\boxplus$ 与减法 $\boxminus$ 定义为

$$
\text{plus}(s,\delta):\quad
\begin{cases}
p'=p+\delta p\\
q'=q\otimes\mathrm{Exp}(\delta\phi)\\
v'=v+\delta v\\
b_g'=b_g+\delta b_g\\
b_a'=b_a+\delta b_a
\end{cases}
\tag{X.42}
$$

$$
\text{minus}(a,b):\quad
\begin{cases}
\delta p=a.p-b.p\\
\delta\phi=\mathrm{Log}(b.q^{-1}\otimes a.q)\\
\delta v=a.v-b.v\\
\delta b_g=a.b_g-b.b_g\\
\delta b_a=a.b_a-b.b_a
\end{cases}
\tag{X.43}
$$

**命题 X.1**（对偶性）对任意 $a,b\in\mathcal M$，有

$$
\text{plus}(b,\text{minus}(a,b))=a
\tag{X.44}
$$

**证明** 对旋转分量：

$$
b.q\otimes\mathrm{Exp}\big(\mathrm{Log}(b.q^{-1}\otimes a.q)\big)=b.q\otimes b.q^{-1}\otimes a.q=a.q
$$

其余分量由加法/减法互逆即得。$\square$

### X.3.3 Ceres 流形雅可比

在 Ceres 中，`PoseManifold` 的环境维数为 7，切空间维数为 6。`PlusJacobian` 为 $7\times 6$：

$$
J_{\mathrm{plus}}=\begin{bmatrix}
I_3&0\\
0&\dfrac{\partial q}{\partial\delta\phi}
\end{bmatrix}
\tag{X.45}
$$

其中

$$
\frac{\partial q}{\partial\delta\phi}=\frac12
\begin{bmatrix}
-q_x&-q_y&-q_z\\
q_w&-q_z&q_y\\
q_z&q_w&-q_x\\
-q_y&q_x&q_w
\end{bmatrix}
\tag{X.46}
$$

`MinusJacobian` 为 $6\times 7$：

$$
J_{\mathrm{minus}}=\begin{bmatrix}
I_3&0\\
0&J_q
\end{bmatrix},\qquad
J_q=2\begin{bmatrix}
-q_y&q_x&-q_w&q_z\\
-q_z&q_w&q_x&-q_y\\
-q_w&-q_z&q_y&q_x
\end{bmatrix}
\tag{X.47}
$$

---

## X.4 观测因子建模

### X.4.1 GNSS 位置因子

观测模型：

$$
p_{\text{gnss}}=p+n_p,\qquad n_p\sim\mathcal N(0,\Sigma_p)
\tag{X.48}
$$

残差：

$$
r_p=\Sigma_p^{-1/2}(p-p_{\text{gnss}})
\tag{X.49}
$$

参数块：pose(7)。残差维数 3。对 $p$ 的雅可比为 $\Sigma_p^{-1/2}$，对 $q,v,b_g,b_a$ 为 0。

### X.4.2 GNSS 速度因子

观测模型：

$$
v_{\text{gnss}}=v+n_v,\qquad n_v\sim\mathcal N(0,\Sigma_v)
\tag{X.50}
$$

残差：

$$
r_v=\Sigma_v^{-1/2}(v-v_{\text{gnss}})
\tag{X.51}
$$

参数块：mix(10)。残差维数 3。对 $v$ 的雅可比为 $\Sigma_v^{-1/2}$，对其余分量为 0。加入 Huber 核可抑制野值。

### X.4.3 GNSS 航向因子

观测模型：

$$
\psi_{\text{gnss}}=\text{yaw}(q)+n_\psi,\qquad n_\psi\sim\mathcal N(0,\sigma_\psi^2)
\tag{X.52}
$$

其中 $\text{yaw}(q)$ 由式 (X.1) 给出：

$$
\text{yaw}(q)=\operatorname{atan2}(-R_{01},R_{11})
=\operatorname{atan2}\big(2(q_0q_3-q_1q_2),\ q_0^2-q_1^2+q_2^2-q_3^2\big)
\tag{X.53}
$$

残差：

$$
r_\psi=\frac{1}{\sigma_\psi}\operatorname{wrap}\big(\text{yaw}(q)-\psi_{\text{gnss}}\big)
\tag{X.54}
$$

其中角度规整

$$
\operatorname{wrap}(d)=\operatorname{atan2}(\sin d,\cos d)
\tag{X.55}
$$

参数块：pose(7)。残差维数 1。

**定理 X.2**（航向对右扰动的雅可比）在右扰动 $q'=q\otimes\mathrm{Exp}(\delta\phi)$ 下，航向对 $\delta\phi$ 的雅可比为

$$
\frac{\partial\text{yaw}}{\partial\delta\phi}
=\frac{1}{R_{01}^2+R_{11}^2}
\big(-R_{11}R_{02}+R_{01}R_{12},\ 0,\ R_{11}R_{00}-R_{01}R_{10}\big)
\tag{X.56}
$$

进一步可化简为

$$
\frac{\partial\text{yaw}}{\partial\delta\phi}
=\left(-\frac{\tan(\text{roll})}{\cos(\text{pitch})},\ 0,\ \frac{\cos(\text{roll})}{\cos(\text{pitch})}\right)
\tag{X.57}
$$

**证明** 记 $N=-R_{01}$，$D=R_{11}$，则 $\text{yaw}=\operatorname{atan2}(N,D)$。由链式法则

$$
\frac{\partial\text{yaw}}{\partial\delta\phi}
=\frac{D\frac{\partial N}{\partial\delta\phi}-N\frac{\partial D}{\partial\delta\phi}}{N^2+D^2}
$$

在右扰动下，$R'=R(I+[\delta\phi]_\times)$，故

$$
\frac{\partial R(i,1)}{\partial\delta\phi}=\big(R(i,2),0,-R(i,0)\big)
$$

代入 $N,D$ 得

$$
\frac{\partial N}{\partial\delta\phi}=(-R_{02},0,R_{00}),\qquad
\frac{\partial D}{\partial\delta\phi}=(R_{12},0,-R_{10})
$$

故

$$
\frac{\partial\text{yaw}}{\partial\delta\phi}
=\frac{1}{R_{01}^2+R_{11}^2}
\big(-R_{11}R_{02}+R_{01}R_{12},\ 0,\ R_{11}R_{00}-R_{01}R_{10}\big)
$$

即式 (X.56)。再将 $R$ 的欧拉角展开式代入，$R_{01}^2+R_{11}^2=c_i^2$，第一分量

$$
-R_{11}R_{02}+R_{01}R_{12}=-c_is_j
$$

第三分量

$$
R_{11}R_{00}-R_{01}R_{10}=c_ic_j
$$

得到式 (X.57)。$\square$

**注 X.2** 式 (X.57) 中第二分量为 0，表明绕机体横轴旋转不改变航向；pitch $\to\pm90°$ 时雅可比发散，对应万向节锁。

### X.4.4 静止零速因子（ZUPT）

观测模型：

$$
v_{\text{zupt}}=0+n_v,\qquad n_v\sim\mathcal N(0,\sigma_{\text{zupt}}^2I)
\tag{X.58}
$$

残差：

$$
r_{\text{zupt}}=\frac{1}{\sigma_{\text{zupt}}}v
\tag{X.59}
$$

参数块：mix(10)。残差维数 3。触发条件：滑动窗口内陀螺与加表去均值 RMS 小于阈值：

$$
\sigma_\omega<\tau_\omega,\qquad \sigma_a<\tau_a
\tag{X.60}
$$

**注 X.3** 静止时 ZUPT 使速度被绝对约束，进而使零偏可观测。

### X.4.5 里程计速度因子

里程计测量前向位移 $dS$，速度模型为

$$
v_{\text{odo}}^b=(1+s_{\text{odo}})\frac{dS}{dt}C_b^m(\boldsymbol\alpha)\mathbf e_x+\boldsymbol\omega_{ib}^b\times\mathbf l_{\text{OD}}
\tag{X.61}
$$

其中安装角 $\boldsymbol\alpha=[\alpha_p,0,\alpha_y]^T$，且

$$
C_b^m\mathbf e_x=
\begin{bmatrix}
\cos\alpha_y\cos\alpha_p\\
-\sin\alpha_y\cos\alpha_p\\
-\sin\alpha_p
\end{bmatrix}
\tag{X.62}
$$

观测模型：

$$
v_{\text{odo}}^n=R_b^n v_{\text{odo}}^b+n_{\text{odo}},\qquad n_{\text{odo}}\sim\mathcal N(0,\sigma_{\text{odo}}^2I)
\tag{X.63}
$$

残差：

$$
r_{\text{odo}}=R_b^n v_{\text{odo}}^b-v^n
\tag{X.64}
$$

参数块：pose(7)、mix(10)。残差维数 3。刻度因子 $s_{\text{odo}}$ 位于 mix[9]，加性更新。

**注 X.4**（可观性）$s_{\text{odo}}$ 与速度耦合，$\alpha_y$ 与航向耦合，$\alpha_p$ 与俯仰耦合，需 GNSS 速度/位置/航向或双天线航向才能分离。

---

## X.5 边缘化理论

### X.5.1 法方程累积

对每个残差块 $r_k(x)$ 在当前线性化点处展开

$$
r_k(x+\delta x)\approx r_k(x)+J_k\delta x
\tag{X.65}
$$

高斯牛顿最小化 $\sum_k\|r_k+J_k\delta x\|^2$ 得到法方程

$$
H\delta x=b,\qquad H=\sum_k J_k^TJ_k,\qquad b=\sum_k J_k^Tr_k
\tag{X.66}
$$

### X.5.2 Schur 补

将状态分为待边缘化变量 $\delta x_m$ 与保留变量 $\delta x_r$：

$$
\begin{bmatrix}
H_{mm}&H_{mr}\\
H_{mr}^T&H_{rr}
\end{bmatrix}
\begin{bmatrix}
\delta x_m\\
\delta x_r
\end{bmatrix}
=
\begin{bmatrix}
b_m\\
b_r
\end{bmatrix}
\tag{X.67}
$$

由第一行消去 $\delta x_m$：

$$
H^*\delta x_r=b^*
\tag{X.68}
$$

其中

$$
H^*=H_{rr}-H_{mr}^TH_{mm}^{-1}H_{mr}
\tag{X.69}
$$

$$
b^*=b_r-H_{mr}^TH_{mm}^{-1}b_m
\tag{X.70}
$$

若 $H_{mm}$ 非正定，采用特征分解伪逆：

$$
H_{mm}^{-1}\approx V D^{-1}V^T
\tag{X.71}
$$

仅保留 $|d_i|>\epsilon\max_j|d_j|$ 的特征方向。

### X.5.3 先验恢复

将 $(H^*,b^*)$ 分解为 $H^*=J_{\text{out}}^TJ_{\text{out}}$，$b^*=J_{\text{out}}^Tr_{\text{out}}$。若 Cholesky 分解 $H^*=LL^T$，则取

$$
J_{\text{out}}=L^T,\qquad r_{\text{out}}=L^{-1}b^*
\tag{X.72}
$$

若用特征分解 $H^*=VDV^T$，则取

$$
J_{\text{out}}=D^{1/2}V^T,\qquad r_{\text{out}}=D^{-1/2}V^Tb^*
\tag{X.73}
$$

先验因子残差为

$$
r_{\text{prior}}(\delta x)=J_{\text{prior}}\delta x-r_{\text{prior}}
\tag{X.74}
$$

**命题 X.2**（先验一致性）由式 (X.72) 或 (X.73) 定义的 $(J_{\text{out}},r_{\text{out}})$ 满足

$$
J_{\text{out}}^TJ_{\text{out}}=H^*,\qquad J_{\text{out}}^Tr_{\text{out}}=b^*
\tag{X.75}
$$

**证明** 对式 (X.72)：$J_{\text{out}}^TJ_{\text{out}}=LL^T=H^*$，$J_{\text{out}}^Tr_{\text{out}}=LL^{-1}b^*=b^*$。对式 (X.73) 同理。$\square$

---

## X.6 整体优化问题与求解

### X.6.1 最大后验估计

给定所有观测 $\mathcal Z=\{z_k\}$，状态 $x$ 的最大后验估计为

$$
x^*=\arg\max_x\ p(x|\mathcal Z)=\arg\max_x\ p(x)p(\mathcal Z|x)
\tag{X.76}
$$

在高斯假设下等价于非线性最小二乘

$$
x^*=\arg\min_x\ \frac12\sum_k\|r_k(x)\|^2
\tag{X.77}
$$

其中残差按观测协方差白化。

### X.6.2 整体残差

综合各因子，整体残差为

$$
r(x)=\begin{bmatrix}
\{r_{\text{imu},k}\}_{k=0}^{N-2}\\
\{r_{\text{gnss},k}^{\text{pos}}\}_{k\in\mathcal G_p}\\
\{r_{\text{gnss},k}^{\text{vel}}\}_{k\in\mathcal G_v}\\
\{r_{\text{gnss},k}^{\text{yaw}}\}_{k\in\mathcal G_\psi}\\
\{r_{\text{zupt},k}\}_{k\in\mathcal S}\\
\{r_{\text{odo},k}\}_{k\in\mathcal O}\\
r_{\text{prior}}
\end{bmatrix}
\tag{X.78}
$$

### X.6.3 Levenberg-Marquardt 迭代

在第 $t$ 次迭代，线性化并求解

$$
(H+\lambda I)\delta x=-b
\tag{X.79}
$$

更新

$$
x^{(t+1)}=x^{(t)}\boxplus\delta x
\tag{X.80}
$$

其中 $\boxplus$ 为式 (X.42) 定义的流形加法。

### X.6.4 滑窗与边缘化

窗口满时，将最旧帧的切空间变量作为 $\delta x_m$ 消去，得到先验 $(J_{\text{out}},r_{\text{out}})$，加入下一轮优化。状态维数保持固定，计算量有界。

---

## X.7 本章小结

本章建立了 GNSS/INS 滑窗图优化松组合的完整理论：

1. **预积分**：在 $b_i$ 系中定义 $\Delta p,\Delta v,\Delta q$，推导中值离散与误差传播，得到残差 (X.34)–(X.37)；
2. **流形**：采用右扰动，定义 $\boxplus,\boxminus$ 与 Ceres 雅可比 (X.45)–(X.47)；
3. **观测因子**：推导 GNSS 位置/速度/航向、ZUPT、里程计速度的残差与雅可比，其中航向雅可比 (X.57) 具有简洁闭式；
4. **边缘化**：Schur 补 (X.69)–(X.70) 与先验恢复 (X.72)–(X.73)；
5. **整体优化**：MAP 估计 (X.76)、整体残差 (X.78)、LM 迭代 (X.79)。

该框架在流形上统一处理旋转，利用 Ceres AutoDiff 自动求导，通过滑窗与边缘化实现固定计算量的实时估计。