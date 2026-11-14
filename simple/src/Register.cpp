#include "Register.hpp"

#include <string>

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

Register::Register(double convergenceTol, double sigma, const std::string &solver, int maxIters)
    : convTol_(convergenceTol),
      registrationScore_(0.0),
      rewardParam_(pow(sigma,-2)/2),
      sigma_(sigma),
      useGN_(solver == "gn"),
      maxIters_(maxIters) {
}

namespace {
Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
    Eigen::Matrix3d S;
    S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return S;
}

// SE(3) exponential, xi = [phi; rho] (rotation first).
Eigen::Matrix4d expSE3(const Eigen::Matrix<double, 6, 1> &xi) {
    Eigen::Vector3d phi = xi.head<3>(), rho = xi.tail<3>();
    double th = phi.norm();
    Eigen::Matrix3d W = skew(phi), R, V;
    if (th < 1e-10) {
        R = Eigen::Matrix3d::Identity() + W;
        V = Eigen::Matrix3d::Identity() + 0.5 * W;
    } else {
        double a = sin(th) / th, b = (1 - cos(th)) / (th * th), c = (th - sin(th)) / (th * th * th);
        R = Eigen::Matrix3d::Identity() + a * W + b * W * W;
        V = Eigen::Matrix3d::Identity() + b * W + c * W * W;
    }
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = V * rho;
    return T;
}

struct NormalEq {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
    double reward = 0.0;
};

// Normal equations at pose T. Deterministic reduction (fixed split order),
// so results are repeatable run to run.
NormalEq buildNormalEq(const std::vector<Eigen::Vector4d> &scan, my_kd_tree_t *tree,
                       const Eigen::Matrix4d &T, double rewardParam, bool withJacobian) {
    const Eigen::Matrix3d R = T.block<3,3>(0,0);
    const Eigen::Vector3d t = T.block<3,1>(0,3);
    return tbb::parallel_deterministic_reduce(
        tbb::blocked_range<size_t>(0, scan.size(), 256), NormalEq(),
        [&](const tbb::blocked_range<size_t> &rg, NormalEq ne) {
            for (size_t i = rg.begin(); i < rg.end(); i++) {
                Eigen::Vector3d y = R * scan[i].head<3>() + t;
                double q[3] = {y(0), y(1), y(2)};
                uint32_t idx;
                double d2;
                tree->knnSearch(q, 1, &idx, &d2);
                double w = exp(-d2 * rewardParam);
                ne.reward += w;
                if (!withJacobian || w < 1e-6) continue;
                const auto &m = tree->dataset_.pts[idx];
                Eigen::Vector3d r = y - Eigen::Vector3d(m.x, m.y, m.z);
                Eigen::Matrix<double, 3, 6> J;
                J.block<3,3>(0,0) = -skew(y);
                J.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
                ne.H.noalias() += w * J.transpose() * J;
                ne.g.noalias() += w * J.transpose() * r;
            }
            return ne;
        },
        [](const NormalEq &a, const NormalEq &b) {
            NormalEq c; c.H = a.H + b.H; c.g = a.g + b.g; c.reward = a.reward + b.reward; return c;
        });
}
}  // namespace

// Objective (same as SiMpLE Eq. 2-5): maximise sum_i exp(-d_i^2 / 2 sigma^2),
// i.e. minimise the Welsch loss sum_i sigma^2 (1 - exp(-d_i^2 / 2 sigma^2)) of the
// point-to-point residual r_i = T p_i - q_i, q_i = nearest map point.
// IRLS: w_i = exp(-d_i^2 / 2 sigma^2), H = sum w J^T J, g = sum w J^T r.
// Left perturbation T <- Exp(xi) T, so dr/dxi = [-[y]x, I] with y = T p.
double Register::solveGN_(const std::vector<Eigen::Vector4d> &scan, my_kd_tree_t *tree, Eigen::Matrix4d &T) {
    // Plain IRLS steps are always uphill but short when the seed is far from the optimum:
    // consecutive steps then point the same way (cos ~ 0.99) and only gain ~1 cm each.
    // Accelerate: while consecutive steps stay aligned, double the step multiplier alpha.
    // Safeguard: an extrapolated step that lowers the reward is redone as a plain step,
    // so the reward still never decreases.
    const double kAlignCos = 0.9, kMaxAlpha = 16.0;
    typedef Eigen::Matrix<double, 6, 1> Vec6;

    NormalEq ne = buildNormalEq(scan, tree, T, rewardParam_, true);
    Vec6 dxPrev = Vec6::Zero();
    double alpha = 1.0;
    int it = 0;  // number of normal-equation builds after the first (= cost)
    while (it < maxIters_) {
        Vec6 dx = ne.H.ldlt().solve(-ne.g);
        if (!dx.allFinite()) break;
        // Stop on small update: rotation [rad] and translation [m].
        const bool converged = dx.head<3>().norm() < convTol_ * 1e-1 && dx.tail<3>().norm() < convTol_;
        const bool aligned = dx.dot(dxPrev) > kAlignCos * dx.norm() * dxPrev.norm();
        alpha = (aligned && !converged) ? std::min(2.0 * alpha, kMaxAlpha) : 1.0;

        Eigen::Matrix4d Tn = expSE3(alpha * dx) * T;
        NormalEq nn = buildNormalEq(scan, tree, Tn, rewardParam_, !converged);
        it++;
        if (alpha > 1.0 && nn.reward < ne.reward) {  // overshoot: fall back to the plain step
            alpha = 1.0;
            Tn = expSE3(dx) * T;
            nn = buildNormalEq(scan, tree, Tn, rewardParam_, true);
            it++;
        }
        T = Tn;
        ne = nn;
        dxPrev = dx;
        if (converged) break;
    }
    totalIters_ += it;
    numRegs_++;
    return -ne.reward;  // reward at the returned pose
}

double Register::getRegistrationScore() const {
    return registrationScore_;
}

column_vector Register::getRegResult() const {
    return regResult_;
}

void Register::registerScan(const std::vector<Eigen::Vector4d> &scan,
                            const NanoflannPointsContainer<double> &kdTreePts) {
    my_kd_tree_t *subMapKdTree = new my_kd_tree_t(3, kdTreePts, {10});
    ObjectiveFunction objFunc = ObjectiveFunction(rewardParam_, scan,
                                                  subMapKdTree);

    // Use the previous estimate and a constant velocity model to set the seed
    // for the registration.
    regResult_ = {seedConstVel_[0], seedConstVel_[1], seedConstVel_[2],
                  seedConstVel_[3], seedConstVel_[4], seedConstVel_[5]};

    // The parameter regResult is replaced with the current pose estimate.
    // Find the best solution to the objective function (Equation 2).
    // The large number (last parameter) is the maximum negative reward
    // required by dlib (never achieved or used).
    if (useGN_) {
        Eigen::Matrix4d T = utils::homogeneous(regResult_(0), regResult_(1), regResult_(2),
                                               regResult_(3), regResult_(4), regResult_(5));
        registrationScore_ = solveGN_(scan, subMapKdTree, T);
        std::vector<double> v = utils::hom2rpyxyz(T);
        regResult_ = {v[0], v[1], v[2], v[3], v[4], v[5]};
    } else {
        registrationScore_ = dlib::find_min_using_approximate_derivatives(
                                dlib::bfgs_search_strategy(),
                                dlib::objective_delta_stop_strategy(convTol_),
                                objFunc, regResult_, -10000000);
    }
    
    // calculate the seed for the next pose estimate using the constant
    // velocity model, Equation 6.
    seedConstVel_ = utils::hom2rpyxyz(utils::homogeneous(regResult_(0), regResult_(1), regResult_(2),
                                                         regResult_(3), regResult_(4), regResult_(5)) *
                                     (utils::homogeneous(prevResult_(0), prevResult_(1), prevResult_(2),
                                                        prevResult_(3), prevResult_(4), prevResult_(5)).inverse() *
                                      utils::homogeneous(regResult_(0), regResult_(1), regResult_(2),
                                                         regResult_(3), regResult_(4), regResult_(5))));
    
    // Save the current registration result for the next constant velocity model calculation.
    prevResult_ = regResult_;
    
    delete subMapKdTree;
}