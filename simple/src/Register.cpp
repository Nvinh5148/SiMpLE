#include "Register.hpp"

#include <string>

#include <tbb/combinable.h>

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
}  // namespace

// Objective (same as SiMpLE Eq. 2-5): maximise sum_i exp(-d_i^2 / 2 sigma^2),
// i.e. minimise the Welsch loss sum_i sigma^2 (1 - exp(-d_i^2 / 2 sigma^2)) of the
// point-to-point residual r_i = T p_i - q_i, q_i = nearest map point.
// IRLS: w_i = exp(-d_i^2 / 2 sigma^2), H = sum w J^T J, g = sum w J^T r.
// Left perturbation T <- Exp(xi) T, so dr/dxi = [-[y]x, I] with y = T p.
double Register::solveGN_(const std::vector<Eigen::Vector4d> &scan, my_kd_tree_t *tree, Eigen::Matrix4d &T) {
    double reward = 0.0;
    int it = 0;
    for (; it < maxIters_; it++) {
        const Eigen::Matrix3d R = T.block<3,3>(0,0);
        const Eigen::Vector3d t = T.block<3,1>(0,3);
        tbb::combinable<NormalEq> acc;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, scan.size()), [&](const tbb::blocked_range<size_t> &rg) {
            NormalEq &ne = acc.local();
            for (size_t i = rg.begin(); i < rg.end(); i++) {
                Eigen::Vector3d y = R * scan[i].head<3>() + t;
                double q[3] = {y(0), y(1), y(2)};
                uint32_t idx;
                double d2;
                tree->knnSearch(q, 1, &idx, &d2);
                double w = exp(-d2 * rewardParam_);
                ne.reward += w;
                if (w < 1e-6) continue;
                const auto &m = tree->dataset_.pts[idx];
                Eigen::Vector3d r = y - Eigen::Vector3d(m.x, m.y, m.z);
                Eigen::Matrix<double, 3, 6> J;
                J.block<3,3>(0,0) = -skew(y);
                J.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
                ne.H.noalias() += w * J.transpose() * J;
                ne.g.noalias() += w * J.transpose() * r;
            }
        });
        NormalEq tot = acc.combine([](const NormalEq &a, const NormalEq &b) {
            NormalEq c; c.H = a.H + b.H; c.g = a.g + b.g; c.reward = a.reward + b.reward; return c;
        });
        reward = tot.reward;
        Eigen::Matrix<double, 6, 1> dx = tot.H.ldlt().solve(-tot.g);
        if (!dx.allFinite()) break;
        T = expSE3(dx) * T;
        // Stop on small update: rotation [rad] and translation [m].
        if (dx.head<3>().norm() < convTol_ * 1e-1 && dx.tail<3>().norm() < convTol_) { it++; break; }
    }
    totalIters_ += it;
    numRegs_++;
    return -reward;
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