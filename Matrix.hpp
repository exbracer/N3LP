#pragma once

#include <Eigen/Core>
//#define USE_FLOAT
#ifdef USE_FLOAT
typedef float Real;
typedef Eigen::MatrixXf MatD;
typedef Eigen::VectorXf VecD;
#else
typedef double Real;
typedef Eigen::MatrixXd MatD;
typedef Eigen::VectorXd VecD;
#endif

typedef Eigen::MatrixXi MatI;
