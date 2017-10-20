#pragma once
#include "Matrix.hpp"
#include "MemoryFootprint.hpp"
#include "MemoryFootprint2.hpp"
class SoftMax{
public:
  SoftMax(){};
  SoftMax(const int inputDim, const int classNum):
    weight(MatD::Zero(classNum, inputDim)), bias(VecD::Zero(classNum))
  {};

  class Grad;

  MatD weight; VecD bias;

  void calcDist(const VecD& input, VecD& output);
  Real calcLoss(const VecD& output, const int label);
  void backward(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, SoftMax::Grad& grad);
    
    // for smart cache usage
    void backward1(VecD& output, const int label, SoftMax::Grad& grad);
    void backward2(const VecD& output, VecD& deltaFeature);
    void backward3(VecD& input, const VecD& output, SoftMax::Grad& grad, int index);
    
    void sgd(const SoftMax::Grad& grad, const Real learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);

	/* functions created by qiaoyc for experiments for memory footprint record */
  void calcDist_mf_v1(const VecD& input, VecD& output, MemoryFootprint* mf);
  Real calcLoss_mf_v1(const VecD& input, const int label, MemoryFootprint* mf);
  void backward_mf_v1(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, SoftMax::Grad& grad, MemoryFootprint* mf);
};

class SoftMax::Grad{
public:
  Grad(){}
  Grad(const SoftMax& softmax){
    this->weight = MatD::Zero(softmax.weight.rows(), softmax.weight.cols());
    this->bias = VecD::Zero(softmax.bias.rows());
  }

  MatD weight; VecD bias;

  void init(){
    this->weight.setZero();
    this->bias.setZero();
  }

  Real norm(){
    return this->weight.squaredNorm()+this->bias.squaredNorm();
  }

  void operator += (const SoftMax::Grad& grad){
    this->weight += grad.weight;
    this->bias += grad.bias;
  }

  void operator /= (const Real val){
    this->weight /= val;
    this->bias /= val;
  }
};
