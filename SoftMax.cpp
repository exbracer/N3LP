#include "SoftMax.hpp"
#include "Utils.hpp"
#include <sys/time.h>
// #include <iostream>

void SoftMax::calcDist(const VecD& input, VecD& output){
  output = this->bias;
  output.noalias() += this->weight*input;
  output.array() -= output.maxCoeff(); //for numerical stability
  output = output.array().exp();
  output /= output.array().sum();
}

Real SoftMax::calcLoss(const VecD& output, const int label){
  return -log(output.coeff(label, 0));
}

void SoftMax::backward(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, SoftMax::Grad& grad){
  VecD delta = output;

  delta.coeffRef(label, 0) -= 1.0;
  deltaFeature = this->weight.transpose()*delta;
  /*
  std::cout << "==== during softmax backward ====" << std::endl;
  std::cout << "deltaFeature = " << deltaFeature.squaredNorm() << std::endl;
  std::cout << "delta = " << delta.squaredNorm() << std::endl; 
  std::cout << "this->weight = " << this->weight.squaredNorm() << std::endl;
  */
  grad.weight += delta*input.transpose();
  grad.bias += delta;
}

void SoftMax::sgd(const SoftMax::Grad& grad, const Real learningRate){
  this->weight -= learningRate*grad.weight;
  this->bias -= learningRate*grad.bias;
}

void SoftMax::save(std::ofstream& ofs){
  Utils::save(ofs, this->weight);
  Utils::save(ofs, this->bias);
}

void SoftMax::load(std::ifstream& ifs){
  Utils::load(ifs, this->weight);
  Utils::load(ifs, this->bias);
}

// for smart cache usage
void SoftMax::backward1(VecD& output, const int label, SoftMax::Grad& grad)
{
    // this grad belongs to the private softmax of each thread
    output.coeffRef(label, 0) -= 1.0;
    grad.bias += output;
}

void SoftMax::backward2(const VecD& output, VecD& deltaFeature)
{
    // the weight belongs to the shared softmax
    deltaFeature = this->weight.transpose()* output;
}

void SoftMax::backward3(VecD& input, const VecD& output, SoftMax::Grad& grad, int index)
{
    grad.weight.col(index) += output*input(index, 0);
}

void SoftMax::calcDist_mf_v1(const VecD& input, VecD& output, MemoryFootprint* mf)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);
	
	output = this->bias;
	
	gettimeofday(&end, NULL);
	mf->time_s_Softmax_bias.push_back(start);
	mf->time_e_Softmax_bias.push_back(end);

	gettimeofday(&start, NULL);
	
	output.noalias() += this->weight*input;
	
	gettimeofday(&end, NULL);
	mf->time_s_Softmax_weight.push_back(start);
	mf->time_e_Softmax_weight.push_back(end);

	output.array() -= output.maxCoeff(); //for numerical stability
	output = output.array().exp();
	output /= output.array().sum();
}

Real SoftMax::calcLoss_mf_v1(const VecD& output, const int label, MemoryFootprint* mf){
  return -log(output.coeff(label, 0));
}

void SoftMax::backward_mf_v1(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, SoftMax::Grad& grad, MemoryFootprint* mf){
	VecD delta = output;

	delta.coeffRef(label, 0) -= 1.0;
	
	struct timeval start, end;
	gettimeofday(&start, NULL);

	deltaFeature = this->weight.transpose()*delta;
  
	gettimeofday(&end, NULL);
	mf->time_s_Softmax_weight.push_back(start);
	mf->time_e_Softmax_weight.push_back(end);
	/*
  std::cout << "==== during softmax backward ====" << std::endl;
  std::cout << "deltaFeature = " << deltaFeature.squaredNorm() << std::endl;
  std::cout << "delta = " << delta.squaredNorm() << std::endl; 
  std::cout << "this->weight = " << this->weight.squaredNorm() << std::endl;
  */
	gettimeofday(&start, NULL);
	grad.weight += delta*input.transpose();
	gettimeofday(&end, NULL);
	mf->time_s_Softmax_Grad_weight.push_back(start);
	mf->time_e_Softmax_Grad_weight.push_back(end);

	gettimeofday(&start, NULL);
	grad.bias += delta;
	gettimeofday(&end, NULL);
	mf->time_s_Softmax_Grad_bias.push_back(start);
	mf->time_e_Softmax_Grad_bias.push_back(end);
}
