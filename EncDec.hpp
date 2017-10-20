#pragma once

#include "LSTM.hpp"
#include "Vocabulary.hpp"
#include "SoftMax.hpp"
#include <iostream>
#include <string>
#include <sstream>
class EncDec{
public:
	class Data;
	class Grad;
	class DecCandidate;
	class ThreadArg;
    class ThreadArg_2;
	class ThreadArg_3;

	class ThreadTimer; // for experiment to record the time
	// class MemoryFootprint; // for experiment to record the memory footprint
	
  EncDec(Vocabulary& sourceVoc_, Vocabulary& targetVoc, std::vector<EncDec::Data*>& trainData_, std::vector<EncDec::Data*>& devData_,const int inputDim, const int hiddenDim);

  Rand rnd;
  Vocabulary& sourceVoc;
  Vocabulary& targetVoc;
  std::vector<EncDec::Data*>& trainData;
  std::vector<EncDec::Data*>& devData;
  LSTM enc, dec;
  SoftMax softmax;
  MatD sourceEmbed;
  MatD targetEmbed;
  VecD zeros;

  std::vector<std::vector<LSTM::State*> > encStateDev, decStateDev;

  void encode(const std::vector<int>& src, std::vector<LSTM::State*>& encState);
  void encode_qiao(const std::vector<int>& src, std::vector<LSTM::State*>& encState);
  /* encode function created by qiaoyc for memory footprint record */
  void encode_mf_v1(const std::vector<int>& src, std::vector<LSTM::State*>& encState, MemoryFootprint* mf);

  void translate(const std::vector<int>& src, const int beam = 1, const int maxLength = 100, const int showNum = 1);
  bool translate(std::vector<int>& output, const std::vector<int>& src, const int beam = 1, const int maxLength = 100);
  Real calcLoss(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState);
  Real calcPerplexity(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState);
  void gradCheck(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad);
  void gradCheck(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, MatD& param, const MatD& grad);
	void train(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss);
  
	/* train function created by qiaoyc for experiments */
	void train_qiao_1(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss);
	void train_qiao_2(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<double>& timeRecorder);
	void train_qiao_3(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<double>& timeRecorder);
    void train_qiao_4(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<double>& timeRecorder, std::vector<VecD>& target_dist, std::vector<VecD>& delosBuffer, std::vector<VecD>& delisBuffer, std::vector<VecD>& delusBuffer, std::vector<VecD>& delfsBuffer);
	void train_qiao_5(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<double>& timeRecorder, std::vector<VecD>& target_dist, MatD& delosBuffer, MatD& delisBuffer, MatD& delusBuffer, MatD& delfsBuffer);

	/* train function created by qiaoyc for experiments for memory footprint */
	void train_mf_1(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, MemoryFootprint* mf);	
	void train_mf_2(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, MemoryFootprint* mf);	
	void train_mf_3(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<double>& timeRecoder, std::vector<VecD>& target_dist, MatD& delosBuffer, MatD& delisBuffer, MatD& delusBuffer, MatD& delfsBuffer);
	
	// new train function created ny qiaoyc for practical usage, no time recorder */
	void train_new_v1(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<VecD>& target_dist, std::vector<VecD>& delosBuffer, std::vector<VecD>& delisBuffer, std::vector<VecD>& delusBuffer, std::vector<VecD>& delfsBuffer);
	void train_new_v2(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<VecD>& target_dist, MatD& delosBuffer, MatD& delisBuffer, MatD& delusBuffer, MatD& delfsBuffer);

    void trainOpenMP(const Real learningRate, const int miniBatchSize = 1, const int numThreads = 1);
	
	/* trainOpenMP function created by qiaoyc for experiments */	
	void trainOpenMP_qiao(const Real learningRate, const int miniBatchSize = 1, const int numThreads = 1);
    void trainOpenMP_qiao_2(const Real learningRate, const int miniBatchSize = 1, const int numThreads = 1);
    void trainOpenMP_qiao_3(const Real learningRate, const int miniBatchSize = 1, const int numThreads = 1);

	/* trainOpenMP function created by qiaoyc for experiments for memory footprint */
	void trainOpenMP_mf_1(const Real learningRate, const int miniBatchSize = 1, const int numThreads = 1);
	void trainOpenMP_mf_2(const Real learningRate, const int miniBatchSize = 1, const int numThreads = 1);

	/* trainOpenMP function created by qiaoyc for practical usage, no time recorder*/
	void trainOpenMP_new_v1(const Real learningRate, const int miniBatchSize = 1, const int numThreads = 1);
	void trainOpenMP_new_v2(const Real learningRate, const int miniBatchSize = 1, const int numThreads = 2);
	
	

	void save(const std::string& fileName);
	void load(const std::string& fileName);
	
	static void demo(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev);
	/* demo function created by qiaoyc for experiments, with time recorders */	
	static void demo_qiao(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads);
    static void demo_qiao_2(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads);
	static void demo_qiao_3(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads);

	/* demo function created by qiaoyc for experiments for memory footprint */
	static void demo_mf_1(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads);
	static void demo_mf_2(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads);
	/* demo function created by qiaoyc for practical usage, no time recorder */
	static void demo_new_v1(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads);
	static void demo_new_v2(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads);

};
class EncDec::Data{
public:
    std::vector<int> src, tgt;
};

class EncDec::Grad{
public:
  std::unordered_map<int, VecD> sourceEmbed, targetEmbed;
  LSTM::Grad lstmSrcGrad;
  LSTM::Grad lstmTgtGrad;
  SoftMax::Grad softmaxGrad;

  // <??> PART I: this part may be the bottleneck
  // <??> PART I: BEGIN
  void init(){
	  //std::cout << "1" << std::endl;
	  this->sourceEmbed.clear(); // destructor are called, all the elements in the unordered_map are dropped
		//std::cout << "2" << std::endl;
	  this->targetEmbed.clear(); // destructor are called, all the elements in the unordered_map are dropped
		//std::cout << "3" << std::endl;
	  this->lstmSrcGrad.init(); // all the matrix and vector set to zero
		//std::cout << "4" << std::endl;
	  this->lstmTgtGrad.init(); // all the matrix and vector set to zero
		//std::cout << "5" << std::endl;
	  this->softmaxGrad.init(); // all the matrix and vector set to zero
	//std::cout << "6" << std::endl;
  }
  void init_qiao()
  {
	  // just set all the elements in sourceEmbed to 0
	  for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it)
	  {
		  it->second.setZero();
	  }
	  // just set all the elements in targetEmbed to 0
	  for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it)
	  {
		  it->second.setZero();
	  }

	  this->lstmSrcGrad.init();
	  this->lstmTgtGrad.init();
	  this->softmaxGrad.init();
  }
  // <??> PART I: END

  Real norm(){
    Real res = this->lstmSrcGrad.norm()+this->lstmTgtGrad.norm()+this->softmaxGrad.norm();
	for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it){
      res += it->second.squaredNorm();
    }
    for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it){
      res += it->second.squaredNorm();
    }

    return res;
  }

  void operator += (const EncDec::Grad& grad){
    this->lstmSrcGrad += grad.lstmSrcGrad;
    this->lstmTgtGrad += grad.lstmTgtGrad;
    this->softmaxGrad += grad.softmaxGrad;

    for (auto it = grad.sourceEmbed.begin(); it != grad.sourceEmbed.end(); ++it){
      if (this->sourceEmbed.count(it->first)){
	this->sourceEmbed.at(it->first) += it->second;
      }
      else {
	this->sourceEmbed[it->first] = it->second;
      }
    }
    for (auto it = grad.targetEmbed.begin(); it != grad.targetEmbed.end(); ++it){
      if (this->targetEmbed.count(it->first)){
	this->targetEmbed.at(it->first) += it->second;
      }
      else {
	this->targetEmbed[it->first] = it->second;
      }
    }
  }

  void operator /= (const Real val){
    this->lstmSrcGrad /= val;
    this->lstmTgtGrad /= val;
    this->softmaxGrad /= val;

    for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it){
      it->second /= val;
    }
    for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it){
      it->second /= val;
    }
  }
};

class EncDec::DecCandidate{
public:
  DecCandidate():
    score(0.0), stop(false)
  {}

  Real score;
  std::vector<int> tgt;
  std::vector<LSTM::State*> decState;
  bool stop;
};

class EncDec::ThreadArg{
public:
  ThreadArg(EncDec& encdec_):
    encdec(encdec_), loss(0.0)
  {
    this->grad.lstmSrcGrad = LSTM::Grad(this->encdec.enc);
    this->grad.lstmTgtGrad = LSTM::Grad(this->encdec.dec);
    this->grad.softmaxGrad = SoftMax::Grad(this->encdec.softmax);
  };

  int beg, end;
  EncDec& encdec;
  EncDec::Grad grad;
  Real loss;
  std::vector<LSTM::State*> encState, decState;
};

// added by qiao for new thread args that contains temp buffer for delc and so on
class EncDec::ThreadArg_2{
public:
    ThreadArg_2(EncDec& encdec_):encdec(encdec_), loss(0.0)
    {
        this->grad.lstmSrcGrad = LSTM::Grad(this->encdec.enc);
        this->grad.lstmTgtGrad = LSTM::Grad(this->encdec.dec);
        this->grad.softmaxGrad = SoftMax::Grad(this->encdec.softmax);
    };

    int beg, end;
    EncDec& encdec;
    EncDec::Grad grad;
    Real loss;
    std::vector<LSTM::State*> encState, decState;

    // buffer for smart cache usage
    std::vector<VecD> target_dist;
    std::vector<VecD> delosBuffer;
    std::vector<VecD> delisBuffer;
    std::vector<VecD> delusBuffer;
    std::vector<VecD> delfsBuffer;

    std::vector<VecD> deltaFeatureBuffer;
    std::vector<VecD> gradWeightBuffer;
};

class EncDec::ThreadArg_3{
public:
    ThreadArg_3(EncDec& encdec_):encdec(encdec_), loss(0.0)
    {
        this->grad.lstmSrcGrad = LSTM::Grad(this->encdec.enc);
        this->grad.lstmTgtGrad = LSTM::Grad(this->encdec.dec);
        this->grad.softmaxGrad = SoftMax::Grad(this->encdec.softmax);
    };

    int beg, end;
    EncDec& encdec;
    EncDec::Grad grad;
    Real loss;
    std::vector<LSTM::State*> encState, decState;

    // buffer for smart cache usage
    std::vector<VecD> target_dist;
    MatD delosBuffer;
    MatD delisBuffer;
    MatD delusBuffer;
    MatD delfsBuffer;

    std::vector<VecD> deltaFeatureBuffer;
    std::vector<VecD> gradWeightBuffer;
};

// add by korchaign for time record in each thread 
class EncDec::ThreadTimer 
{
public:
	ThreadTimer(EncDec& encdec_, int size):encdec(encdec_)
	{
		for (int i = 0; i < size; i ++)
		{
			timeRecorder.push_back(0.0);
		}
	};

	void init()
	{
		for (int i = 0; i < timeRecorder.size(); i ++)
		{
			timeRecorder[i] = 0.0;
		}
	}
	int timeRecorderSize()
	{
		return (int)(timeRecorder.size());
	}
	EncDec& encdec;
	std::vector<double> timeRecorder;
}; // end of class EncDec::ThreadTimer
/*
class EncDec::MemoryFootprint
{
public:
	MemoryFootprint(EncDec& encdec_):encdec(encdec_)
	{

	};

	void init()
	{

	}

	EncDec& encdec;
	// address of parameters 
	Real*	
		addr_LSTM_Wxi, addr_LSTM_Whi, addr_LSTM_bi,
		addr_LSTM_Wxf, addr_LSTM_Whf, addr_LSTM_bf, 
		addr_LSTM_Wxo, addr_LSTM_Who, addr_LSTM_bo, 
		addr_LSTM_Wxu, addr_LSTM_Whu, addr_LSTM_bu;
	Real*	
		addr_LSTM_Grad_Wxi, addr_LSTM_Grad_Whi, addr_LSTM_Grad_bi,
		addr_LSTM_Grad_Wxf, addr_LSTM_Grad_Whf, addr_LSTM_Grad_bf, 
		addr_LSTM_Grad_Wxo, addr_LSTM_Grad_Who, addr_LSTM_Grad_bo, 
		addr_LSTM_Grad_Wxu, addr_LSTM_Grad_Whu, addr_LSTM_Grad_bu;
	Real*
		addr_Softmax_weight, addr_Softmax_bias;
	Real*
		addr_Softmax_Grad_weight, addr_Softmax_Grad_bias;

	// time start point of memory access 
	std::vector<struct timeval> 
		time_s_LSTM_Wxi, time_s_LSTM_Whi, time_s_LSTM_bi, 
		time_s_LSTM_Wxf, time_s_LSTM_Whf, time_s_LSTM_bf, 
		time_s_LSTM_Wxo, time_s_LSTM_Who, time_s_LSTM_bo, 
		time_s_LSTM_Wxu, time_s_LSTM_Whu, time_s_LSTM_bu;
	std::vector<struct timeval> 
		time_s_LSTM_Grad_Wxi, time_s_LSTM_Grad_Whi, time_s_LSTM_Grad_bi, 
		time_s_LSTM_Grad_Wxf, time_s_LSTM_Grad_Whf, time_s_LSTM_Grad_bf, 
		time_s_LSTM_Grad_Wxo, time_s_LSTM_Grad_Who, time_s_LSTM_Grad_bo, 
		time_s_LSTM_Grad_Wxu, time_s_LSTM_Grad_Whu, time_s_LSTM_Grad_bu;
	std::vector<struct timeval>
		time_s_Softmax_weight, time_s_Softmax_bias;
	std::vector<struct timeval>
		time_s_Softmax_Grad_weight, time_s_Softmax_Grad_bias;

	// time end point of memory access 
	std::vector<struct timeval> 
		time_e_LSTM_Wxi, time_e_LSTM_Whi, time_e_LSTM_bi, 
		time_e_LSTM_Wxf, time_e_LSTM_Whf, time_e_LSTM_bf, 
		time_e_LSTM_Wxo, time_e_LSTM_Who, time_e_LSTM_bo, 
		time_e_LSTM_Wxu, time_e_LSTM_Whu, time_e_LSTM_bu;
	std::vector<struct timeval>
		time_e_LSTM_Grad_Wxi, time_e_LSTM_Grad_Whi, time_e_LSTM_Grad_bi, 
		time_e_LSTM_Grad_Wxf, time_e_LSTM_Grad_Whf, time_e_LSTM_Grad_bf, 
		time_e_LSTM_Grad_Wxo, time_e_LSTM_Grad_Who, time_e_LSTM_Grad_bo, 
		time_e_LSTM_Grad_Wxu, time_e_LSTM_Grad_Whu, time_e_LSTM_Grad_bu;
	std::vector<struct timeval>
		time_e_Softmax_weight, time_e_Softmax_bias;
	std::vector<struct timeval>
		time_e_Softmax_Grad_weight, time_e_Softmax_Grad_bias;
	
};
*/
