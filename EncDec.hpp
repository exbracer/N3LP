#pragma once

#include "LSTM.hpp"
#include "Vocabulary.hpp"
#include "SoftMax.hpp"

class EncDec{
public:
  class Data;
  class Grad;
  class DecCandidate;
  class ThreadArg;

	// for trainOpenMP_qiao
	class ThreadTimer;

  EncDec(Vocabulary& sourceVoc_, Vocabulary& targetVoc_,
	 std::vector<EncDec::Data*>& trainData_, std::vector<EncDec::Data*>& devData_,
	 const int inputDim, const int hiddenDim);

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
  void translate(const std::vector<int>& src, const int beam = 1, const int maxLength = 100, const int showNum = 1);
  bool translate(std::vector<int>& output, const std::vector<int>& src, const int beam = 1, const int maxLength = 100);
  Real calcLoss(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState);
  Real calcPerplexity(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState);
  void gradCheck(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad);
  void gradCheck(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, MatD& param, const MatD& grad);
  void train(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss);
  void train_qiao_1(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss);
  void train_qiao_2(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<double>& timeRecorder);
  void train_qiao_3(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<double>& timeRecorder);
  void trainOpenMP(const Real learningRate, const int miniBatchSize = 1, const int numThreads = 1);
  void trainOpenMP_qiao(const Real learningRate, const int miniBatchSize = 1, const int numThreads = 1);
  void save(const std::string& fileName);
  void load(const std::string& fileName);
  static void demo(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev);
	static void demo_qiao(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads);
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
    this->sourceEmbed.clear(); // destructor are called, all the elements in the unordered_map are dropped
    this->targetEmbed.clear(); // destructor are called, all the elements in the unordered_map are dropped
    this->lstmSrcGrad.init(); // all the matrix and vector set to zero
    this->lstmTgtGrad.init(); // all the matrix and vector set to zero
    this->softmaxGrad.init(); // all the matrix and vector set to zero
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
};
