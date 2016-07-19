#include "EncDec.hpp"
#include "Utils.hpp"
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <omp.h>

EncDec::EncDec(Vocabulary& sourceVoc_, Vocabulary& targetVoc_, std::vector<EncDec::Data*>& trainData_, std::vector<EncDec::Data*>& devData_, const int inputDim, const int hiddenDim):
  sourceVoc(sourceVoc_), targetVoc(targetVoc_), trainData(trainData_), devData(devData_)
{
  const Real scale = 0.1;

  this->enc = LSTM(inputDim, hiddenDim);
  this->enc.init(this->rnd, scale);
  this->dec = LSTM(inputDim, hiddenDim);
  this->dec.init(this->rnd, scale);
  this->sourceEmbed = MatD(inputDim, this->sourceVoc.tokenList.size());
  this->targetEmbed = MatD(inputDim, this->targetVoc.tokenList.size());
  this->rnd.uniform(this->sourceEmbed, scale);
  this->rnd.uniform(this->targetEmbed, scale);
  this->zeros = VecD::Zero(hiddenDim);

  //Bias 1
  this->enc.bf.fill(1.0);
  this->dec.bf.fill(1.0);

  this->softmax = SoftMax(hiddenDim, this->targetVoc.tokenList.size());

  for (int j = 0; j < (int)this->devData.size(); ++j){
    this->encStateDev.push_back(std::vector<LSTM::State*>());
    this->decStateDev.push_back(std::vector<LSTM::State*>());
    
    for (int i = 0; i < 200; ++i){
      this->encStateDev.back().push_back(new LSTM::State);
      this->decStateDev.back().push_back(new LSTM::State);
    }
  }
}

void EncDec::encode(const std::vector<int>& src, std::vector<LSTM::State*>& encState){
  encState[0]->h = this->zeros; // <??> can be faster by using setZeros?
  encState[0]->c = this->zeros; // <??> can be faster by using setZeros?

  for (int i = 0; i < (int)src.size(); ++i){
    this->enc.forward(this->sourceEmbed.col(src[i]), encState[i], encState[i+1]);
  }
}

struct sort_pred {
  bool operator()(const EncDec::DecCandidate left, const EncDec::DecCandidate right) {
    return left.score > right.score;
  }

  bool operator()(const EncDec::Data* left, const EncDec::Data* right) {
    return (left->src.size()+left->tgt.size()) < (right->src.size()+right->tgt.size());
    //return left->tgt.size() < right->tgt.size();
  }
};

void EncDec::translate(const std::vector<int>& src, const int beam, const int maxLength, const int showNum){
  const Real minScore = -1.0e+05;
  MatD score(this->targetEmbed.cols(), beam);
  VecD targetDist;
  std::vector<LSTM::State*> encState, stateList;
  std::vector<EncDec::DecCandidate> candidate(beam), candidateTmp(beam);

  for (int i = 0; i <= (int)src.size(); ++i){
    encState.push_back(new LSTM::State);
    stateList.push_back(encState.back());
  }

  this->encode(src, encState);

  for (int i = 0; i < maxLength; ++i){
    for (int j = 0; j < beam; ++j){
      if (candidate[j].stop){
	score.col(j).fill(candidate[j].score);
	continue;
      }

      candidate[j].decState.push_back(new LSTM::State);
      stateList.push_back(candidate[j].decState.back());

      if (i == 0){
	candidate[j].decState[i]->h = encState.back()->h;
	candidate[j].decState[i]->c = encState.back()->c;
      }
      else {
	this->dec.forward(this->targetEmbed.col(candidate[j].tgt[i-1]), candidate[j].decState[i-1], candidate[j].decState[i]);
      }

      this->softmax.calcDist(candidate[j].decState[i]->h, targetDist);
      score.col(j).array() = candidate[j].score+targetDist.array().log();
    }

    for (int j = 0, row, col; j < beam; ++j){
      score.maxCoeff(&row, &col);
      candidateTmp[j] = candidate[col];
      candidateTmp[j].score = score.coeff(row, col);

      if (candidateTmp[j].stop){
	score.col(col).fill(minScore);
	continue;
      }

      if (row == this->targetVoc.eosIndex){
	candidateTmp[j].stop = true;
      }

      candidateTmp[j].tgt.push_back(row);

      if (i == 0){
	score.row(row).fill(minScore);
      }
      else {
	score.coeffRef(row, col) = minScore;
      }
    }

    candidate = candidateTmp;
    std::sort(candidate.begin(), candidate.end(), sort_pred());

    if (candidate[0].tgt.back() == this->targetVoc.eosIndex){
      break;
    }
  }

  if (showNum <= 0){
    return;
  }

  for (auto it = src.begin(); it != src.end(); ++it){
    std::cout << this->sourceVoc.tokenList[*it]->str << " ";
  }
  std::cout << std::endl;

  for (int i = 0; i < showNum; ++i){
    std::cout << i+1 << " (" << candidate[i].score << "): ";
    for (auto it = candidate[i].tgt.begin(); it != candidate[i].tgt.end(); ++it){
      std::cout << this->targetVoc.tokenList[*it]->str << " ";
    }
    std::cout << std::endl;
  }

  for (auto it = stateList.begin(); it != stateList.end(); ++it){
    delete *it;
  }
}

bool EncDec::translate(std::vector<int>& output, const std::vector<int>& src, const int beam, const int maxLength){
  const Real minScore = -1.0e+05;
  MatD score(this->targetEmbed.cols(), beam);
  VecD targetDist;
  std::vector<LSTM::State*> encState, stateList;
  std::vector<EncDec::DecCandidate> candidate(beam), candidateTmp(beam);

  for (int i = 0; i <= (int)src.size(); ++i){
    encState.push_back(new LSTM::State);
    stateList.push_back(encState.back());
  }

  this->encode(src, encState);

  for (int i = 0; i < maxLength; ++i){
    for (int j = 0; j < beam; ++j){
      if (candidate[j].stop){
	score.col(j).fill(candidate[j].score);
	continue;
      }

      candidate[j].decState.push_back(new LSTM::State);
      stateList.push_back(candidate[j].decState.back());

      if (i == 0){
	candidate[j].decState[i]->h = encState.back()->h;
	candidate[j].decState[i]->c = encState.back()->c;
      }
      else {
	this->dec.forward(this->targetEmbed.col(candidate[j].tgt[i-1]), candidate[j].decState[i-1], candidate[j].decState[i]);
      }

      this->softmax.calcDist(candidate[j].decState[i]->h, targetDist);
      score.col(j).array() = candidate[j].score+targetDist.array().log();
    }

    for (int j = 0, row, col; j < beam; ++j){
      score.maxCoeff(&row, &col);
      candidateTmp[j] = candidate[col];
      candidateTmp[j].score = score.coeff(row, col);

      if (candidateTmp[j].stop){
	score.col(col).fill(minScore);
	continue;
      }

      if (row == this->targetVoc.eosIndex){
	candidateTmp[j].stop = true;
      }

      candidateTmp[j].tgt.push_back(row);

      if (i == 0){
	score.row(row).fill(minScore);
      }
      else {
	score.coeffRef(row, col) = minScore;
      }
    }

    candidate = candidateTmp;
    std::sort(candidate.begin(), candidate.end(), sort_pred());

    if (candidate[0].tgt.back() == this->targetVoc.eosIndex){
      break;
    }
  }

  bool status;

  output.clear();

  if (candidate[0].tgt.back() == this->targetVoc.eosIndex){
    for (int i = 0; i < (int)candidate[0].tgt.size()-1; ++i){
      output.push_back(candidate[0].tgt[i]);
    }

    status = true;
  }
  else {
    output = candidate[0].tgt;
    status = false;
  }

  for (auto it = stateList.begin(); it != stateList.end(); ++it){
    delete *it;
  }

  return status;
}

Real EncDec::calcLoss(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState){
  VecD targetDist;
  Real loss = 0.0;

  this->encode(data->src, encState);

  for (int i = 0; i < (int)data->tgt.size(); ++i){
    if (i == 0){
      decState[0]->h = encState[data->src.size()]->h;
      decState[0]->c = encState[data->src.size()]->c;
    }
    else {
      this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), decState[i-1], decState[i]);
    }

    this->softmax.calcDist(decState[i]->h, targetDist);
    loss += this->softmax.calcLoss(targetDist, data->tgt[i]);
  }
  
  return loss;
}

Real EncDec::calcPerplexity(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState){
  VecD targetDist;
  Real perp = 0.0;

  this->encode(data->src, encState);

  for (int i = 0; i < (int)data->tgt.size(); ++i){
    if (i == 0){
      decState[0]->h = encState[data->src.size()]->h;
      decState[0]->c = encState[data->src.size()]->c;
    }
    else {
      this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), decState[i-1], decState[i]);
    }

    this->softmax.calcDist(decState[i]->h, targetDist);
    perp -= log(targetDist.coeff(data->tgt[i], 0));
  }
  
  return exp(perp/data->tgt.size());
}

void EncDec::gradCheck(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, MatD& param, const MatD& grad){
  const Real EPS = 1.0e-04;
  Real val = 0.0, objPlus = 0.0, objMinus = 0.0;

  for (int i = 0; i < param.rows(); ++i){
    for (int j = 0; j < param.cols(); ++j){
      val = param.coeff(i, j);
      param.coeffRef(i, j) = val+EPS;
      objPlus = this->calcLoss(data, encState, decState);
      param.coeffRef(i, j) = val-EPS;
      objMinus = this->calcLoss(data, encState, decState);
      param.coeffRef(i, j) = val;

      std::cout << "Grad: " << grad.coeff(i, j) << std::endl;
      std::cout << "Enum: " << (objPlus-objMinus)/(2.0*EPS) << std::endl;
    }
  }
}

void EncDec::gradCheck(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad){
  std::cout << "Gradient checking..." << std::endl;
  this->gradCheck(data, encState, decState, this->enc.Whi, grad.lstmSrcGrad.Whi);
}

void EncDec::train(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss){
  VecD targetDist; // <??> created in stack of this thread

  loss = 0.0;
  this->encode(data->src, encState);

  for (int i = 0; i < (int)data->tgt.size(); ++i){
    if (i == 0){
      decState[0]->h = encState[data->src.size()]->h;
      decState[0]->c = encState[data->src.size()]->c;
    }
    else {
      this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), decState[i-1], decState[i]);
    }

    // <??> PART S: THIS PART MAY BE THE BOTTLENEC
    // <??> PART S: BEGIN
    this->softmax.calcDist(decState[i]->h, targetDist); // <??> are each thead competing for softmax
    loss += this->softmax.calcLoss(targetDist, data->tgt[i]); // <??> same question as above
    this->softmax.backward(decState[i]->h, targetDist, data->tgt[i], decState[i]->delh, grad.softmaxGrad);
    // <??> PART S: END
  }

  decState[data->tgt.size()-1]->delc = this->zeros; // <??> can be faster by using setZeros?

  for (int i = data->tgt.size()-1; i >= 1; --i){
    decState[i-1]->delc = this->zeros; // <??> can be faster by using setZeros?
    this->dec.backward(decState[i-1], decState[i], grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]));

    // <??> PART A: THIS PART MAY BE THE BOTTLENECK, create new member
    // <??> PART A: BEGIN
    if (grad.targetEmbed.count(data->tgt[i-1])){
      grad.targetEmbed.at(data->tgt[i-1]) += decState[i]->delx;
    }
    else {
      grad.targetEmbed[data->tgt[i-1]] = decState[i]->delx;
    }
    // <??> PART A: END
  }
  
  encState[data->src.size()]->delc = decState[0]->delc;
  encState[data->src.size()]->delh = decState[0]->delh;

  for (int i = data->src.size(); i >= 1; --i){
    encState[i-1]->delh = this->zeros; // <??> can be faster by using setZeros?
    encState[i-1]->delc = this->zeros; // <??> can be faster bu using setZeros?
    this->enc.backward(encState[i-1], encState[i], grad.lstmSrcGrad, this->sourceEmbed.col(data->src[i-1]));

    // <??> PART B: THIS PART MAY BE THE BOTTLENECK
    // <??> PART B: BEGIN
    if (grad.sourceEmbed.count(data->src[i-1])){
      grad.sourceEmbed.at(data->src[i-1]) += encState[i]->delx;
    }
    else {
      grad.sourceEmbed[data->src[i-1]] = encState[i]->delx;
    }
    // <??> PART B: END
  }
}

void EncDec::trainOpenMP(const Real learningRate, const int miniBatchSize, const int numThreads){
  static std::vector<EncDec::ThreadArg*> args;
  static std::vector<std::pair<int, int> > miniBatch;
  static EncDec::Grad grad;
  Real lossTrain = 0.0, perpDev = 0.0, denom = 0.0;
  Real gradNorm, lr = learningRate;
  const Real clipThreshold = 3.0;
  struct timeval start, end;

	// 
	struct timeval k_start, k_end;
	struct timeval k_start_1, k_end_1;
	struct timeval k_start_2, k_end_2;
	Real k_time = 0.0;
	Real k_time_1 = 0.0;
	Real k_time_2 = 0.0;

  if (args.empty()){
	std::cout << "hell0'" << std::endl;
	  for (int i = 0; i < numThreads; ++i){
      args.push_back(new EncDec::ThreadArg(*this));

      for (int j = 0; j < 200; ++j){    //<??> why j < 200 here
	args[i]->encState.push_back(new LSTM::State);
	args[i]->decState.push_back(new LSTM::State);
      }
    }

    for (int i = 0, step = this->trainData.size()/miniBatchSize; i < step; ++i){
      miniBatch.push_back(std::pair<int, int>(i*miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*miniBatchSize-1)));
    }
	
    grad.lstmSrcGrad = LSTM::Grad(this->enc);
    grad.lstmTgtGrad = LSTM::Grad(this->dec);
    grad.softmaxGrad = SoftMax::Grad(this->softmax);

    //std::sort(this->trainData.begin(), this->trainData.end(), sort_pred());
  }

	 std::cout << "number of miniBatch = " << miniBatch.size() << std::endl;
	 std::cout << "first pair is " << miniBatch[0].first << " and " << miniBatch[0].second << std::endl;
  

	
	//this->rnd.shuffle(miniBatch);
  this->rnd.shuffle(this->trainData); // <??> this part can be faster??

	// add time recorder here
	struct timeval time_rec_start[numThreads];
	struct timeval time_rec_end[numThreads];
	__time_t sec_start[numThreads][trainData.size()];
	__time_t sec_end[numThreads][trainData.size()];
	__suseconds_t usec_start[numThreads][trainData.size()];
	__suseconds_t usec_end[numThreads][trainData.size()];
	int iter_counter[numThreads];

	for (int ii = 0; ii < numThreads; ii ++)
	{
		iter_counter[ii] = -1;
	}

  gettimeofday(&start, 0);

  int count = 0;
  k_time = 0.0;
  k_time_1 = 0.0;
  k_time_2 = 0.0;


  for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it){
    //  std::cout << "\r" << "Progress: " << ++count << "/" << miniBatch.size() << " mini batches" << std::flush;
	
	  gettimeofday(&k_start, NULL); 

#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(args)
    for (int i = it->first; i <= it->second; ++i){
      int id = omp_get_thread_num();
      Real loss;
	  iter_counter[id] ++;
	  gettimeofday(&(time_rec_start[id]), NULL);
      this->train(this->trainData[i], args[id]->encState, args[id]->decState, args[id]->grad, loss);
	  gettimeofday(&(time_rec_end[id]), NULL);
	  sec_start[id][iter_counter[id]] = time_rec_start[id].tv_sec;
	  usec_start[id][iter_counter[id]] = time_rec_start[id].tv_usec;
	  sec_end[id][iter_counter[id]] = time_rec_end[id].tv_sec;
	  usec_end[id][iter_counter[id]] = time_rec_end[id].tv_usec;
      args[id]->loss += loss;
    }

	gettimeofday(&k_end, NULL);
	k_time += ((k_end.tv_sec-k_start.tv_sec)*1000000+(k_end.tv_usec-k_start.tv_usec))/1000.0;
    gettimeofday(&k_start_1, NULL);
	for (int id = 0; id < numThreads; ++id){
      grad += args[id]->grad;
      args[id]->grad.init();
      lossTrain += args[id]->loss;
      args[id]->loss = 0.0;
    }

    gradNorm = sqrt(grad.norm())/miniBatchSize;
    Utils::infNan(gradNorm);
    lr = (gradNorm > clipThreshold ? clipThreshold*learningRate/gradNorm : learningRate);
    lr /= miniBatchSize;

    this->enc.sgd(grad.lstmSrcGrad, lr);
    this->dec.sgd(grad.lstmTgtGrad, lr);

    this->softmax.sgd(grad.softmaxGrad, lr);

    for (auto it = grad.sourceEmbed.begin(); it != grad.sourceEmbed.end(); ++it){
      this->sourceEmbed.col(it->first) -= lr*it->second;
    }
    for (auto it = grad.targetEmbed.begin(); it != grad.targetEmbed.end(); ++it){
      this->targetEmbed.col(it->first) -= lr*it->second;
    }

    grad.init();
	gettimeofday(&k_end_1, NULL);
	k_time_1 += ((k_end_1.tv_sec-k_start_1.tv_sec)*1000000+(k_end_1.tv_usec-k_start_1.tv_usec))/1000.0;
  } // end of for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it)

  std::cout << std::endl;
  gettimeofday(&end, 0);
  //std::cout << "Training time for this epoch: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
  std::cout << "Training time for this epoch: " << ((end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec))/1000.0 << " ms." << std::endl;
  std::cout << "Time for parallel part: " << k_time << " ms." << std::endl;
  std::cout << "Time for seq part: " << k_time_1 << " ms." << std::endl;

  std::cout << "Training Loss (/sentence):    " << lossTrain/this->trainData.size() << std::endl;
	int sum_iter_counter = 0;
	for (int ii = 0; ii < numThreads; ii ++)
	{
		std::cout << "thread id  = " << ii << ", count = " << iter_counter[ii] << std::endl;
		sum_iter_counter += iter_counter[ii];
	}
	std::cout << "sum = " << sum_iter_counter << std::endl;
	// here for record into file
	std::ofstream fout_start("time_rec_start.log");
	std::ofstream fout_end("time_rec_end.log");
	
	for (int ii = 0; ii < numThreads; ii ++)
	{
		for (int jj = 0; jj <= iter_counter[ii]; jj ++)
		{
			double start_time = (double)((sec_start[ii][jj] - start.tv_sec) * 1000000.0 + (usec_start[ii][jj]-start.tv_usec))/1000.0;
			double end_time = (double)((sec_end[ii][jj] - start.tv_sec) * 1000000.0 + (usec_end[ii][jj]-start.tv_usec))/1000.0;
			fout_start << ii << " " << start_time << std::endl;
			fout_end << ii << " " << end_time << std::endl;
		}
	}
	fout_start.close();
	fout_end.close();

  gettimeofday(&start, 0);

#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(perpDev, denom)
  for (int i = 0; i < (int)this->devData.size(); ++i){
    Real perp = this->calcLoss(this->devData[i], this->encStateDev[i], this->decStateDev[i]);
    
    for (auto it = this->encStateDev[i].begin(); it != this->encStateDev[i].end(); ++it){
      (*it)->clear();
    }
    for (auto it = this->decStateDev[i].begin(); it != this->decStateDev[i].end(); ++it){
      (*it)->clear();
    }

#pragma omp critical
    {
      perpDev += perp;
      denom += this->devData[i]->tgt.size();
    }
  }

  gettimeofday(&end, 0);
  //std::cout << "Evaluation time for this epoch: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
  std::cout << "Evaluation time for this epoch: " << ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000.0 << " ms." << std::endl;
  std::cout << "Development loss (/sentence): " << perpDev/this->devData.size() << std::endl;
  std::cout << "Development perplexity (global): " << exp(perpDev/denom) << std::endl;
}

void EncDec::demo(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev){
  const int threSource = 1;
  const int threTarget = 1;
  Vocabulary sourceVoc(srcTrain, threSource);
  Vocabulary targetVoc(tgtTrain, threTarget);
  std::vector<EncDec::Data*> trainData, devData;
  std::ifstream ifsSrcTrain(srcTrain.c_str());
  std::ifstream ifsTgtTrain(tgtTrain.c_str());
  std::ifstream ifsSrcDev(srcDev.c_str());
  std::ifstream ifsTgtDev(tgtDev.c_str());
  std::vector<std::string> tokens;
  int numLine = 0;

  //training data
  for (std::string line; std::getline(ifsSrcTrain, line); ){
    trainData.push_back(new EncDec::Data);
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      trainData.back()->src.push_back(sourceVoc.tokenIndex.count(*it) ? sourceVoc.tokenIndex.at(*it) : sourceVoc.unkIndex);
    }

    std::reverse(trainData.back()->src.begin(), trainData.back()->src.end());
    trainData.back()->src.push_back(sourceVoc.eosIndex);
  }

  for (std::string line; std::getline(ifsTgtTrain, line); ){
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      trainData[numLine]->tgt.push_back(targetVoc.tokenIndex.count(*it) ? targetVoc.tokenIndex.at(*it) : targetVoc.unkIndex);
    }

    trainData[numLine]->tgt.push_back(targetVoc.eosIndex);
    ++numLine;
  }

  //development data
  numLine = 0;

  for (std::string line; std::getline(ifsSrcDev, line); ){
    devData.push_back(new EncDec::Data);
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      devData.back()->src.push_back(sourceVoc.tokenIndex.count(*it) ? sourceVoc.tokenIndex.at(*it) : sourceVoc.unkIndex);
    }

    std::reverse(devData.back()->src.begin(), devData.back()->src.end());
    devData.back()->src.push_back(sourceVoc.eosIndex);
  }

  for (std::string line; std::getline(ifsTgtDev, line); ){
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      devData[numLine]->tgt.push_back(targetVoc.tokenIndex.count(*it) ? targetVoc.tokenIndex.at(*it) : targetVoc.unkIndex);
    }

    devData[numLine]->tgt.push_back(targetVoc.eosIndex);
    ++numLine;
  }

  Real learningRate = 0.5;
  const int inputDim = 50;
  const int hiddenDim = 50;
  const int miniBatchSize = 128; // <!!> modify this parameter to test
  const int numThread = 8;
  EncDec encdec(sourceVoc, targetVoc, trainData, devData, inputDim, hiddenDim);
  auto test = trainData[0]->src;

  std::cout << "# of training data:    " << trainData.size() << std::endl;
  std::cout << "# of development data: " << devData.size() << std::endl;
  std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
  std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;
  
  int break_point;

	// std::cin >> break_point;

  for (int i = 0; i < 3; ++i){
    if (i+1 >= 6){
      //learningRate *= 0.5;
    }

    std::cout << "\nEpoch " << i+1 << std::endl;
    encdec.trainOpenMP(learningRate, miniBatchSize, numThread);
    // std::cout << "### Greedy ###" << std::endl;
    // encdec.translate(test, 1, 100, 1);
    // std::cout << "### Beam search ###" << std::endl;
    // encdec.translate(test, 20, 100, 5);

    std::ostringstream oss;
    oss << "model." << i+1 << "itr.bin";
	// 
	// std::cin >> break_point;
    //encdec.save(oss.str());
  }
  

	return;

  encdec.load("model.1itr.bin");

  struct timeval start, end;
  
  //translation
  std::vector<std::vector<int> > output(encdec.devData.size());
  gettimeofday(&start, 0);
#pragma omp parallel for num_threads(numThread) schedule(dynamic) shared(output, encdec)
  for (int i = 0; i < (int)encdec.devData.size(); ++i){
    encdec.translate(output[i], encdec.devData[i]->src, 20, 100);
  }

  gettimeofday(&end, 0);
  std::cout << "Translation time: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
 
  std::ofstream ofs("translation.txt");
  
  for (auto it = output.begin(); it != output.end(); ++it){
    for (auto it2 = it->begin(); it2 != it->end(); ++it2){
      ofs << encdec.targetVoc.tokenList[(*it2)]->str << " ";
    }
    ofs << std::endl;
  }
} // end of demo

// demo_qiao() written by qiaoyc, for convenient test on different arguments
void EncDec::demo_qiao(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads)
{
  const int threSource = 1;
  const int threTarget = 1;
  Vocabulary sourceVoc(srcTrain, threSource);
  Vocabulary targetVoc(tgtTrain, threTarget);
  std::vector<EncDec::Data*> trainData, devData;
  std::ifstream ifsSrcTrain(srcTrain.c_str());
  std::ifstream ifsTgtTrain(tgtTrain.c_str());
  std::ifstream ifsSrcDev(srcDev.c_str());
  std::ifstream ifsTgtDev(tgtDev.c_str());
  std::vector<std::string> tokens;
  int numLine = 0;

  //training data
  for (std::string line; std::getline(ifsSrcTrain, line); ){
    trainData.push_back(new EncDec::Data);
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      trainData.back()->src.push_back(sourceVoc.tokenIndex.count(*it) ? sourceVoc.tokenIndex.at(*it) : sourceVoc.unkIndex);
    }

    std::reverse(trainData.back()->src.begin(), trainData.back()->src.end());
    trainData.back()->src.push_back(sourceVoc.eosIndex);
  }

  for (std::string line; std::getline(ifsTgtTrain, line); ){
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      trainData[numLine]->tgt.push_back(targetVoc.tokenIndex.count(*it) ? targetVoc.tokenIndex.at(*it) : targetVoc.unkIndex);
    }

    trainData[numLine]->tgt.push_back(targetVoc.eosIndex);
    ++numLine;
  }

  //development data
  numLine = 0;

  for (std::string line; std::getline(ifsSrcDev, line); ){
    devData.push_back(new EncDec::Data);
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      devData.back()->src.push_back(sourceVoc.tokenIndex.count(*it) ? sourceVoc.tokenIndex.at(*it) : sourceVoc.unkIndex);
    }

    std::reverse(devData.back()->src.begin(), devData.back()->src.end());
    devData.back()->src.push_back(sourceVoc.eosIndex);
  }

  for (std::string line; std::getline(ifsTgtDev, line); ){
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      devData[numLine]->tgt.push_back(targetVoc.tokenIndex.count(*it) ? targetVoc.tokenIndex.at(*it) : targetVoc.unkIndex);
    }

    devData[numLine]->tgt.push_back(targetVoc.eosIndex);
    ++numLine;
  }

  Real learningRate = argsLearningRate;
  const int inputDim = argsInputDim;
  const int hiddenDim = argsHiddenDim;
  const int miniBatchSize = argsMiniBatchSize; // <!!> modify this parameter to test
  const int numThread = argsNumThreads;
  EncDec encdec(sourceVoc, targetVoc, trainData, devData, inputDim, hiddenDim);
  auto test = trainData[0]->src;

  std::cout << "# of training data:    " << trainData.size() << std::endl;
  std::cout << "# of development data: " << devData.size() << std::endl;
  std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
  std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;
  
  int break_point;

	// std::cin >> break_point;

  for (int i = 0; i < 3; ++i){
    if (i+1 >= 6){
      //learningRate *= 0.5;
    }

    std::cout << "\nEpoch " << i+1 << std::endl;
    encdec.trainOpenMP(learningRate, miniBatchSize, numThread);
    // std::cout << "### Greedy ###" << std::endl;
    // encdec.translate(test, 1, 100, 1);
    // std::cout << "### Beam search ###" << std::endl;
    // encdec.translate(test, 20, 100, 5);

    std::ostringstream oss;
    oss << "model." << i+1 << "itr.bin";
	// 
	// std::cin >> break_point;
    //encdec.save(oss.str());
  }
  

	return;

  encdec.load("model.1itr.bin");

  struct timeval start, end;
  
  //translation
  std::vector<std::vector<int> > output(encdec.devData.size());
  gettimeofday(&start, 0);
#pragma omp parallel for num_threads(numThread) schedule(dynamic) shared(output, encdec)
  for (int i = 0; i < (int)encdec.devData.size(); ++i){
    encdec.translate(output[i], encdec.devData[i]->src, 20, 100);
  }

  gettimeofday(&end, 0);
  std::cout << "Translation time: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
 
  std::ofstream ofs("translation.txt");
  
  for (auto it = output.begin(); it != output.end(); ++it){
    for (auto it2 = it->begin(); it2 != it->end(); ++it2){
      ofs << encdec.targetVoc.tokenList[(*it2)]->str << " ";
    }
    ofs << std::endl;
  }
} // end of demo_qiao()
void EncDec::save(const std::string& fileName){
  std::ofstream ofs(fileName.c_str(), std::ios::out|std::ios::binary);

  assert(ofs);

  this->enc.save(ofs);
  this->dec.save(ofs);
  Utils::save(ofs, sourceEmbed);
  Utils::save(ofs, targetEmbed);
}

void EncDec::load(const std::string& fileName){
  std::ifstream ifs(fileName.c_str(), std::ios::in|std::ios::binary);

  assert(ifs);

  this->enc.load(ifs);
  this->dec.load(ifs);
  Utils::load(ifs, sourceEmbed);
  Utils::load(ifs, targetEmbed);
}
