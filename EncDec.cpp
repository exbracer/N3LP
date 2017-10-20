#include "EncDec.hpp"
#include "Utils.hpp"
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <omp.h>
#include "ActFunc.hpp"

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
	//std::cout << "i = " << i << std::endl;
		this->enc.forward(this->sourceEmbed.col(src[i]), encState[i], encState[i+1]);
	}
}


void EncDec::encode_qiao(const std::vector<int>& src, std::vector<LSTM::State*>& encState){
	encState[0]->h.setZero();
	encState[0]->c.setZero();

	for (int i = 0; i < (int)src.size(); ++i){
	//std::cout << "i = " << i << std::endl;
		this->enc.forward(this->sourceEmbed.col(src[i]), encState[i], encState[i+1]);
	}
}
void EncDec::encode_mf_v1(const std::vector<int>& src, std::vector<LSTM::State*>& encState, MemoryFootprint* mf)
{
	encState[0]->h = this->zeros;
	encState[0]->c = this->zeros;

	for (int i = 0; i < (int)src.size(); ++i)
	{
		this->enc.forward_mf_v1(this->sourceEmbed.col(src[i]), encState[i], encState[i+1], mf);
	}
} // function created by qiaoyc for experiments for memory footprint */

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
  	// std::cout << "after encode" << std::endl;
  	for (int i = 0; i < (int)data->tgt.size(); ++i)
  	{
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

void EncDec::train_qiao_1(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss){
	VecD targetDist; // <??> created in stack of this thread

	loss = 0.0;
	this->encode_qiao(data->src, encState);
	// std::cout << "after encode" << std::endl;
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

  	//decState[data->tgt.size()-1]->delc = this->zeros; // <??> can be faster by using setZeros?
	decState[data->tgt.size()-1]->delc.setZero();


	for (int i = data->tgt.size()-1; i >= 1; --i){
    	// decState[i-1]->delc = this->zeros; // <??> can be faster by using setZeros?
		decState[i-1]->delc.setZero();
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
    // encState[i-1]->delh = this->zeros; // <??> can be faster by using setZeros?
    // encState[i-1]->delc = this->zeros; // <??> can be faster bu using setZeros?

		encState[i-1]->delh.setZero();
		encState[i-1]->delc.setZero();

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

void EncDec::train_qiao_2(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<double>& timeRecorder)
{	
	// this function is for recording the time of each parts of train
	// to check the scalability for each parts
	struct timeval start, end;
	struct timeval start_1, end_1;

	VecD targetDist; // <??> created in stack of this thread

	loss = 0.0;

	// <!!> TIME PART I: this part needs a time recorder
	gettimeofday(&start, NULL);
	this->encode_qiao(data->src, encState);
	gettimeofday(&end, NULL);
	timeRecorder[0] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
	// <!!> TIME PART I: end
	// std::cout << "time part I" << std::endl;	
	// <!!> TIME PART II: this part needs a time recorder
	gettimeofday(&start, NULL);
	for (int i = 0; i < (int)data->tgt.size(); ++i)
	{
		// <!!> TIME PART II-I: this part needs a time recorder
		gettimeofday(&start_1, NULL);		
		if (i == 0){
			decState[0]->h = encState[data->src.size()]->h;
			decState[0]->c = encState[data->src.size()]->c;
		}
		else {
			this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), decState[i-1], decState[i]);
		}
		gettimeofday(&end_1, NULL);
		timeRecorder[1] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
		// <!!> TIME PART II-I: end

		// <!!> TIME PART II-II: this part needs a time recorder
		gettimeofday(&start_1, NULL);
		// <??> PART S: THIS PART MAY BE THE BOTTLENEC
		// <??> PART S: BEGIN
		this->softmax.calcDist(decState[i]->h, targetDist); // <??> are each thead competing for softmax
		loss += this->softmax.calcLoss(targetDist, data->tgt[i]); // <??> same question as above
		this->softmax.backward(decState[i]->h, targetDist, data->tgt[i], decState[i]->delh, grad.softmaxGrad);
		// <??> PART S: END
		gettimeofday(&end_1, NULL);
		timeRecorder[2] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
		// <!!> TIME PART II-II: end
	}
	gettimeofday(&end, NULL);
	timeRecorder[3] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
	// <!!> TIME PART II: end
	// std::cout << "time part II" << std::endl;
  	//decState[data->tgt.size()-1]->delc = this->zeros; // <??> can be faster by using setZeros?
	decState[data->tgt.size()-1]->delc.setZero();

	// <!!> TIME PART III: this part needs a time recorder
	gettimeofday(&start, NULL);
	for (int i = data->tgt.size()-1; i >= 1; --i)
	{
		// <!!> TIME PART III-I: this part needs a time recorder
    	gettimeofday(&start_1, NULL);
		// decState[i-1]->delc = this->zeros; // <??> can be faster by using setZeros?
		decState[i-1]->delc.setZero();
		this->dec.backward(decState[i-1], decState[i], grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]));
		gettimeofday(&end_1, NULL);
		timeRecorder[4] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
		// <!!> TIME PART III-I: end

		// <!!> TIME PART III-II: this part needs a time recorder
    	gettimeofday(&start_1, NULL);
		// <??> PART A: THIS PART MAY BE THE BOTTLENECK, create new member
    	// <??> PART A: BEGIN
		if (grad.targetEmbed.count(data->tgt[i-1])){
			grad.targetEmbed.at(data->tgt[i-1]) += decState[i]->delx;
		}
		else {
			grad.targetEmbed[data->tgt[i-1]] = decState[i]->delx;
		}
		// <??> PART A: END
		gettimeofday(&end_1, NULL);
		timeRecorder[5] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
		// <!!> TIME PART III-II: end
	}
	gettimeofday(&end, NULL);
	timeRecorder[6] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
	// <!!> TIME PART III: end
	// std::cout << "time part III" << std::endl;
	encState[data->src.size()]->delc = decState[0]->delc;
	encState[data->src.size()]->delh = decState[0]->delh;

	// <!!> TIME PART IV: this part needs a time recorder
	gettimeofday(&start, NULL);
	for (int i = data->src.size(); i >= 1; --i)
	{
		// <!!> TIME PART IV-I: this part needs a time recorder
		gettimeofday(&start_1, NULL);
		encState[i-1]->delh = this->zeros; // <??> can be faster by using setZeros?
		encState[i-1]->delc = this->zeros; // <??> can be faster bu using setZeros?
		//encState[i-1]->delh.setZero();
		//encState[i-1]->delc.setZero();

		this->enc.backward(encState[i-1], encState[i], grad.lstmSrcGrad, this->sourceEmbed.col(data->src[i-1]));
		gettimeofday(&end_1, NULL);
		timeRecorder[7] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
		// <!!> TIME PART IV-I: end
		// std::cout << "time part IV-I" << std::endl;
		// <!!> TIME PART IV-II: this part needs a time recorder
		gettimeofday(&start_1, NULL);
		// <??> PART B: THIS PART MAY BE THE BOTTLENECK
		// <??> PART B: BEGIN
		if (grad.sourceEmbed.count(data->src[i-1])){
			grad.sourceEmbed.at(data->src[i-1]) += encState[i]->delx;
		}
		else {
			grad.sourceEmbed[data->src[i-1]] = encState[i]->delx;
		}
		// <??> PART B: END
		gettimeofday(&end_1, NULL);
		timeRecorder[8] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
		// <!!> PART IV-II: end
		// std::cout << "time part IV-II" << std::endl;
	}
	gettimeofday(&end, NULL);
	timeRecorder[9] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
	// <!!> TIME PART IV: end
	// std::cout << "time part IV" << std::endl;
} // end of train_qiao_2

void EncDec::train_qiao_3(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<double>& timeRecorder)
{	
	// this function is for recording the time of each parts of train
	// to check the scalability for each parts
	struct timeval start, end;
	struct timeval start_1, end_1;
	struct timeval start_2, end_2;

	VecD targetDist; // <??> created in stack of this thread

	loss = 0.0;

	// <!!> TIME PART I: this part needs a time recorder
	gettimeofday(&start, NULL);
	this->encode(data->src, encState);
	gettimeofday(&end, NULL);
	timeRecorder[0] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
	// <!!> TIME PART I: end
	// std::cout << "time part I" << std::endl;	
	// <!!> TIME PART II: this part needs a time recorder
	gettimeofday(&start, NULL);
	for (int i = 0; i < (int)data->tgt.size(); ++i)
	{
		// <!!> TIME PART II-I: this part needs a time recorder
		gettimeofday(&start_1, NULL);		
		if (i == 0){
			decState[0]->h = encState[data->src.size()]->h;
			decState[0]->c = encState[data->src.size()]->c;
		}
		else {
			this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), decState[i-1], decState[i]);
		}
		gettimeofday(&end_1, NULL);
		timeRecorder[1] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
		// <!!> TIME PART II-I: end

		// <!!> TIME PART II-II: this part needs a time recorder
		gettimeofday(&start_1, NULL);
		// <??> PART S: THIS PART MAY BE THE BOTTLENEC
		// <??> PART S: BEGIN
		// <!!> TIME PART II-II-I: this part neeeds a time recorder
		gettimeofday(&start_2, NULL);
		this->softmax.calcDist(decState[i]->h, targetDist); // <??> are each thead competing for softmax
		gettimeofday(&end_2,NULL);
		timeRecorder[10] += (double)((end_2.tv_sec-start_2.tv_sec)*1000000+(end_2.tv_usec-start_2.tv_usec))/1000;
		// <!!> TIME PART II-II-I: end

		// <!!> TIME PART II-II-II: this part needs a time recorder
		gettimeofday(&start_2, NULL);
		loss += this->softmax.calcLoss(targetDist, data->tgt[i]); // <??> same question as above
		gettimeofday(&end_2,NULL);
		timeRecorder[11] += (double)((end_2.tv_sec-start_2.tv_sec)*1000000+(end_2.tv_usec-start_2.tv_usec))/1000;
		// <!!> TIME PART II-II-II: end
		
		// <!!> TIME PART II-II-III: this part needs a time recorder
		gettimeofday(&start_2, NULL);
		this->softmax.backward(decState[i]->h, targetDist, data->tgt[i], decState[i]->delh, grad.softmaxGrad);
		gettimeofday(&end_2, NULL);
		timeRecorder[12] += (double)((end_2.tv_sec-start_2.tv_sec)*1000000+(end_2.tv_usec-start_2.tv_usec))/1000;
		// <!!> TIME PART II-II-III: end
		// <??> PART S: END
		gettimeofday(&end_1, NULL);
		timeRecorder[2] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
		// <!!> TIME PART II-II: end
	}
	//std::cout << "loss = " << loss << std::endl;
	//std::cout << "softmax.grad = " << grad.softmaxGrad.norm() << std::endl;
	int xx_size = (int)data->tgt.size()-1;
	//std::cout << "during softmax delh at " << xx_size  << " is " << decState[xx_size]->delh.squaredNorm() << std::endl;
	//std::cout << "during softmax delh at " << 5 << " is " << decState[5]->delh.squaredNorm() << std::endl;
	gettimeofday(&end, NULL);
	timeRecorder[3] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
	// <!!> TIME PART II: end
	// std::cout << "time part II" << std::endl;
  	decState[data->tgt.size()-1]->delc = this->zeros; // <??> can be faster by using setZeros?
	//decState[data->tgt.size()-1]->delc.setZero();
	/*
	std::cout << "///after set zero ////" << std::endl;
	std::cout << decState[data->tgt.size()-1]->delc.array() << std::endl;
	std::cout << "///////////////////////" << std::endl;
	std::cout << "xx cTanh = " << decState[1]->cTanh.squaredNorm() << std::endl;
	std::cout << "xx delh = " << decState[xx_size-1]->delh.squaredNorm() << std::endl;
	std::cout << "xx o = " << decState[1]->o.squaredNorm() << std::endl;
	
	std::cout << "**** before ***************" << std::endl;
	std::cout << "xx cTanh = " << decState[1]->cTanh.squaredNorm() << std::endl;
	std::cout << "xx delh = " << decState[1]->delh.squaredNorm() << std::endl;
	std::cout << "xx o = " << decState[1]->o.squaredNorm() << std::endl;
	std::cout << "+++++++++++++++++++++++++++" << std::endl;
	std::cout << "curr cTanh = " << decState[xx_size]->cTanh.squaredNorm() << std::endl;
	std::cout << "curr delh = " << decState[xx_size]->delh.squaredNorm() << std::endl;
	std::cout << "curr o = " << decState[xx_size]->o.squaredNorm() << std::endl;
	std::cout << "curr delc = " << decState[xx_size]->delc.squaredNorm() << std::endl;
	std::cout << "curr f = " << decState[xx_size]->f.squaredNorm() << std::endl;
	std::cout << "prev delc = " << decState[xx_size-1]->delc.squaredNorm() << std::endl;
	std::cout << "curr i = " << decState[xx_size]->i.squaredNorm() << std::endl;
	std::cout << "curr u = " << decState[xx_size]->u.squaredNorm() << std::endl;
	std::cout << "curr c = " << decState[xx_size]->c.squaredNorm() << std::endl;
	std::cout << "curr delx = " << decState[xx_size]->delx.squaredNorm() << std::endl;
	std::cout << "prev delh = " << decState[xx_size-1]->delh.squaredNorm() << std::endl;
	std::cout << "****************************" << std::endl;
	*/
	// <!!> TIME PART III: this part needs a time recorder
	gettimeofday(&start, NULL);
	for (int i = data->tgt.size()-1; i >= 1; --i)
	{
		// <!!> TIME PART III-I: this part needs a time recorder
    	gettimeofday(&start_1, NULL);
		decState[i-1]->delc = this->zeros; // <??> can be faster by using setZeros?
		//decState[i-1]->delc.setZero();
		this->dec.backward(decState[i-1], decState[i], grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]));
		gettimeofday(&end_1, NULL);
		timeRecorder[4] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
		// <!!> TIME PART III-I: end

		// <!!> TIME PART III-II: this part needs a time recorder
    	gettimeofday(&start_1, NULL);
		// <??> PART A: THIS PART MAY BE THE BOTTLENECK, create new member
    	// <??> PART A: BEGIN
		if (grad.targetEmbed.count(data->tgt[i-1])){
			grad.targetEmbed.at(data->tgt[i-1]) += decState[i]->delx;
		}
		else {
			grad.targetEmbed[data->tgt[i-1]] = decState[i]->delx;
		}
		// <??> PART A: END
		gettimeofday(&end_1, NULL);
		timeRecorder[5] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
		// <!!> TIME PART III-II: end
	}
	/*
	VecD xx_delo =  ActFunc::logisticPrime(decState[1]->cTanh).array() * decState[1]->delh.array() * decState[1]->o.array();
	std::cout << "xx_delo[1] = " << xx_delo.squaredNorm() << std::endl;
	std::cout << "hello cTanh = " << decState[1]->cTanh.squaredNorm() << std::endl;
	std::cout << "hello delh = " << decState[xx_size-1]->delh.squaredNorm() << std::endl;
	std::cout << "hello o = " << decState[1]->o.squaredNorm() << std::endl;
	
	std::cout << "******after *******************" << std::endl;
	std::cout << "xx cTanh = " << decState[1]->cTanh.squaredNorm() << std::endl;
	std::cout << "xx delh = " << decState[1]->delh.squaredNorm() << std::endl;
	std::cout << "xx o = " << decState[1]->o.squaredNorm() << std::endl;
	std::cout << "+++++++++++++++++++++++++++++++" << std::endl;
	std::cout << "curr cTanh = " << decState[xx_size]->cTanh.squaredNorm() << std::endl;
	std::cout << "curr delh = " << decState[xx_size]->delh.squaredNorm() << std::endl;
	std::cout << "curr o = " << decState[xx_size]->o.squaredNorm() << std::endl;
	std::cout << "curr delc = " << decState[xx_size]->delc.squaredNorm() << std::endl;
	std::cout << "curr f = " << decState[xx_size]->f.squaredNorm() << std::endl;
	std::cout << "prev delc = " << decState[xx_size-1]->delc.squaredNorm() << std::endl;
	std::cout << "curr i = " << decState[xx_size]->i.squaredNorm() << std::endl;
	std::cout << "curr u = " << decState[xx_size]->u.squaredNorm() << std::endl;
	std::cout << "curr c = " << decState[xx_size]->c.squaredNorm() << std::endl;
	std::cout << "curr delx = " << decState[xx_size]->delx.squaredNorm() << std::endl;
	std::cout << "prev delh = " << decState[xx_size-1]->delh.squaredNorm() << std::endl;
	std::cout << "*******************************" << std::endl;
	*/
	
	gettimeofday(&end, NULL);
	timeRecorder[6] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
	//std::cout << "lstmTgtGrad = " << grad.lstmTgtGrad.norm() << std::endl;
	
	//int input_xx;
	//std::cin >> input_xx;
	
	// <!!> TIME PART III: end
	// std::cout << "time part III" << std::endl;
	encState[data->src.size()]->delc = decState[0]->delc;
	encState[data->src.size()]->delh = decState[0]->delh;

	// <!!> TIME PART IV: this part needs a time recorder
	gettimeofday(&start, NULL);
	for (int i = data->src.size(); i >= 1; --i)
	{
		// <!!> TIME PART IV-I: this part needs a time recorder
		gettimeofday(&start_1, NULL);
		encState[i-1]->delh = this->zeros; // <??> can be faster by using setZeros?
		encState[i-1]->delc = this->zeros; // <??> can be faster bu using setZeros?
		//encState[i-1]->delh.setZero();
		//encState[i-1]->delc.setZero();

		this->enc.backward(encState[i-1], encState[i], grad.lstmSrcGrad, this->sourceEmbed.col(data->src[i-1]));
		gettimeofday(&end_1, NULL);
		timeRecorder[7] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
		// <!!> TIME PART IV-I: end
		// std::cout << "time part IV-I" << std::endl;
		// <!!> TIME PART IV-II: this part needs a time recorder
		gettimeofday(&start_1, NULL);
		// <??> PART B: THIS PART MAY BE THE BOTTLENECK
		// <??> PART B: BEGIN
		if (grad.sourceEmbed.count(data->src[i-1])){
			grad.sourceEmbed.at(data->src[i-1]) += encState[i]->delx;
		}
		else {
			grad.sourceEmbed[data->src[i-1]] = encState[i]->delx;
		}
		// <??> PART B: END
		gettimeofday(&end_1, NULL);
		timeRecorder[8] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
		// <!!> PART IV-II: end
		// std::cout << "time part IV-II" << std::endl;
	}
	gettimeofday(&end, NULL);
	timeRecorder[9] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
	// <!!> TIME PART IV: end
	// std::cout << "time part IV" << std::endl;
	//std::cout << "lstmSrcGrad = " << grad.lstmSrcGrad.norm() << std::endl;
} // end of train_qiao_3

void EncDec::train_qiao_4(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<double>& timeRecorder, std::vector<VecD>& target_dist, std::vector<VecD>& delosBuffer, std::vector<VecD>& delisBuffer, std::vector<VecD>& delusBuffer, std::vector<VecD>& delfsBuffer)
{	
	// this function is for recording the time of each parts of train
	// to check the scalability for each parts
	struct timeval start, end;
	struct timeval start_1, end_1;
	struct timeval start_2, end_2;

	VecD targetDist; // <??> created in stack of this thread

	loss = 0.0;

	// <!!> TIME PART I: this part needs a time recorder
	// std::cout << "part 1" << std::endl;
	gettimeofday(&start, NULL);
	this->encode(data->src, encState);
	gettimeofday(&end, NULL);
	timeRecorder[0] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
	// <!!> TIME PART I: end
	// std::cout << "time part I" << std::endl;	
    
    // <!!> TIME PART II: this part needs a time recorder
	// std::cout << "part 2" << std::endl;
	gettimeofday(&start, NULL);

	// <!!> TIME PART II-1: this part needs a time recorder
	// std::cout << "part 2-1" << std::endl;
	gettimeofday(&start_1, NULL);
    // when i = 0, pass h and c from enc to dec
    decState[0]->h = encState[data->src.size()]->h;
    decState[0]->c = encState[data->src.size()]->c;
    // when i > 0, do forward function
	for (int i = 1; i < (int)data->tgt.size(); ++ i)
    {
        this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), decState[i-1], decState[i]);   
    }
    gettimeofday(&end_1, NULL);
    timeRecorder[1] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
    // <!!> TIME PART II-I: end
    
    // <!!> TIME PART II-II: this part needs a time recorder
	// std::cout << "part 2-2" << std::endl;
	gettimeofday(&start_1, NULL);
    // <!!> TIME PART II-II-I: this time needs a time recorder
	// std::cout << "part 2-2-1" << std::endl;
    gettimeofday(&start_2, NULL);
	// std::cout << "tgt.size = " << (int)data->tgt.size() << std::endl;
    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        this->softmax.calcDist(decState[i]->h, target_dist[i]);
    }
	// std::cout << "xxx" << std::endl;
    gettimeofday(&end_2, NULL);
    timeRecorder[10] += (double)((end_2.tv_sec-start_2.tv_sec)*1000000+(end_2.tv_usec-start_2.tv_usec))/1000;
    // <!!> TIME PART II-II-I: end

    // <!!> TIME PART II-II-II: this time needs a time recorder
	// std::cout << "part 2-2-2" << std::endl;
	gettimeofday(&start_2, NULL);
    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        loss += this->softmax.calcLoss(target_dist[i], data->tgt[i]);
		
    }
	//std::cout << "loss = " << loss << std::endl;
    gettimeofday(&end_2, NULL);
    timeRecorder[11] += (double)((end_2.tv_sec-start_2.tv_sec)*1000000+(end_2.tv_usec-start_2.tv_usec))/1000;
    // <!!> TIME PART II-II-II: end

    // <!!> TIME PART II-II-III: this part needs a time recorder
	// std::cout << "part 2-2-3" << std::endl;
	gettimeofday(&start_2, NULL);
    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        this->softmax.backward1(target_dist[i], data->tgt[i], grad.softmaxGrad);
    }

    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        this->softmax.backward2(target_dist[i], decState[i]->delh);
    }
    for (int index = 0; index < 512; ++ index)
    {
        for (int i = 0; i < (int)data->tgt.size(); ++ i)
        {
            this->softmax.backward3(decState[i]->h, target_dist[i], grad.softmaxGrad, index);
        }
    }

	int xx_size = (int)data->tgt.size()-1; 
	//std::cout << "during softmax delh at " << xx_size << " is " << decState[xx_size]->delh.squaredNorm() << std::endl;
	//std::cout << "during softmax delh at " << 5 << " is " << decState[5]->delh.squaredNorm() << std::endl;
    gettimeofday(&end_2, NULL);
    timeRecorder[12] += (double)((end_2.tv_sec-start_2.tv_sec)*1000000+(end_2.tv_usec-start_2.tv_usec))/1000;
    // <!!> TIME PART II-II-III: end
	gettimeofday(&end_1, NULL);
	timeRecorder[2] += (double)((end_2.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_2.tv_usec))/1000;
	//std::cout << "softmax.grad = " << grad.softmaxGrad.norm() << std::endl; 
    gettimeofday(&end, NULL);
    timeRecorder[3] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
    // <!!> TIME PART II: end

    // this part can't be removed
  	decState[data->tgt.size()-1]->delc = this->zeros;
	/*
	decState[data->tgt.size()-1]->delc = this->zeros; // <??> can be faster by using setZeros?
	std::cout << "////////after assign this->zeros//////" << std::endl;
	std::cout << decState[data->tgt.size()-1]->delc.array() << std::endl;
	decState[data->tgt.size()-1]->delc.setZero();
	std::cout << "**** before *********************" << std::endl;
	std::cout << "xx cTanh = " << decState[1]->cTanh.squaredNorm() << std::endl;
	std::cout << "xx delh = " << decState[1]->delh.squaredNorm() << std::endl;
	std::cout << "xx o = " << decState[1]->o.squaredNorm() << std::endl;
	std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
	std::cout << "curr cTanh = " << decState[xx_size]->cTanh.squaredNorm() << std::endl;
	std::cout << "curr delh = " << decState[xx_size]->delh.squaredNorm() << std::endl;
	std::cout << "curr o = " << decState[xx_size]->o.squaredNorm() << std::endl;
	std::cout << "curr delc = " << decState[xx_size]->delc.squaredNorm() << std::endl;
	std::cout << "curr f = " << decState[xx_size]->f.squaredNorm() << std::endl;
	std::cout << "prev delc = " << decState[xx_size-1]->delc.squaredNorm() << std::endl;
	std::cout << "curr i = " << decState[xx_size]->i.squaredNorm() << std::endl;
	std::cout << "curr u = " << decState[xx_size]->u.squaredNorm() << std::endl;
	std::cout << "curr c = " << decState[xx_size]->c.squaredNorm() << std::endl;
	std::cout << "curr delx = " << decState[xx_size]->delx.squaredNorm() << std::endl;
	std::cout << "prev delh = " << decState[xx_size-1]->delh.squaredNorm() << std::endl;
	std::cout << "*********************************" << std::endl;
	*/
	// <!!> TIME PART III: this part needs a time recorder
	// std::cout << "part 3" << std::endl;
	gettimeofday(&start, NULL);
	for (int i = data->src.size()-1; i >= 1; -- i)
	{
		//delisBuffer[i-1].setZero();
		//delfsBuffer[i-1].setZero();
		//delosBuffer[i-1].setZero();
		//delusBuffer[i-1].setZero();
		delisBuffer[i-1] = this->zeros;
		delfsBuffer[i-1] = this->zeros;
		delosBuffer[i-1] = this->zeros;
		delfsBuffer[i-1] = this->zeros;
	}
    // <!!> TIME PART III-I: this part needs a time recorder
    gettimeofday(&start_1, NULL);
    int xx_boarder = 1;
	for (int i = data->tgt.size()-1; i >= 1; --i)
    {
        decState[i-1]->delc = this->zeros;
        this->dec.backward1(decState[i-1], decState[i], delosBuffer[i-1], delisBuffer[i-1], delusBuffer[i-1], delfsBuffer[i-1]);
	}
	/*
	VecD xx_delo = ActFunc::logisticPrime(decState[1]->cTanh).array() * decState[1]->delh.array() * decState[1]->o.array();
	std::cout << "xx_delo[1] = " << xx_delo.squaredNorm() << std::endl;
	std::cout << "hello cTanh = " << decState[1]->cTanh.squaredNorm() << std::endl;
	std::cout << "hello delh = " << decState[xx_size-1]->delh.squaredNorm() << std::endl;
	std::cout << "hello o = " << decState[1]->o.squaredNorm() << std::endl;
 
	std::cout << "***** after ************************" << std::endl;
	std::cout << "xx cTanh = " << decState[1]->cTanh.squaredNorm() << std::endl;
	std::cout << "xx delh = " << decState[1]->delh.squaredNorm() << std::endl;
	std::cout << "xx o = " << decState[1]->o.squaredNorm() << std::endl;
	std::cout << "++++++++++++++++++++++++++++++++++++" << std::endl;
	std::cout << "curr cTanh = " << decState[xx_size]->cTanh.squaredNorm() << std::endl;
	std::cout << "curr delh = " << decState[xx_size]->delh.squaredNorm() << std::endl;
	std::cout << "curr o = " << decState[xx_size]->o.squaredNorm() << std::endl;
	std::cout << "curr delc = " << decState[xx_size]->delc.squaredNorm() << std::endl;
	std::cout << "curr f = " << decState[xx_size]->f.squaredNorm() << std::endl;
	std::cout << "prev delc = " << decState[xx_size-1]->delc.squaredNorm() << std::endl;
	std::cout << "curr i = " << decState[xx_size]->i.squaredNorm() << std::endl;
	std::cout << "curr u = " << decState[xx_size]->u.squaredNorm() << std::endl;
	std::cout << "curr c = " << decState[xx_size]->c.squaredNorm() << std::endl;
	std::cout << "curr delx = " << decState[xx_size]->delx.squaredNorm() << std::endl;
	std::cout << "prev delh = " << decState[xx_size-1]->delh.squaredNorm() << std::endl;
	std::cout << "*************************" << std::endl;
	
	// 8+4 for loopi
	*/

    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Wxi.noalias() += delisBuffer[i-1] * this->targetEmbed.col(data->tgt[i-1]).transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Whi.noalias() += delisBuffer[i-1] * decState[i-1]->h.transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Wxf.noalias() += delfsBuffer[i-1] * this->targetEmbed.col(data->tgt[i-1]).transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Whf.noalias() += delfsBuffer[i-1] * decState[i-1]->h.transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Wxo.noalias() += delosBuffer[i-1] * this->targetEmbed.col(data->tgt[i-1]).transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Who.noalias() += delosBuffer[i-1] * decState[i-1]->h.transpose(); 
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Wxu.noalias() += delusBuffer[i-1] * this->targetEmbed.col(data->tgt[i-1]).transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Whu.noalias() += delusBuffer[i-1] * decState[i-1]->h.transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bi += delisBuffer[i-1];
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bf += delfsBuffer[i-1];
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bo += delosBuffer[i-1];
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bu += delusBuffer[i-1];
    }
	//std::cout << "1st = " << grad.lstmTgtGrad.Wxi.coeff(5, 5) << std::endl;
    gettimeofday(&end_1, NULL);
    timeRecorder[4] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
    // <!!> TIME PART III-I: end

    // <!!> TIME PART III-II: this part need a time recorder
	// std::cout << "part 3-2" << std::endl;
	gettimeofday(&start_1, NULL);
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        if (grad.targetEmbed.count(data->tgt[i-1]))
        {
            grad.targetEmbed.at(data->tgt[i-1]) += decState[i]->delx;
        }
        else
        {
            grad.targetEmbed[data->tgt[i-1]] = decState[i]->delx;
        }
    }
    gettimeofday(&end_1, NULL);
    timeRecorder[5] += (double)((end_1.tv_sec-start.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
    // <!!> TIME PART III-II: end
    gettimeofday(&end, NULL);
    timeRecorder[6] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
    // <!!> TIME PART III: end
	//std::cout << "lstmTgtGrad = " << grad.lstmTgtGrad.norm() << std::endl;
    
	//int xx_input;
	// std::cin >> xx_input;
	
	// this part can't be removed
    encState[data->src.size()]->delc = decState[0]->delc;
    encState[data->src.size()]->delh = decState[0]->delh;

    // <!!> TIME PART IV: this part needs a time recorder
	// std::cout << "part 4" << std::endl;
	gettimeofday(&start, NULL);
	for (int i = data->tgt.size(); i >= 1; -- i)
	{
		//delisBuffer[i-1].setZero();
		//delfsBuffer[i-1].setZero();
		//delosBuffer[i-1].setZero();
		//delusBuffer[i-1].setZero();
		delisBuffer[i-1] = this->zeros;
		delfsBuffer[i-1] = this->zeros;
		delosBuffer[i-1] = this->zeros;
		delusBuffer[i-1] = this->zeros;
	}
    // <!!> TIME PART IV-I: this part needs a time recorder
	// std::cout << "part 4-1" << std::endl;
	gettimeofday(&start_1, NULL);
    for (int i = data->src.size(); i >= 1; -- i)
    {
        encState[i-1]->delh = this->zeros;
        encState[i-1]->delc = this->zeros;
        enc.backward1(encState[i-1], encState[i], delosBuffer[i-1], delisBuffer[i-1], delusBuffer[i-1], delfsBuffer[i-1]);
    }
    // 8 + 4 for loop
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Wxi.noalias() += delisBuffer[i-1] * this->sourceEmbed.col(data->src[i-1]).transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Whi.noalias() += delisBuffer[i-1] * encState[i-1]->h.transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Wxf.noalias() += delfsBuffer[i-1] * this->sourceEmbed.col(data->src[i-1]).transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Whf.noalias() += delfsBuffer[i-1] * encState[i-1]->h.transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Wxo.noalias() += delosBuffer[i-1] * this->sourceEmbed.col(data->src[i-1]).transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Who.noalias() += delosBuffer[i-1] * encState[i-1]->h.transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Wxu.noalias() += delusBuffer[i-1] * this->sourceEmbed.col(data->src[i-1]).transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Whu.noalias() += delusBuffer[i-1] * encState[i-1]->h.transpose();
    }

    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bi += delisBuffer[i-1];
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bf += delfsBuffer[i-1];
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bo += delosBuffer[i-1];
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bu += delusBuffer[i-1];
    }

    gettimeofday(&end_1, NULL);
    timeRecorder[7] += ((end_1.tv_sec-start_1.tv_sec)*1000000 + (end_1.tv_usec-start_1.tv_usec))/1000;
    // <!!> TIME PART IV-I: end

    // <!!> TIME PART IV-II: this part needs a time recorder
	// std::cout << "part 4-2" << std::endl;
	gettimeofday(&start_1, NULL);
    for (int i = data->src.size(); i >= 1; -- i)
    {
        if (grad.sourceEmbed.count(data->src[i-1]))
        {
            grad.sourceEmbed.at(data->src[i-1]) += encState[i]->delx;
        }
        else
        {
            grad.sourceEmbed[data->src[i-1]] = encState[i]->delx;
        }
    }
    gettimeofday(&end_1, NULL);
    timeRecorder[8] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000 + (end_1.tv_usec-start_1.tv_usec))/1000;
    // <!!> TIME PART IV-II: end

    gettimeofday(&end, NULL);
    timeRecorder[9] += (double)((end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec))/1000;
    // <!!> TIME PART IV: end
	//std::cout << "lstmSrcGrad = " << grad.lstmSrcGrad.norm() << std::endl;
} // end of train_qiao_4

void EncDec::train_qiao_5(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<double>& timeRecorder, std::vector<VecD>& target_dist, MatD& delosBuffer, MatD& delisBuffer, MatD& delusBuffer, MatD& delfsBuffer)
{	
	// this function is for recording the time of each parts of train
	// to check the scalability for each parts
	struct timeval start, end;
	struct timeval start_1, end_1;
	struct timeval start_2, end_2;

	VecD targetDist; // <??> created in stack of this thread
	int hidden_dim = delosBuffer.rows();
	int max_num_terms = delosBuffer.cols();

	loss = 0.0;

	// <!!> TIME PART I: this part needs a time recorder
	// std::cout << "part 1" << std::endl;
	gettimeofday(&start, NULL);
	this->encode(data->src, encState);
	gettimeofday(&end, NULL);
	timeRecorder[0] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
	// <!!> TIME PART I: end
	// std::cout << "time part I" << std::endl;	
    
    // <!!> TIME PART II: this part needs a time recorder
	// std::cout << "part 2" << std::endl;
	gettimeofday(&start, NULL);

	// <!!> TIME PART II-1: this part needs a time recorder
	// std::cout << "part 2-1" << std::endl;
	gettimeofday(&start_1, NULL);
    // when i = 0, pass h and c from enc to dec
    decState[0]->h = encState[data->src.size()]->h;
    decState[0]->c = encState[data->src.size()]->c;
    // when i > 0, do forward function
	for (int i = 1; i < (int)data->tgt.size(); ++ i)
    {
        this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), decState[i-1], decState[i]);   
    }
    gettimeofday(&end_1, NULL);
    timeRecorder[1] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
    // <!!> TIME PART II-I: end
    
    // <!!> TIME PART II-II: this part needs a time recorder
	// std::cout << "part 2-2" << std::endl;
	gettimeofday(&start_1, NULL);
    // <!!> TIME PART II-II-I: this time needs a time recorder
	// std::cout << "part 2-2-1" << std::endl;
    gettimeofday(&start_2, NULL);
	// std::cout << "tgt.size = " << (int)data->tgt.size() << std::endl;
    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        this->softmax.calcDist(decState[i]->h, target_dist[i]);
    }
	// std::cout << "xxx" << std::endl;
    gettimeofday(&end_2, NULL);
    timeRecorder[10] += (double)((end_2.tv_sec-start_2.tv_sec)*1000000+(end_2.tv_usec-start_2.tv_usec))/1000;
    // <!!> TIME PART II-II-I: end

    // <!!> TIME PART II-II-II: this time needs a time recorder
	// std::cout << "part 2-2-2" << std::endl;
	gettimeofday(&start_2, NULL);
    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        loss += this->softmax.calcLoss(target_dist[i], data->tgt[i]);
		
    }
	//std::cout << "loss = " << loss << std::endl;
    gettimeofday(&end_2, NULL);
    timeRecorder[11] += (double)((end_2.tv_sec-start_2.tv_sec)*1000000+(end_2.tv_usec-start_2.tv_usec))/1000;
    // <!!> TIME PART II-II-II: end

    // <!!> TIME PART II-II-III: this part needs a time recorder
	// std::cout << "part 2-2-3" << std::endl;
	gettimeofday(&start_2, NULL);
    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        this->softmax.backward1(target_dist[i], data->tgt[i], grad.softmaxGrad);
    }

    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        this->softmax.backward2(target_dist[i], decState[i]->delh);
    }
    for (int index = 0; index < 512; ++ index)
    {
        for (int i = 0; i < (int)data->tgt.size(); ++ i)
        {
            this->softmax.backward3(decState[i]->h, target_dist[i], grad.softmaxGrad, index);
        }
    }

	int xx_size = (int)data->tgt.size()-1; 
	//std::cout << "during softmax delh at " << xx_size << " is " << decState[xx_size]->delh.squaredNorm() << std::endl;
	//std::cout << "during softmax delh at " << 5 << " is " << decState[5]->delh.squaredNorm() << std::endl;
    gettimeofday(&end_2, NULL);
    timeRecorder[12] += (double)((end_2.tv_sec-start_2.tv_sec)*1000000+(end_2.tv_usec-start_2.tv_usec))/1000;
    // <!!> TIME PART II-II-III: end
	gettimeofday(&end_1, NULL);
	timeRecorder[2] += (double)((end_2.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_2.tv_usec))/1000;
	//std::cout << "softmax.grad = " << grad.softmaxGrad.norm() << std::endl; 
    gettimeofday(&end, NULL);
    timeRecorder[3] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
    // <!!> TIME PART II: end

    // this part can't be removed
  	decState[data->tgt.size()-1]->delc = this->zeros;
	// <!!> TIME PART III: this part needs a time recorder
	// std::cout << "part 3" << std::endl;
	gettimeofday(&start, NULL);
	for (int i = data->tgt.size(); i >= 1; -- i)
	{
		delisBuffer.col(i-1) = this->zeros;
		delfsBuffer.col(i-1) = this->zeros;
		delosBuffer.col(i-1) = this->zeros;
		delusBuffer.col(i-1) = this->zeros;
	}
	//delisBuffer.setZero(hidden_dim, max_num_terms);
	//delfsBuffer.setZero(hidden_dim, max_num_terms);
	//delosBuffer.setZero(hidden_dim, max_num_terms);
	//delusBuffer.setZero(hidden_dim, max_num_terms);
	
	// create and initialize the buffer that store the h and embd for target
	MatD targetEmbedBuffer = MatD(data->tgt.size()-1, hidden_dim);
	MatD decStateHBuffer = MatD(data->tgt.size()-1, hidden_dim);
	for (int i = data->tgt.size()-1; i >= 1; -- i)
	{
		targetEmbedBuffer.row(i-1) = this->targetEmbed.col(data->tgt[i-1]).transpose();
	}
	for (int i = data->tgt.size()-1; i >= 1; -- i)
	{
		decStateHBuffer.row(i-1) = decState[i-1]->h.transpose();
	}
	
	
	// <!!> TIME PART III-I: this part needs a time recorder
    gettimeofday(&start_1, NULL);
    int xx_boarder = 1;
	for (int i = data->tgt.size()-1; i >= 1; --i)
    {
        decState[i-1]->delc = this->zeros;
        // this->dec.backward1(decState[i-1], decState[i], delosBuffer[i-1], delisBuffer[i-1], delusBuffer[i-1], delfsBuffer[i-1]);
		this->dec.backward1_v2(decState[i-1], decState[i], delosBuffer, delisBuffer, delusBuffer, delfsBuffer, i-1);
	}

	grad.lstmTgtGrad.Wxi.noalias() += delisBuffer.leftCols(data->tgt.size()-1) * targetEmbedBuffer;
	grad.lstmTgtGrad.Whi.noalias() += delisBuffer.leftCols(data->tgt.size()-1) * decStateHBuffer;
	grad.lstmTgtGrad.Wxf.noalias() += delfsBuffer.leftCols(data->tgt.size()-1) * targetEmbedBuffer;
	grad.lstmTgtGrad.Whf.noalias() += delfsBuffer.leftCols(data->tgt.size()-1) * decStateHBuffer;
	grad.lstmTgtGrad.Wxo.noalias() += delosBuffer.leftCols(data->tgt.size()-1) * targetEmbedBuffer;
	grad.lstmTgtGrad.Who.noalias() += delosBuffer.leftCols(data->tgt.size()-1) * decStateHBuffer;
	grad.lstmTgtGrad.Wxu.noalias() += delusBuffer.leftCols(data->tgt.size()-1) * targetEmbedBuffer;
	grad.lstmTgtGrad.Whu.noalias() += delusBuffer.leftCols(data->tgt.size()-1) * decStateHBuffer;
    
	for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bi += delisBuffer.col(i-1);
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bf += delfsBuffer.col(i-1);
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bo += delosBuffer.col(i-1);
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bu += delusBuffer.col(i-1);
    }
	//std::cout << "1st = "<< grad.lstmTgtGrad.Wxi.coeff(5, 5) << std::endl;
    gettimeofday(&end_1, NULL);
    timeRecorder[4] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
    // <!!> TIME PART III-I: end

    // <!!> TIME PART III-II: this part need a time recorder
	// std::cout << "part 3-2" << std::endl;
	gettimeofday(&start_1, NULL);
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        if (grad.targetEmbed.count(data->tgt[i-1]))
        {
            grad.targetEmbed.at(data->tgt[i-1]) += decState[i]->delx;
        }
        else
        {
            grad.targetEmbed[data->tgt[i-1]] = decState[i]->delx;
        }
    }
    gettimeofday(&end_1, NULL);
    timeRecorder[5] += (double)((end_1.tv_sec-start.tv_sec)*1000000+(end_1.tv_usec-start_1.tv_usec))/1000;
    // <!!> TIME PART III-II: end
    gettimeofday(&end, NULL);
    timeRecorder[6] += (double)((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
    // <!!> TIME PART III: end
	//std::cout << "lstmTgtGrad = " << grad.lstmTgtGrad.norm() << std::endl;
    
	//int xx_input;
	// std::cin >> xx_input;
	
	// this part can't be removed
    encState[data->src.size()]->delc = decState[0]->delc;
    encState[data->src.size()]->delh = decState[0]->delh;

    // <!!> TIME PART IV: this part needs a time recorder
	// std::cout << "part 4" << std::endl;
	gettimeofday(&start, NULL);
	for (int i = data->src.size(); i >= 1; -- i)
	{
		delisBuffer.col(i-1) = this->zeros;
		delfsBuffer.col(i-1) = this->zeros;
		delosBuffer.col(i-1) = this->zeros;
		delusBuffer.col(i-1) = this->zeros;
	}
    
	// delisBuffer.setZero(hidden_dim, max_num_terms);
	// delfsBuffer.setZero(hidden_dim, max_num_terms);
	// delosBuffer.setZero(hidden_dim, max_num_terms);
	// delusBuffer.setZero(hidden_dim, max_num_terms);
	
	MatD sourceEmbedBuffer = MatD(data->src.size(), hidden_dim);
	MatD encStateHBuffer = MatD(data->src.size(), hidden_dim);
	for (int i = data->src.size(); i >= 1; -- i)
	{
		sourceEmbedBuffer.row(i-1) = this->sourceEmbed.col(data->src[i-1]).transpose();
	}
	for (int i = data->src.size(); i >= 1; -- i)
	{
		encStateHBuffer.row(i-1) = encState[i-1]->h.transpose();
	}	
	// <!!> TIME PART IV-I: this part needs a time recorder
	// std::cout << "part 4-1" << std::endl;
	gettimeofday(&start_1, NULL);
    for (int i = data->src.size(); i >= 1; -- i)
    {
        encState[i-1]->delh = this->zeros;
        encState[i-1]->delc = this->zeros;
        // enc.backward1(encState[i-1], encState[i], delosBuffer[i-1], delisBuffer[i-1], delusBuffer[i-1], delfsBuffer[i-1]);
		enc.backward1_v2(encState[i-1], encState[i], delosBuffer, delisBuffer, delusBuffer, delfsBuffer, i-1);
	}
    
	grad.lstmSrcGrad.Wxi.noalias() += delisBuffer.leftCols(data->src.size()) * sourceEmbedBuffer;
	grad.lstmSrcGrad.Whi.noalias() += delisBuffer.leftCols(data->src.size()) * encStateHBuffer;
	grad.lstmSrcGrad.Wxf.noalias() += delfsBuffer.leftCols(data->src.size()) * sourceEmbedBuffer;
	grad.lstmSrcGrad.Whf.noalias() += delfsBuffer.leftCols(data->src.size()) * encStateHBuffer;
	grad.lstmSrcGrad.Wxo.noalias() += delosBuffer.leftCols(data->src.size()) * sourceEmbedBuffer;
	grad.lstmSrcGrad.Who.noalias() += delosBuffer.leftCols(data->src.size()) * encStateHBuffer;
	grad.lstmSrcGrad.Wxu.noalias() += delusBuffer.leftCols(data->src.size()) * sourceEmbedBuffer;
	grad.lstmSrcGrad.Whu.noalias() += delusBuffer.leftCols(data->src.size()) * encStateHBuffer;
	
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bi += delisBuffer.col(i-1);
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bf += delfsBuffer.col(i-1);
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bo += delosBuffer.col(i-1);
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bu += delusBuffer.col(i-1);
    }

    gettimeofday(&end_1, NULL);
    timeRecorder[7] += ((end_1.tv_sec-start_1.tv_sec)*1000000 + (end_1.tv_usec-start_1.tv_usec))/1000;
    // <!!> TIME PART IV-I: end

    // <!!> TIME PART IV-II: this part needs a time recorder
	// std::cout << "part 4-2" << std::endl;
	gettimeofday(&start_1, NULL);
    for (int i = data->src.size(); i >= 1; -- i)
    {
        if (grad.sourceEmbed.count(data->src[i-1]))
        {
            grad.sourceEmbed.at(data->src[i-1]) += encState[i]->delx;
        }
        else
        {
            grad.sourceEmbed[data->src[i-1]] = encState[i]->delx;
        }
    }
    gettimeofday(&end_1, NULL);
    timeRecorder[8] += (double)((end_1.tv_sec-start_1.tv_sec)*1000000 + (end_1.tv_usec-start_1.tv_usec))/1000;
    // <!!> TIME PART IV-II: end

    gettimeofday(&end, NULL);
    timeRecorder[9] += (double)((end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec))/1000;
    // <!!> TIME PART IV: end
	//std::cout << "lstmSrcGrad = " << grad.lstmSrcGrad.norm() << std::endl;
} // end of train_qiao_5

void EncDec::train_mf_1(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, MemoryFootprint* mf){
  	VecD targetDist; // <??> created in stack of this thread

  	loss = 0.0;
  	this->encode(data->src, encState);
  	// std::cout << "after encode" << std::endl;
  	for (int i = 0; i < (int)data->tgt.size(); ++i)
  	{
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
} // end of train_mf_1

void EncDec::train_mf_2(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, MemoryFootprint* mf){
  	VecD targetDist; // <??> created in stack of this thread

  	loss = 0.0;
  	this->encode_mf_v1(data->src, encState, mf);
  	// std::cout << "after encode" << std::endl;
  	for (int i = 0; i < (int)data->tgt.size(); ++i)
  	{
  		if (i == 0){
  			decState[0]->h = encState[data->src.size()]->h;
  			decState[0]->c = encState[data->src.size()]->c;
  		}
  		else {
  			this->dec.forward_mf_v1(this->targetEmbed.col(data->tgt[i-1]), decState[i-1], decState[i], mf);
  		}

    	// <??> PART S: THIS PART MAY BE THE BOTTLENEC
    	// <??> PART S: BEGIN
    	this->softmax.calcDist_mf_v1(decState[i]->h, targetDist, mf); // <??> are each thead competing for softmax
    	loss += this->softmax.calcLoss_mf_v1(targetDist, data->tgt[i], mf); // <??> same question as above
    	this->softmax.backward_mf_v1(decState[i]->h, targetDist, data->tgt[i], decState[i]->delh, grad.softmaxGrad, mf);
    	// <??> PART S: END
    }

	decState[data->tgt.size()-1]->delc = this->zeros; // <??> can be faster by using setZeros?


	for (int i = data->tgt.size()-1; i >= 1; --i){
    	decState[i-1]->delc = this->zeros; // <??> can be faster by using setZeros?
    	this->dec.backward_mf_v1(decState[i-1], decState[i], grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]), mf);

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

		this->enc.backward_mf_v1(encState[i-1], encState[i], grad.lstmSrcGrad, this->sourceEmbed.col(data->src[i-1]), mf);

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
} // end of train_mf_2
void EncDec::train_new_v1(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<VecD>& target_dist, std::vector<VecD>& delosBuffer, std::vector<VecD>& delisBuffer, std::vector<VecD>& delusBuffer, std::vector<VecD>& delfsBuffer)
{	
	VecD targetDist; // <??> created in stack of this thread

	loss = 0.0;

	this->encode(data->src, encState);
    // when i = 0, pass h and c from enc to dec
    decState[0]->h = encState[data->src.size()]->h;
    decState[0]->c = encState[data->src.size()]->c;
    // when i > 0, do forward function
	for (int i = 1; i < (int)data->tgt.size(); ++ i)
    {
        this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), decState[i-1], decState[i]);   
    }
    
    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        this->softmax.calcDist(decState[i]->h, target_dist[i]);
    }
    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        loss += this->softmax.calcLoss(target_dist[i], data->tgt[i]);
		
    }
    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        this->softmax.backward1(target_dist[i], data->tgt[i], grad.softmaxGrad);
    }

    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        this->softmax.backward2(target_dist[i], decState[i]->delh);
    }
    for (int index = 0; index < 512; ++ index)
    {
        for (int i = 0; i < (int)data->tgt.size(); ++ i)
        {
            this->softmax.backward3(decState[i]->h, target_dist[i], grad.softmaxGrad, index);
        }
    }

	int xx_size = (int)data->tgt.size()-1; 

    // this part can't be removed
  	decState[data->tgt.size()-1]->delc = this->zeros;
	for (int i = data->src.size()-1; i >= 1; -- i)
	{
		delisBuffer[i-1] = this->zeros;
		delfsBuffer[i-1] = this->zeros;
		delosBuffer[i-1] = this->zeros;
		delfsBuffer[i-1] = this->zeros;
	}
    int xx_boarder = 1;
	for (int i = data->tgt.size()-1; i >= 1; --i)
    {
        decState[i-1]->delc = this->zeros;
        this->dec.backward1(decState[i-1], decState[i], delosBuffer[i-1], delisBuffer[i-1], delusBuffer[i-1], delfsBuffer[i-1]);
	}

    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Wxi.noalias() += delisBuffer[i-1] * this->targetEmbed.col(data->tgt[i-1]).transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Whi.noalias() += delisBuffer[i-1] * decState[i-1]->h.transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Wxf.noalias() += delfsBuffer[i-1] * this->targetEmbed.col(data->tgt[i-1]).transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Whf.noalias() += delfsBuffer[i-1] * decState[i-1]->h.transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Wxo.noalias() += delosBuffer[i-1] * this->targetEmbed.col(data->tgt[i-1]).transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Who.noalias() += delosBuffer[i-1] * decState[i-1]->h.transpose(); 
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Wxu.noalias() += delusBuffer[i-1] * this->targetEmbed.col(data->tgt[i-1]).transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.Whu.noalias() += delusBuffer[i-1] * decState[i-1]->h.transpose();
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bi += delisBuffer[i-1];
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bf += delfsBuffer[i-1];
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bo += delosBuffer[i-1];
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bu += delusBuffer[i-1];
    }
    
	for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        if (grad.targetEmbed.count(data->tgt[i-1]))
        {
            grad.targetEmbed.at(data->tgt[i-1]) += decState[i]->delx;
        }
        else
        {
            grad.targetEmbed[data->tgt[i-1]] = decState[i]->delx;
        }
    }
	
	// this part can't be removed
    encState[data->src.size()]->delc = decState[0]->delc;
    encState[data->src.size()]->delh = decState[0]->delh;

	for (int i = data->tgt.size(); i >= 1; -- i)
	{
		delisBuffer[i-1] = this->zeros;
		delfsBuffer[i-1] = this->zeros;
		delosBuffer[i-1] = this->zeros;
		delusBuffer[i-1] = this->zeros;
	}
    for (int i = data->src.size(); i >= 1; -- i)
    {
        encState[i-1]->delh = this->zeros;
        encState[i-1]->delc = this->zeros;
        enc.backward1(encState[i-1], encState[i], delosBuffer[i-1], delisBuffer[i-1], delusBuffer[i-1], delfsBuffer[i-1]);
    }
    // 8 + 4 for loop
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Wxi.noalias() += delisBuffer[i-1] * this->sourceEmbed.col(data->src[i-1]).transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Whi.noalias() += delisBuffer[i-1] * encState[i-1]->h.transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Wxf.noalias() += delfsBuffer[i-1] * this->sourceEmbed.col(data->src[i-1]).transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Whf.noalias() += delfsBuffer[i-1] * encState[i-1]->h.transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Wxo.noalias() += delosBuffer[i-1] * this->sourceEmbed.col(data->src[i-1]).transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Who.noalias() += delosBuffer[i-1] * encState[i-1]->h.transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Wxu.noalias() += delusBuffer[i-1] * this->sourceEmbed.col(data->src[i-1]).transpose();
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.Whu.noalias() += delusBuffer[i-1] * encState[i-1]->h.transpose();
    }

    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bi += delisBuffer[i-1];
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bf += delfsBuffer[i-1];
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bo += delosBuffer[i-1];
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bu += delusBuffer[i-1];
    }

    for (int i = data->src.size(); i >= 1; -- i)
    {
        if (grad.sourceEmbed.count(data->src[i-1]))
        {
            grad.sourceEmbed.at(data->src[i-1]) += encState[i]->delx;
        }
        else
        {
            grad.sourceEmbed[data->src[i-1]] = encState[i]->delx;
        }
    }
} // end of train_new_v1

void EncDec::train_new_v2(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss, std::vector<VecD>& target_dist, MatD& delosBuffer, MatD& delisBuffer, MatD& delusBuffer, MatD& delfsBuffer)
{	
	VecD targetDist; // <??> created in stack of this thread

	loss = 0.0;
	int hidden_dim = delosBuffer.rows();
	int max_num_terms = delosBuffer.cols();

	this->encode(data->src, encState);
    // when i = 0, pass h and c from enc to dec
    decState[0]->h = encState[data->src.size()]->h;
    decState[0]->c = encState[data->src.size()]->c;
    // when i > 0, do forward function
	for (int i = 1; i < (int)data->tgt.size(); ++ i)
    {
        this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), decState[i-1], decState[i]);   
    }
    
    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        this->softmax.calcDist(decState[i]->h, target_dist[i]);
    }
    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        loss += this->softmax.calcLoss(target_dist[i], data->tgt[i]);
		
    }
    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        this->softmax.backward1(target_dist[i], data->tgt[i], grad.softmaxGrad);
    }

    for (int i = 0; i < (int)data->tgt.size(); ++ i)
    {
        this->softmax.backward2(target_dist[i], decState[i]->delh);
    }
    for (int index = 0; index < 512; ++ index)
    {
        for (int i = 0; i < (int)data->tgt.size(); ++ i)
        {
            this->softmax.backward3(decState[i]->h, target_dist[i], grad.softmaxGrad, index);
        }
    }

	int xx_size = (int)data->tgt.size()-1; 

    // this part can't be removed
  	decState[data->tgt.size()-1]->delc = this->zeros;
	for (int i = data->tgt.size()-1; i >= 1; -- i)
	{
		delisBuffer.col(i-1) = this->zeros;
		delfsBuffer.col(i-1) = this->zeros;
		delosBuffer.col(i-1) = this->zeros;
		delfsBuffer.col(i-1) = this->zeros;
	}
	// delisBuffer.setZero(hidden_dim, max_num_terms);
	// delfsBuffer.setZero(hidden_dim, max_num_terms);
	// delosBuffer.setZero(hidden_dim, max_num_terms);
	// delusBuffer.setZero(hidden_dim, max_num_terms);
    
	// create and initialize the buffer that store the h and embed for target
	MatD targetEmbedBuffer = MatD(data->tgt.size()-1, hidden_dim);
	MatD decStateHBuffer = MatD(data->tgt.size()-1, hidden_dim);
	for (int i = data->tgt.size()-1; i >= 1; -- i)
	{
		targetEmbedBuffer.row(i-1) = this->targetEmbed.col(data->tgt[i-1]).transpose();
	}
	for (int i = data->tgt.size()-1; i >= 1; -- i)
	{
		decStateHBuffer.row(i-1) = decState[i-1]->h.transpose();
	}

	int xx_boarder = 1;
	for (int i = data->tgt.size()-1; i >= 1; --i)
    {
        decState[i-1]->delc = this->zeros;
        this->dec.backward1_v2(decState[i-1], decState[i], delosBuffer, delisBuffer, delusBuffer, delfsBuffer, i-1);
	}

	grad.lstmTgtGrad.Wxi.noalias() += delisBuffer.leftCols(data->tgt.size()-1) * targetEmbedBuffer;
	grad.lstmTgtGrad.Whi.noalias() += delisBuffer.leftCols(data->tgt.size()-1) * decStateHBuffer;
	grad.lstmTgtGrad.Wxf.noalias() += delfsBuffer.leftCols(data->tgt.size()-1) * targetEmbedBuffer;
	grad.lstmTgtGrad.Whf.noalias() += delfsBuffer.leftCols(data->tgt.size()-1) * decStateHBuffer;
	grad.lstmTgtGrad.Wxo.noalias() += delosBuffer.leftCols(data->tgt.size()-1) * targetEmbedBuffer;
	grad.lstmTgtGrad.Who.noalias() += delosBuffer.leftCols(data->tgt.size()-1) * decStateHBuffer;
	grad.lstmTgtGrad.Wxu.noalias() += delusBuffer.leftCols(data->tgt.size()-1) * targetEmbedBuffer;
	grad.lstmTgtGrad.Whu.noalias() += delusBuffer.leftCols(data->tgt.size()-1) * decStateHBuffer;
    
	for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bi += delisBuffer.col(i-1);
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bf += delfsBuffer.col(i-1);
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bo += delosBuffer.col(i-1);
    }
    for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        grad.lstmTgtGrad.bu += delusBuffer.col(i-1);
    }
    
	for (int i = data->tgt.size()-1; i >= xx_boarder; -- i)
    {
        if (grad.targetEmbed.count(data->tgt[i-1]))
        {
            grad.targetEmbed.at(data->tgt[i-1]) += decState[i]->delx;
        }
        else
        {
            grad.targetEmbed[data->tgt[i-1]] = decState[i]->delx;
        }
    }
	
	// this part can't be removed
    encState[data->src.size()]->delc = decState[0]->delc;
    encState[data->src.size()]->delh = decState[0]->delh;

	for (int i = data->src.size(); i >= 1; -- i)
	{
		delisBuffer.col(i-1) = this->zeros;
		delfsBuffer.col(i-1) = this->zeros;
		delosBuffer.col(i-1) = this->zeros;
		delusBuffer.col(i-1) = this->zeros;
	}
	
	// delisBuffer.setZero(hidden_dim, max_num_terms);
	// delfsBuffer.setZero(hidden_dim, max_num_terms);
	// delosBuffer.setZero(hidden_dim, max_num_terms);
	// delusBuffer.setZero(hidden_dim, max_num_terms);
    
	MatD sourceEmbedBuffer = MatD(data->src.size(), hidden_dim);
	MatD encStateHBuffer = MatD(data->src.size(), hidden_dim);
	for (int i = data->src.size(); i >= 1; -- i)
	{
		sourceEmbedBuffer.row(i-1) = this->sourceEmbed.col(data->src[i-1]).transpose();
	}
	for (int i = data->src.size(); i >= 1; -- i)
	{
		encStateHBuffer.row(i-1) = encState[i-1]->h.transpose();
	}

	for (int i = data->src.size(); i >= 1; -- i)
    {
        encState[i-1]->delh = this->zeros;
        encState[i-1]->delc = this->zeros;
        enc.backward1_v2(encState[i-1], encState[i], delosBuffer, delisBuffer, delusBuffer, delfsBuffer, i-1);
    }

	grad.lstmSrcGrad.Wxi.noalias() += delisBuffer.leftCols(data->src.size()) * sourceEmbedBuffer;
	grad.lstmSrcGrad.Whi.noalias() += delisBuffer.leftCols(data->src.size()) * encStateHBuffer;
	grad.lstmSrcGrad.Wxf.noalias() += delfsBuffer.leftCols(data->src.size()) * sourceEmbedBuffer;
	grad.lstmSrcGrad.Whf.noalias() += delfsBuffer.leftCols(data->src.size()) * encStateHBuffer;
	grad.lstmSrcGrad.Wxo.noalias() += delosBuffer.leftCols(data->src.size()) * sourceEmbedBuffer;
	grad.lstmSrcGrad.Who.noalias() += delosBuffer.leftCols(data->src.size()) * encStateHBuffer;
	grad.lstmSrcGrad.Wxu.noalias() += delusBuffer.leftCols(data->src.size()) * sourceEmbedBuffer;
	grad.lstmSrcGrad.Whu.noalias() += delusBuffer.leftCols(data->src.size()) * encStateHBuffer;

    // 8 + 4 for loop

    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bi += delisBuffer.col(i-1);
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bf += delfsBuffer.col(i-1);
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bo += delosBuffer.col(i-1);
    }
    for (int i = data->src.size(); i >= 1; -- i)
    {
        grad.lstmSrcGrad.bu += delusBuffer.col(i-1);
    }

    for (int i = data->src.size(); i >= 1; -- i)
    {
        if (grad.sourceEmbed.count(data->src[i-1]))
        {
            grad.sourceEmbed.at(data->src[i-1]) += encState[i]->delx;
        }
        else
        {
            grad.sourceEmbed[data->src[i-1]] = encState[i]->delx;
        }
    }
} // end of train_new_v2

void EncDec::trainOpenMP(const Real learningRate, const int miniBatchSize, const int numThreads){
	static std::vector<EncDec::ThreadArg*> args;
	static std::vector<std::pair<int, int> > miniBatch;
	static EncDec::Grad grad;
	Real lossTrain = 0.0, perpDev = 0.0, denom = 0.0;
	Real gradNorm, lr = learningRate;
	const Real clipThreshold = 3.0;
	struct timeval start, end;
	std::cout << "size = " << sizeof(Real) << std::endl;
	// 
	struct timeval k_start, k_end;
	struct timeval k_start_1, k_end_1;
	struct timeval k_start_2, k_end_2;
	Real k_time = 0.0;
	Real k_time_1 = 0.0;
	Real k_time_2 = 0.0;

	if (args.empty()){
		for (int i = 0; i < numThreads; ++i)
		{
			args.push_back(new EncDec::ThreadArg(*this));

      		for (int j = 0; j < 200; ++j){    //<??> why j < 200 here
      			args[i]->encState.push_back(new LSTM::State);
      			args[i]->encState[0]->h = this->zeros;
      			args[i]->encState[0]->c = this->zeros;
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

  	// int max_batch_count = 4;
  	// int batch_count = 0;

	struct timeval start_temp, end_temp;

  	for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it){
		gettimeofday(&k_start, NULL);  
		

#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(args)
  		for (int i = it->first; i <= it->second; ++i){
  			int id = omp_get_thread_num();
  			Real loss;
  			iter_counter[id] ++;
			// the main training function
  			this->train(this->trainData[i], args[id]->encState, args[id]->decState, args[id]->grad, loss);
  			// end of the main training function
  			args[id]->loss += loss;
  		}

  		gettimeofday(&k_end, NULL);
  		
		Real temp_time = ((k_end.tv_sec-k_start.tv_sec)*1000000+(k_end.tv_usec-k_start.tv_usec))/1000.0;
  		k_time_1 += temp_time;
		std::cout << "for one minibatch: " << temp_time << std::endl << std::endl;
		gettimeofday(&k_start_1, NULL);
  		for (int id = 0; id < numThreads; ++id){
  			grad += args[id]->grad;
			args[id]->grad.init();
  			//args[id]->grad.init_qiao();
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
  		std::cout << "thread id  = " << ii << ", count = " << iter_counter[ii]+1 << std::endl;
  		sum_iter_counter += iter_counter[ii]+1;
  	}
  	std::cout << "sum = " << sum_iter_counter << std::endl;
  	
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

	return;
}

void EncDec::trainOpenMP_qiao(const Real learningRate, const int miniBatchSize, const int numThreads){
	
	static std::vector<EncDec::ThreadArg*> args;
	static std::vector<std::pair<int, int> > miniBatch;
	static EncDec::Grad grad;
	Real lossTrain = 0.0, perpDev = 0.0, denom = 0.0;
	Real gradNorm, lr = learningRate;
	const Real clipThreshold = 3.0;
	
	struct timeval start, end;
	
	std::cout << "size = " << sizeof(Real) << std::endl;
	
	// for recording time of different parts of train function
	int sizeTimers = 20;
	int numTimers = 13;
	static std::vector<EncDec::ThreadTimer*> timers;
	static std::vector<double> allBatchTimer; 

	struct timeval k_start, k_end;
	struct timeval k_start_1, k_end_1;
	struct timeval k_start_2, k_end_2;
	
	Real k_time_1 = 0.0;
	Real k_time_2 = 0.0;

	if (args.empty()){
		for (int i = 0; i < numThreads; ++i)
		{
			args.push_back(new EncDec::ThreadArg(*this));

      		for (int j = 0; j < 200; ++j){    //<??> why j < 200 here
      			args[i]->encState.push_back(new LSTM::State);
      			args[i]->encState[0]->h = this->zeros;
      			args[i]->encState[0]->c = this->zeros;
      			args[i]->decState.push_back(new LSTM::State);
      		}
      	}

      	for (int i = 0, step = this->trainData.size()/miniBatchSize; i < step; ++i){
      		miniBatch.push_back(std::pair<int, int>(i*miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*miniBatchSize-1)));
      	}

      	grad.lstmSrcGrad = LSTM::Grad(this->enc);
      	grad.lstmTgtGrad = LSTM::Grad(this->dec);
      	grad.softmaxGrad = SoftMax::Grad(this->softmax);

	}

	// init for timers
	if (timers.empty())
	{
		std::cout << "initialize timers" << std::endl;
		for (int i = 0; i < numThreads; ++i)
		{
			timers.push_back(new EncDec::ThreadTimer(*this,sizeTimers));
			std::cout << "for thread " << i << ", timeRecorder size = " << timers[i]->timeRecorder.size() << std::endl;
		}
	}
	if (allBatchTimer.size() < sizeTimers)
	{
		for (int i = 0; i < allBatchTimer.size(); i ++)
		{
			allBatchTimer[i] = 0.0;
		}
		for (int i = allBatchTimer.size(); i < sizeTimers; i++)
		{
			allBatchTimer.push_back(0.0);
		}
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
  	k_time_1 = 0.0;
  	k_time_2 = 0.0;

  	// int max_batch_count = 4;
  	// int batch_count = 0;

	struct timeval start_temp, end_temp;

  	for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it)
	{
		//  std::cout << "\r" << "Progress: " << ++count << "/" << miniBatch.size() << " mini batches" << std::flush;

		/* use for quick test to do stop after several mini batch
		batch_count ++;
		std::cout << "batch count = " << batch_count << std::endl << std::endl;
		if (batch_count > max_batch_count)
		{
			break;
		}
		*/


		gettimeofday(&k_start_1, NULL); // used to record the time used for parallel part 
		
		// parallel part 
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(args)
  		for (int i = it->first; i <= it->second; ++i){
  			int id = omp_get_thread_num();
  			Real loss;
  			iter_counter[id] ++;
  			
			// record the time used for one sentence: start point
			gettimeofday(&(time_rec_start[id]), NULL);

			// the main training function
  			this->train_qiao_3(this->trainData[i], args[id]->encState, args[id]->decState, args[id]->grad, loss, timers[id]->timeRecorder);
  			// end of the main training function
			
			// record the time used for one sentence: end point and save the time  
			gettimeofday(&(time_rec_end[id]), NULL);
  			sec_start[id][iter_counter[id]] = time_rec_start[id].tv_sec;
  			usec_start[id][iter_counter[id]] = time_rec_start[id].tv_usec;
  			sec_end[id][iter_counter[id]] = time_rec_end[id].tv_sec;
  			usec_end[id][iter_counter[id]] = time_rec_end[id].tv_usec;

  			args[id]->loss += loss;
  		}

  		gettimeofday(&k_end_1, NULL); //used to record the time used for parallel part
  		
		Real temp_time = ((k_end_1.tv_sec-k_start_1.tv_sec)*1000000+(k_end_1.tv_usec-k_start_1.tv_usec))/1000.0;
  		k_time_1 += temp_time;
		std::cout << "for one minibatch: " << temp_time << " ms" << std::endl << std::endl;

		// ouput the recorded time
		for (int i = 0; i < numTimers; i ++)
		{
			double sum = 0.0;
			for (int j = 0; j < numThreads; j ++)
			{
				sum += timers[j]->timeRecorder[i];
			}
			allBatchTimer[i] += sum/numThreads;
			// std::cout << "average time used in part " << i << " = " << sum/numThreads << " ms" << std::endl;
		}

		for (int id = 0; id < numThreads; id ++)
		{
			timers[id]->init();
		}

		// serial  part
		gettimeofday(&k_start_2, NULL); // record the time used for serial part 
  		
		for (int id = 0; id < numThreads; ++id){
  			grad += args[id]->grad;
			args[id]->grad.init();
  			//args[id]->grad.init_qiao();
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
  		
		gettimeofday(&k_end_2, NULL); // record the time used for serial part 
  		k_time_2 += ((k_end_2.tv_sec-k_start_2.tv_sec)*1000000+(k_end_2.tv_usec-k_start_2.tv_usec))/1000.0;
  	} // end of for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it)

  	std::cout << std::endl;
  	gettimeofday(&end, 0);
  	//std::cout << "Training time for this epoch: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
  	std::cout << "Training time for this epoch: " << ((end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec))/1000.0 << " ms." << std::endl;
  	std::cout << "Time for parallel part: " << k_time_1 << " ms." << std::endl;
  	std::cout << "Time for seq part: " << k_time_2 << " ms." << std::endl;

  	std::cout << "Training Loss (/sentence):    " << lossTrain/this->trainData.size() << std::endl << std::endl;
  	
	int sum_iter_counter = 0;
  	for (int ii = 0; ii < numThreads; ii ++)
  	{
  		std::cout << "thread id  = " << ii << ", count = " << iter_counter[ii]+1 << std::endl;
  		sum_iter_counter += iter_counter[ii]+1;
  	}
  	std::cout << "sum = " << sum_iter_counter << std::endl;

	std::cout << std::endl;
	std::cout << "time used for each part after an epoch" << std::endl;
	for (int i = 0; i < numTimers; i ++)
	{
		std::cout << allBatchTimer[i] << std::endl;
	}
	std::cout << std::endl;
	
	// here for record into file, this is for gantt figure
  	std::ofstream fout_start("time_rec_start.log");
  	std::ofstream fout_end("time_rec_end.log");
	std::ofstream fout_minibatch("time_each_minibatch.log");
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

	for (int i = 0; i < numTimers; i ++)
	{
		fout_minibatch << allBatchTimer[i] << std::endl;
	}
	fout_minibatch << std::endl;
	for (int i = 0; i < numTimers; i ++)
	{
		fout_minibatch << allBatchTimer[i]/miniBatch.size() << std::endl;
	}
  	fout_start.close();
  	fout_end.close();
	fout_minibatch.close();
	// Evaluation part of trainOpenMP() function  
  	gettimeofday(&start, 0); // used to record the time used for the evaluation part

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
	std::cout << std::endl << "The end of trainOpenMP() function" << std::endl << std::endl;
	return;
} // end of trainOpenMP_qiao

void EncDec::trainOpenMP_qiao_2(const Real learningRate, const int miniBatchSize, const int numThreads){
	static std::vector<EncDec::ThreadArg_2*> args;
	static std::vector<std::pair<int, int> > miniBatch;
	static EncDec::Grad grad;
	Real lossTrain = 0.0, perpDev = 0.0, denom = 0.0;
	Real gradNorm, lr = learningRate;
	const Real clipThreshold = 3.0;
	
	struct timeval start, end;
	std::cout << "size = " << sizeof(Real) << std::endl;
	
	// for recording time of different parts of train function
	int sizeTimers = 20;
	int numTimers = 13;
	static std::vector<EncDec::ThreadTimer*> timers;
	static std::vector<double> allBatchTimer; 

	struct timeval k_start, k_end;
	struct timeval k_start_1, k_end_1;
	struct timeval k_start_2, k_end_2;
	
	Real k_time_1 = 0.0;
	Real k_time_2 = 0.0;

    // for smart cache usage and get the size of tgt_voc and max_num_terms;
    int max_num_terms = 60; // for full data set max_num_terms would be 50
    int tgt_voc_size = this->targetVoc.tokenList.size(); 
	if (args.empty())
    {
		for (int i = 0; i < numThreads; ++i)
		{
			args.push_back(new EncDec::ThreadArg_2(*this));

      		for (int j = 0; j < 200; ++j){    //<??> why j < 200 here
      			args[i]->encState.push_back(new LSTM::State);
      			args[i]->encState[0]->h = this->zeros;
      			args[i]->encState[0]->c = this->zeros;
      			args[i]->decState.push_back(new LSTM::State);
      		}

            for (int j = 0; j < max_num_terms; j ++)
            {
                args[i]->delosBuffer.push_back(this->zeros);
                args[i]->delisBuffer.push_back(this->zeros);
                args[i]->delusBuffer.push_back(this->zeros);
                args[i]->delfsBuffer.push_back(this->zeros);
                args[i]->target_dist.push_back(VecD::Zero(tgt_voc_size));
            }

            for (int j =0; j < max_num_terms; j ++)
            {
                args[i]->deltaFeatureBuffer;
                args[i]->gradWeightBuffer;
            }
      	}

      	for (int i = 0, step = this->trainData.size()/miniBatchSize; i < step; ++i){
      		miniBatch.push_back(std::pair<int, int>(i*miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*miniBatchSize-1)));
      	}

      	grad.lstmSrcGrad = LSTM::Grad(this->enc);
      	grad.lstmTgtGrad = LSTM::Grad(this->dec);
      	grad.softmaxGrad = SoftMax::Grad(this->softmax);

	}

	// init for timers
	if (timers.empty())
	{
		std::cout << "initialize timers" << std::endl;
		for (int i = 0; i < numThreads; ++i)
		{
			timers.push_back(new EncDec::ThreadTimer(*this,sizeTimers));
			std::cout << "for thread " << i << ", timeRecorder size = " << timers[i]->timeRecorder.size() << std::endl;
		}
	}
	if (allBatchTimer.size() < sizeTimers)
	{
		for (int i = 0; i < allBatchTimer.size(); i ++)
		{
			allBatchTimer[i] - 0.0;
		}
		for (int i = allBatchTimer.size(); i < sizeTimers; i++)
		{
			allBatchTimer.push_back(0.0);
		}
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
  	k_time_1 = 0.0;
  	k_time_2 = 0.0;

  	// int max_batch_count = 4;
  	// int batch_count = 0;

	struct timeval start_temp, end_temp;

  	for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it){
		//  std::cout << "\r" << "Progress: " << ++count << "/" << miniBatch.size() << " mini batches" << std::flush;

		/* use for quick test to do sttop after several mini batch
		batch_count ++;
		std::cout << "batch count = " << batch_count << std::endl << std::endl;
		if (batch_count > max_batch_count)
		{
			break;
		}
		*/
		gettimeofday(&k_start_1, NULL);  

		// parallel part
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(args)
  		for (int i = it->first; i <= it->second; ++i){
  			int id = omp_get_thread_num();
  			Real loss;
  			iter_counter[id] ++;

			// record the time used for one sentence: start point
  			gettimeofday(&(time_rec_start[id]), NULL);

			// the main training function
  			this->train_qiao_4(this->trainData[i], args[id]->encState, args[id]->decState, args[id]->grad, loss, timers[id]->timeRecorder, args[id]->target_dist, args[id]->delosBuffer, args[id]->delisBuffer, args[id]->delusBuffer, args[id]->delfsBuffer);
  			// end of the main training function
			
			// record the time used for one sentence: end point an dsave the time
			gettimeofday(&(time_rec_end[id]), NULL);
  			sec_start[id][iter_counter[id]] = time_rec_start[id].tv_sec;
  			usec_start[id][iter_counter[id]] = time_rec_start[id].tv_usec;
  			sec_end[id][iter_counter[id]] = time_rec_end[id].tv_sec;
  			usec_end[id][iter_counter[id]] = time_rec_end[id].tv_usec;

  			args[id]->loss += loss;
  		}

  		gettimeofday(&k_end_1, NULL);
  		
		Real temp_time = ((k_end_1.tv_sec-k_start_1.tv_sec)*1000000+(k_end_1.tv_usec-k_start_1.tv_usec))/1000.0;
  		k_time_1 += temp_time;
		std::cout << "for one minibatch: " << temp_time << std::endl << std::endl;

		// ouput the recorded time
		for (int i = 0; i < numTimers; i ++)
		{
			double sum = 0.0;
			for (int j = 0; j < numThreads; j ++)
			{
				sum += timers[j]->timeRecorder[i];
			}
			allBatchTimer[i] += sum/numThreads;
			// std::cout << "average time used in part " << i << " = " << sum/numThreads << " ms" << std::endl;
		}

		for (int id = 0; id < numThreads; id ++)
		{
			timers[id]->init();
		}

		// serial part
		gettimeofday(&k_start_2, NULL); // record the time used for serial part
  		
		for (int id = 0; id < numThreads; ++id){
  			grad += args[id]->grad;
			args[id]->grad.init();
  			//args[id]->grad.init_qiao();
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

  		gettimeofday(&k_end_2, NULL); // record the time used for serial part
  		k_time_2 += ((k_end_2.tv_sec-k_start_2.tv_sec)*1000000+(k_end_2.tv_usec-k_start_2.tv_usec))/1000.0;
  	} // end of for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it)

  	std::cout << std::endl;
  	gettimeofday(&end, 0);
  	//std::cout << "Training time for this epoch: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
  	std::cout << "Training time for this epoch: " << ((end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec))/1000.0 << " ms." << std::endl;
  	std::cout << "Time for parallel part: " << k_time_1 << " ms." << std::endl;
  	std::cout << "Time for seq part: " << k_time_2 << " ms." << std::endl;

  	std::cout << "Training Loss (/sentence):    " << lossTrain/this->trainData.size() << std::endl;
  	
	int sum_iter_counter = 0;
  	for (int ii = 0; ii < numThreads; ii ++)
  	{
  		std::cout << "thread id  = " << ii << ", count = " << iter_counter[ii]+1 << std::endl;
  		sum_iter_counter += iter_counter[ii]+1;
  	}
  	std::cout << "sum = " << sum_iter_counter << std::endl;

	std::cout << std::endl;
	std::cout << "time used for each part after an epoch" << std::endl;
	for (int i = 0; i < numTimers; i ++)
	{
		std::cout << allBatchTimer[i] << std::endl;
	}
	std::cout << std::endl;
	
	// here for record into file, this is for gantt figure
  	std::ofstream fout_start("time_rec_start.log");
  	std::ofstream fout_end("time_rec_end.log");
	std::ofstream fout_minibatch("time_each_minibatch.log");
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
	for (int i = 0; i < numTimers; i ++)
	{
		fout_minibatch << allBatchTimer[i] << std::endl;
	}
	fout_minibatch << std::endl;
	for (int i = 0; i < numTimers; i ++)
	{
		fout_minibatch << allBatchTimer[i]/miniBatch.size() << std::endl;
	}
  	fout_start.close();
  	fout_end.close();
	fout_minibatch.close();
	// for a quick test

	// Evaluation part of trainOpenMP() function
  	gettimeofday(&start, 0); // used to record the time used for the evaluation part

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
	std::cout << std::endl << "The end of trainOpenMP() function" << std::endl << std::endl;

	return;
} // end of trainOpenMp_qiao_2

void EncDec::trainOpenMP_qiao_3(const Real learningRate, const int miniBatchSize, const int numThreads){
	// static std::vector<EncDec::ThreadArg_2*> args;
	static std::vector<EncDec::ThreadArg_3*> args;
	static std::vector<std::pair<int, int> > miniBatch;
	static EncDec::Grad grad;
	Real lossTrain = 0.0, perpDev = 0.0, denom = 0.0;
	Real gradNorm, lr = learningRate;
	const Real clipThreshold = 3.0;
	
	struct timeval start, end;
	std::cout << "size = " << sizeof(Real) << std::endl;
	
	// for recording time of different parts of train function
	int sizeTimers = 20;
	int numTimers = 13;
	static std::vector<EncDec::ThreadTimer*> timers;
	static std::vector<double> allBatchTimer; 

	struct timeval k_start, k_end;
	struct timeval k_start_1, k_end_1;
	struct timeval k_start_2, k_end_2;
	
	Real k_time_1 = 0.0;
	Real k_time_2 = 0.0;

    // for smart cache usage and get the size of tgt_voc and max_num_terms;
    int max_num_terms = 60; // for full data set max_num_terms would be 50
    int tgt_voc_size = this->targetVoc.tokenList.size(); 
	int hidden_dim = this->zeros.size();
	if (args.empty())
    {
		for (int i = 0; i < numThreads; ++i)
		{
			args.push_back(new EncDec::ThreadArg_3(*this));

      		for (int j = 0; j < 200; ++j){    //<??> why j < 200 here
      			args[i]->encState.push_back(new LSTM::State);
      			args[i]->encState[0]->h = this->zeros;
      			args[i]->encState[0]->c = this->zeros;
      			args[i]->decState.push_back(new LSTM::State);
      		}

            for (int j = 0; j < max_num_terms; j ++)
            {
                args[i]->target_dist.push_back(VecD::Zero(tgt_voc_size));
            }

			args[i]->delosBuffer = MatD(hidden_dim, max_num_terms);
			args[i]->delisBuffer = MatD(hidden_dim, max_num_terms);
			args[i]->delfsBuffer = MatD(hidden_dim, max_num_terms);
			args[i]->delusBuffer = MatD(hidden_dim, max_num_terms);

            for (int j =0; j < max_num_terms; j ++)
            {
                args[i]->deltaFeatureBuffer;
                args[i]->gradWeightBuffer;
            }
      	}

      	for (int i = 0, step = this->trainData.size()/miniBatchSize; i < step; ++i){
      		miniBatch.push_back(std::pair<int, int>(i*miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*miniBatchSize-1)));
      	}

      	grad.lstmSrcGrad = LSTM::Grad(this->enc);
      	grad.lstmTgtGrad = LSTM::Grad(this->dec);
      	grad.softmaxGrad = SoftMax::Grad(this->softmax);

	}

	// init for timers
	if (timers.empty())
	{
		std::cout << "initialize timers" << std::endl;
		for (int i = 0; i < numThreads; ++i)
		{
			timers.push_back(new EncDec::ThreadTimer(*this,sizeTimers));
			std::cout << "for thread " << i << ", timeRecorder size = " << timers[i]->timeRecorder.size() << std::endl;
		}
	}
	if (allBatchTimer.size() < sizeTimers)
	{
		for (int i = 0; i < allBatchTimer.size(); i ++)
		{
			allBatchTimer[i] - 0.0;
		}
		for (int i = allBatchTimer.size(); i < sizeTimers; i++)
		{
			allBatchTimer.push_back(0.0);
		}
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
  	k_time_1 = 0.0;
  	k_time_2 = 0.0;

  	// int max_batch_count = 4;
  	// int batch_count = 0;

	struct timeval start_temp, end_temp;

  	for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it){
		//  std::cout << "\r" << "Progress: " << ++count << "/" << miniBatch.size() << " mini batches" << std::flush;

		/* use for quick test to do sttop after several mini batch
		batch_count ++;
		std::cout << "batch count = " << batch_count << std::endl << std::endl;
		if (batch_count > max_batch_count)
		{
			break;
		}
		*/
		gettimeofday(&k_start_1, NULL);  

		// parallel part
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(args)
  		for (int i = it->first; i <= it->second; ++i){
  			int id = omp_get_thread_num();
  			Real loss;
  			iter_counter[id] ++;

			// record the time used for one sentence: start point
  			gettimeofday(&(time_rec_start[id]), NULL);

			// the main training function
  			this->train_qiao_5(this->trainData[i], args[id]->encState, args[id]->decState, args[id]->grad, loss, timers[id]->timeRecorder, args[id]->target_dist, args[id]->delosBuffer, args[id]->delisBuffer, args[id]->delusBuffer, args[id]->delfsBuffer);
  			// end of the main training function
			
			// record the time used for one sentence: end point an dsave the time
			gettimeofday(&(time_rec_end[id]), NULL);
  			sec_start[id][iter_counter[id]] = time_rec_start[id].tv_sec;
  			usec_start[id][iter_counter[id]] = time_rec_start[id].tv_usec;
  			sec_end[id][iter_counter[id]] = time_rec_end[id].tv_sec;
  			usec_end[id][iter_counter[id]] = time_rec_end[id].tv_usec;

  			args[id]->loss += loss;
  		}

  		gettimeofday(&k_end_1, NULL);
  		
		Real temp_time = ((k_end_1.tv_sec-k_start_1.tv_sec)*1000000+(k_end_1.tv_usec-k_start_1.tv_usec))/1000.0;
  		k_time_1 += temp_time;
		std::cout << "for one minibatch: " << temp_time << std::endl << std::endl;

		// ouput the recorded time
		for (int i = 0; i < numTimers; i ++)
		{
			double sum = 0.0;
			for (int j = 0; j < numThreads; j ++)
			{
				sum += timers[j]->timeRecorder[i];
			}
			allBatchTimer[i] += sum/numThreads;
			// std::cout << "average time used in part " << i << " = " << sum/numThreads << " ms" << std::endl;
		}

		for (int id = 0; id < numThreads; id ++)
		{
			timers[id]->init();
		}

		// serial part
		gettimeofday(&k_start_2, NULL); // record the time used for serial part
  		
		for (int id = 0; id < numThreads; ++id){
  			grad += args[id]->grad;
			args[id]->grad.init();
  			//args[id]->grad.init_qiao();
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

  		gettimeofday(&k_end_2, NULL); // record the time used for serial part
  		k_time_2 += ((k_end_2.tv_sec-k_start_2.tv_sec)*1000000+(k_end_2.tv_usec-k_start_2.tv_usec))/1000.0;
  	} // end of for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it)

  	std::cout << std::endl;
  	gettimeofday(&end, 0);
  	//std::cout << "Training time for this epoch: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
  	std::cout << "Training time for this epoch: " << ((end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec))/1000.0 << " ms." << std::endl;
  	std::cout << "Time for parallel part: " << k_time_1 << " ms." << std::endl;
  	std::cout << "Time for seq part: " << k_time_2 << " ms." << std::endl;

  	std::cout << "Training Loss (/sentence):    " << lossTrain/this->trainData.size() << std::endl;
  	
	int sum_iter_counter = 0;
  	for (int ii = 0; ii < numThreads; ii ++)
  	{
  		std::cout << "thread id  = " << ii << ", count = " << iter_counter[ii]+1 << std::endl;
  		sum_iter_counter += iter_counter[ii]+1;
  	}
  	std::cout << "sum = " << sum_iter_counter << std::endl;

	std::cout << std::endl;
	std::cout << "time used for each part after an epoch" << std::endl;
	for (int i = 0; i < numTimers; i ++)
	{
		std::cout << allBatchTimer[i] << std::endl;
	}
	std::cout << std::endl;
	
	// here for record into file, this is for gantt figure
  	std::ofstream fout_start("time_rec_start.log");
  	std::ofstream fout_end("time_rec_end.log");
	std::ofstream fout_minibatch("time_each_minibatch.log");
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
	for (int i = 0; i < numTimers; i ++)
	{
		fout_minibatch << allBatchTimer[i] << std::endl;
	}
	fout_minibatch << std::endl;
	for (int i = 0; i < numTimers; i ++)
	{
		fout_minibatch << allBatchTimer[i]/miniBatch.size() << std::endl;
	}
  	fout_start.close();
  	fout_end.close();
	fout_minibatch.close();
	// for a quick test

	// Evaluation part of trainOpenMP() function
  	gettimeofday(&start, 0); // used to record the time used for the evaluation part

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
	std::cout << std::endl << "The end of trainOpenMP() function" << std::endl << std::endl;

	return;
} // end of trainOpenMp_qiao_3

void EncDec::trainOpenMP_mf_1(const Real learningRate, const int miniBatchSize, const int numThreads){
	
	static std::vector<EncDec::ThreadArg*> args;
	static std::vector<std::pair<int, int> > miniBatch;
	static EncDec::Grad grad;
	Real lossTrain = 0.0, perpDev = 0.0, denom = 0.0;
	Real gradNorm, lr = learningRate;
	const Real clipThreshold = 3.0;
	
	struct timeval start, end;
	std::cout << "size = " << sizeof(Real) << std::endl;
	
	// for recording time of different parts of train function
	int sizeTimers = 20;
	int numTimers = 13;
	// static std::vector<EncDec::ThreadTimer*> timers;
	static std::vector<MemoryFootprint*> mfs;
	static std::vector<double> allBatchTimer; 

	struct timeval k_start, k_end;
	struct timeval k_start_1, k_end_1;
	struct timeval k_start_2, k_end_2;
	
	Real k_time_1 = 0.0;
	Real k_time_2 = 0.0;

    // for smart cache usage and get the size of tgt_voc and max_num_terms;
    int max_num_terms = 60; // for full data set max_num_terms would be 50
    int tgt_voc_size = this->targetVoc.tokenList.size(); 
	int hidden_dim = this->zeros.size();
	if (args.empty())
    {
		for (int i = 0; i < numThreads; ++i)
		{
			args.push_back(new EncDec::ThreadArg(*this));

      		for (int j = 0; j < 200; ++j){    //<??> why j < 200 here
      			args[i]->encState.push_back(new LSTM::State);
      			args[i]->encState[0]->h = this->zeros;
      			args[i]->encState[0]->c = this->zeros;
      			args[i]->decState.push_back(new LSTM::State);
      		}

      	}

      	for (int i = 0, step = this->trainData.size()/miniBatchSize; i < step; ++i){
      		miniBatch.push_back(std::pair<int, int>(i*miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*miniBatchSize-1)));
      	}

      	grad.lstmSrcGrad = LSTM::Grad(this->enc);
      	grad.lstmTgtGrad = LSTM::Grad(this->dec);
      	grad.softmaxGrad = SoftMax::Grad(this->softmax);

	}

	// init for timers
	/*
	if (timers.empty())
	{
		std::cout << "initialize timers" << std::endl;
		for (int i = 0; i < numThreads; ++i)
		{
			timers.push_back(new EncDec::ThreadTimer(*this,sizeTimers));
			std::cout << "for thread " << i << ", timeRecorder size = " << timers[i]->timeRecorder.size() << std::endl;
		}
	}
	if (allBatchTimer.size() < sizeTimers)
	{
		for (int i = 0; i < allBatchTimer.size(); i ++)
		{
			allBatchTimer[i] - 0.0;
		}
		for (int i = allBatchTimer.size(); i < sizeTimers; i++)
		{
			allBatchTimer.push_back(0.0);
		}
	}
	*/
	// init for memory footprint recoders
	if (mfs.empty())
	{
		std::cout << "initialize memory footprint recorders " << std::endl;
		for (int i = 0; i < numThreads; ++i)
		{
			mfs.push_back(new MemoryFootprint());
		}
	}

    std::cout << "number of miniBatch = " << miniBatch.size() << std::endl;
    std::cout << "first pair is " << miniBatch[0].first << " and " << miniBatch[0].second << std::endl;

	//this->rnd.shuffle(miniBatch);
  	this->rnd.shuffle(this->trainData); // <??> this part can be faster??

	// add time recorder here
  	/*
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
	*/
  	gettimeofday(&start, 0);

  	int count = 0;
  	k_time_1 = 0.0;
  	k_time_2 = 0.0;

  	int max_batch_count = 1;
  	int batch_count = 0;

	struct timeval start_temp, end_temp;

  	for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it){
		//  std::cout << "\r" << "Progress: " << ++count << "/" << miniBatch.size() << " mini batches" << std::flush;

		/* use for quick test to do sttop after several mini batch */
		batch_count ++;
		std::cout << "batch count = " << batch_count << std::endl << std::endl;
		if (batch_count > max_batch_count)
		{
			break;
		}
		gettimeofday(&k_start_1, NULL);  

		// parallel part
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(args)
  		for (int i = it->first; i <= it->second; ++i){
  			int id = omp_get_thread_num();
  			Real loss;
  			//iter_counter[id] ++;

			// record the time used for one sentence: start point
  			// gettimeofday(&(time_rec_start[id]), NULL);

			// the main training function
  			this->train_mf_1(this->trainData[i], args[id]->encState, args[id]->decState, args[id]->grad, loss, mfs[id]);
  			// end of the main training function
			
			// record the time used for one sentence: end point an dsave the time
			// gettimeofday(&(time_rec_end[id]), NULL);
  			/*
			sec_start[id][iter_counter[id]] = time_rec_start[id].tv_sec;
  			usec_start[id][iter_counter[id]] = time_rec_start[id].tv_usec;
  			sec_end[id][iter_counter[id]] = time_rec_end[id].tv_sec;
  			usec_end[id][iter_counter[id]] = time_rec_end[id].tv_usec;
			*/
  			args[id]->loss += loss;
  		}

  		gettimeofday(&k_end_1, NULL);
  		
		Real temp_time = ((k_end_1.tv_sec-k_start_1.tv_sec)*1000000+(k_end_1.tv_usec-k_start_1.tv_usec))/1000.0;
  		k_time_1 += temp_time;
		std::cout << "for one minibatch: " << temp_time << std::endl << std::endl;

		// ouput the recorded time
		/*
		for (int i = 0; i < numTimers; i ++)
		{
			double sum = 0.0;
			for (int j = 0; j < numThreads; j ++)
			{
				sum += timers[j]->timeRecorder[i];
			}
			allBatchTimer[i] += sum/numThreads;
			// std::cout << "average time used in part " << i << " = " << sum/numThreads << " ms" << std::endl;
		}

		for (int id = 0; id < numThreads; id ++)
		{
			timers[id]->init();
		}
		*/

		// serial part
		gettimeofday(&k_start_2, NULL); // record the time used for serial part
  		
		for (int id = 0; id < numThreads; ++id){
  			grad += args[id]->grad;
			args[id]->grad.init();
  			//args[id]->grad.init_qiao();
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

  		gettimeofday(&k_end_2, NULL); // record the time used for serial part
  		k_time_2 += ((k_end_2.tv_sec-k_start_2.tv_sec)*1000000+(k_end_2.tv_usec-k_start_2.tv_usec))/1000.0;
  	} // end of for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it)

  	std::cout << std::endl;
  	gettimeofday(&end, 0);
  	//std::cout << "Training time for this epoch: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
  	std::cout << "Training time for this epoch: " << ((end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec))/1000.0 << " ms." << std::endl;
  	std::cout << "Time for parallel part: " << k_time_1 << " ms." << std::endl;
  	std::cout << "Time for seq part: " << k_time_2 << " ms." << std::endl;

  	std::cout << "Training Loss (/sentence):    " << lossTrain/this->trainData.size() << std::endl;
	/*	
	int sum_iter_counter = 0;
  	for (int ii = 0; ii < numThreads; ii ++)
  	{
  		std::cout << "thread id  = " << ii << ", count = " << iter_counter[ii]+1 << std::endl;
  		sum_iter_counter += iter_counter[ii]+1;
  	}
  	std::cout << "sum = " << sum_iter_counter << std::endl;

	std::cout << std::endl;
	std::cout << "time used for each part after an epoch" << std::endl;
	for (int i = 0; i < numTimers; i ++)
	{
		std::cout << allBatchTimer[i] << std::endl;
	}
	std::cout << std::endl;
	
	// here for record into file, this is for gantt figure
  	std::ofstream fout_start("time_rec_start.log");
  	std::ofstream fout_end("time_rec_end.log");
	std::ofstream fout_minibatch("time_each_minibatch.log");
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
	for (int i = 0; i < numTimers; i ++)
	{
		fout_minibatch << allBatchTimer[i] << std::endl;
	}
	fout_minibatch << std::endl;
	for (int i = 0; i < numTimers; i ++)
	{
		fout_minibatch << allBatchTimer[i]/miniBatch.size() << std::endl;
	}
  	fout_start.close();
  	fout_end.close();
	fout_minibatch.close();
	*/

	// for a quick test
	return;
	// Evaluation part of trainOpenMP() function
  	gettimeofday(&start, 0); // used to record the time used for the evaluation part

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
	std::cout << std::endl << "The end of trainOpenMP() function" << std::endl << std::endl;

	return;
} // end of trainOpenMp_mf_1

void EncDec::trainOpenMP_mf_2(const Real learningRate, const int miniBatchSize, const int numThreads)
{	
	static std::vector<EncDec::ThreadArg*> args;
	static std::vector<std::pair<int, int> > miniBatch;
	static EncDec::Grad grad;
	Real lossTrain = 0.0, perpDev = 0.0, denom = 0.0;
	Real gradNorm, lr = learningRate;
	const Real clipThreshold = 3.0;
	
	struct timeval start, end;
	std::cout << "size = " << sizeof(Real) << std::endl;
	
	// for recording time of different parts of train function
	int sizeTimers = 20;
	int numTimers = 13;
	// static std::vector<EncDec::ThreadTimer*> timers;
	static std::vector<MemoryFootprint*> mfs;
	static std::vector<double> allBatchTimer; 

	struct timeval k_start, k_end;
	struct timeval k_start_1, k_end_1;
	struct timeval k_start_2, k_end_2;
	
	Real k_time_1 = 0.0;
	Real k_time_2 = 0.0;

    // for smart cache usage and get the size of tgt_voc and max_num_terms;
    int max_num_terms = 60; // for full data set max_num_terms would be 50
    int tgt_voc_size = this->targetVoc.tokenList.size(); 
	int hidden_dim = this->zeros.size();
	if (args.empty())
    {
		for (int i = 0; i < numThreads; ++i)
		{
			args.push_back(new EncDec::ThreadArg(*this));

      		for (int j = 0; j < 200; ++j){    //<??> why j < 200 here
      			args[i]->encState.push_back(new LSTM::State);
      			args[i]->encState[0]->h = this->zeros;
      			args[i]->encState[0]->c = this->zeros;
      			args[i]->decState.push_back(new LSTM::State);
      		}

      	}

      	for (int i = 0, step = this->trainData.size()/miniBatchSize; i < step; ++i){
      		miniBatch.push_back(std::pair<int, int>(i*miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*miniBatchSize-1)));
      	}

      	grad.lstmSrcGrad = LSTM::Grad(this->enc);
      	grad.lstmTgtGrad = LSTM::Grad(this->dec);
      	grad.softmaxGrad = SoftMax::Grad(this->softmax);

	}

	// init for timers
	/*
	if (timers.empty())
	{
		std::cout << "initialize timers" << std::endl;
		for (int i = 0; i < numThreads; ++i)
		{
			timers.push_back(new EncDec::ThreadTimer(*this,sizeTimers));
			std::cout << "for thread " << i << ", timeRecorder size = " << timers[i]->timeRecorder.size() << std::endl;
		}
	}
	if (allBatchTimer.size() < sizeTimers)
	{
		for (int i = 0; i < allBatchTimer.size(); i ++)
		{
			allBatchTimer[i] - 0.0;
		}
		for (int i = allBatchTimer.size(); i < sizeTimers; i++)
		{
			allBatchTimer.push_back(0.0);
		}
	}
	*/
	// init for memory footprint recoders
	if (mfs.empty())
	{
		std::cout << "initialize memory footprint recorders " << std::endl;
		for (int i = 0; i < numThreads; ++i)
		{
			mfs.push_back(new MemoryFootprint());
		}
	}

    std::cout << "number of miniBatch = " << miniBatch.size() << std::endl;
    std::cout << "first pair is " << miniBatch[0].first << " and " << miniBatch[0].second << std::endl;

	//this->rnd.shuffle(miniBatch);
  	this->rnd.shuffle(this->trainData); // <??> this part can be faster??

	// add time recorder here
  	/*
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
	*/
  	gettimeofday(&start, 0);

  	int count = 0;
  	k_time_1 = 0.0;
  	k_time_2 = 0.0;

  	int max_batch_count = 1;
  	int batch_count = 0;

	struct timeval start_temp, end_temp;

  	for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it){
		//  std::cout << "\r" << "Progress: " << ++count << "/" << miniBatch.size() << " mini batches" << std::flush;

		/* use for quick test to do sttop after several mini batch */
		batch_count ++;
		std::cout << "batch count = " << batch_count << std::endl << std::endl;
		if (batch_count > max_batch_count)
		{
			break;
		}
		gettimeofday(&k_start_1, NULL);  

		// parallel part
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(args)
  		for (int i = it->first; i <= it->second; ++i){
  			int id = omp_get_thread_num();
  			Real loss;
  			//iter_counter[id] ++;

			// record the time used for one sentence: start point
  			// gettimeofday(&(time_rec_start[id]), NULL);

			// the main training function
  			this->train_mf_2(this->trainData[i], args[id]->encState, args[id]->decState, args[id]->grad, loss, mfs[id]);
  			// end of the main training function
			
			// record the time used for one sentence: end point an dsave the time
			// gettimeofday(&(time_rec_end[id]), NULL);
  			/*
			sec_start[id][iter_counter[id]] = time_rec_start[id].tv_sec;
  			usec_start[id][iter_counter[id]] = time_rec_start[id].tv_usec;
  			sec_end[id][iter_counter[id]] = time_rec_end[id].tv_sec;
  			usec_end[id][iter_counter[id]] = time_rec_end[id].tv_usec;
			*/
  			args[id]->loss += loss;
  		}

  		gettimeofday(&k_end_1, NULL);
  		
		Real temp_time = ((k_end_1.tv_sec-k_start_1.tv_sec)*1000000+(k_end_1.tv_usec-k_start_1.tv_usec))/1000.0;
  		k_time_1 += temp_time;
		std::cout << "for one minibatch: " << temp_time << std::endl << std::endl;

		// ouput the recorded time
		/*
		for (int i = 0; i < numTimers; i ++)
		{
			double sum = 0.0;
			for (int j = 0; j < numThreads; j ++)
			{
				sum += timers[j]->timeRecorder[i];
			}
			allBatchTimer[i] += sum/numThreads;
			// std::cout << "average time used in part " << i << " = " << sum/numThreads << " ms" << std::endl;
		}

		for (int id = 0; id < numThreads; id ++)
		{
			timers[id]->init();
		}
		*/

		// serial part
		gettimeofday(&k_start_2, NULL); // record the time used for serial part
  		
		for (int id = 0; id < numThreads; ++id){
  			grad += args[id]->grad;
			args[id]->grad.init();
  			//args[id]->grad.init_qiao();
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

  		gettimeofday(&k_end_2, NULL); // record the time used for serial part
  		k_time_2 += ((k_end_2.tv_sec-k_start_2.tv_sec)*1000000+(k_end_2.tv_usec-k_start_2.tv_usec))/1000.0;
  	} // end of for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it)

  	std::cout << std::endl;
  	gettimeofday(&end, 0);
  	//std::cout << "Training time for this epoch: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
  	std::cout << "Training time for this epoch: " << ((end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec))/1000.0 << " ms." << std::endl;
  	std::cout << "Time for parallel part: " << k_time_1 << " ms." << std::endl;
  	std::cout << "Time for seq part: " << k_time_2 << " ms." << std::endl;

  	std::cout << "Training Loss (/sentence):    " << lossTrain/this->trainData.size() << std::endl;
	
	for (int i = 0; i < numThreads; i ++)
	{
		std::string file_name = std::string("memory_footprint_") + std::to_string(i)  + std::string(".log");
		mfs[i]->record(file_name, start, i);
	}
	
	/*	
	int sum_iter_counter = 0;
  	for (int ii = 0; ii < numThreads; ii ++)
  	{
  		std::cout << "thread id  = " << ii << ", count = " << iter_counter[ii]+1 << std::endl;
  		sum_iter_counter += iter_counter[ii]+1;
  	}
  	std::cout << "sum = " << sum_iter_counter << std::endl;

	std::cout << std::endl;
	std::cout << "time used for each part after an epoch" << std::endl;
	for (int i = 0; i < numTimers; i ++)
	{
		std::cout << allBatchTimer[i] << std::endl;
	}
	std::cout << std::endl;
	
	// here for record into file, this is for gantt figure
  	std::ofstream fout_start("time_rec_start.log");
  	std::ofstream fout_end("time_rec_end.log");
	std::ofstream fout_minibatch("time_each_minibatch.log");
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
	for (int i = 0; i < numTimers; i ++)
	{
		fout_minibatch << allBatchTimer[i] << std::endl;
	}
	fout_minibatch << std::endl;
	for (int i = 0; i < numTimers; i ++)
	{
		fout_minibatch << allBatchTimer[i]/miniBatch.size() << std::endl;
	}
  	fout_start.close();
  	fout_end.close();
	fout_minibatch.close();
	*/

	// for a quick test
	return;
	// Evaluation part of trainOpenMP() function
  	gettimeofday(&start, 0); // used to record the time used for the evaluation part

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
	std::cout << std::endl << "The end of trainOpenMP() function" << std::endl << std::endl;

	return;
} // end of trainOpenMp_mf_2
void EncDec::trainOpenMP_new_v1(const Real learningRate, const int miniBatchSize, const int numThreads){
	
	static std::vector<EncDec::ThreadArg_2*> args;
	static std::vector<std::pair<int, int> > miniBatch;
	static EncDec::Grad grad;
	Real lossTrain = 0.0, perpDev = 0.0, denom = 0.0;
	Real gradNorm, lr = learningRate;
	const Real clipThreshold = 3.0;
	
	struct timeval start, end;
	std::cout << "size = " << sizeof(Real) << std::endl;
	
	struct timeval k_start, k_end;
	struct timeval k_start_1, k_end_1;
	struct timeval k_start_2, k_end_2;
	
	Real k_time_1 = 0.0;
	Real k_time_2 = 0.0;

    // for smart cache usage and get the size of tgt_voc and max_num_terms;
    int max_num_terms = 60; // for full data set max_num_terms would be 50
    int tgt_voc_size = this->targetVoc.tokenList.size(); 
	if (args.empty())
    {
		for (int i = 0; i < numThreads; ++i)
		{
			args.push_back(new EncDec::ThreadArg_2(*this));

      		for (int j = 0; j < 200; ++j){    //<??> why j < 200 here
      			args[i]->encState.push_back(new LSTM::State);
      			args[i]->encState[0]->h = this->zeros;
      			args[i]->encState[0]->c = this->zeros;
      			args[i]->decState.push_back(new LSTM::State);
      		}

            for (int j = 0; j < max_num_terms; j ++)
            {
                args[i]->delosBuffer.push_back(this->zeros);
                args[i]->delisBuffer.push_back(this->zeros);
                args[i]->delusBuffer.push_back(this->zeros);
                args[i]->delfsBuffer.push_back(this->zeros);
                args[i]->target_dist.push_back(VecD::Zero(tgt_voc_size));
            }

            for (int j =0; j < max_num_terms; j ++)
            {
                args[i]->deltaFeatureBuffer;
                args[i]->gradWeightBuffer;
            }
      	}

      	for (int i = 0, step = this->trainData.size()/miniBatchSize; i < step; ++i){
      		miniBatch.push_back(std::pair<int, int>(i*miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*miniBatchSize-1)));
      	}

      	grad.lstmSrcGrad = LSTM::Grad(this->enc);
      	grad.lstmTgtGrad = LSTM::Grad(this->dec);
      	grad.softmaxGrad = SoftMax::Grad(this->softmax);

	}

    std::cout << "number of miniBatch = " << miniBatch.size() << std::endl;
    std::cout << "first pair is " << miniBatch[0].first << " and " << miniBatch[0].second << std::endl;

	//this->rnd.shuffle(miniBatch);
  	this->rnd.shuffle(this->trainData); // <??> this part can be faster??
  	int iter_counter[numThreads];

  	for (int ii = 0; ii < numThreads; ii ++)
  	{
  		iter_counter[ii] = -1;
  	}

  	gettimeofday(&start, 0);

  	int count = 0;
  	k_time_1 = 0.0;
  	k_time_2 = 0.0;

  	// int max_batch_count = 4;
  	// int batch_count = 0;

  	for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it)
	{
		gettimeofday(&k_start_1, NULL);  

		// parallel part
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(args)
  		for (int i = it->first; i <= it->second; ++i)
		{
  			int id = omp_get_thread_num();
  			Real loss;
  			iter_counter[id] ++;

			// the main training function
			this->train_new_v1(this->trainData[i], args[id]->encState, args[id]->decState, args[id]->grad, loss, args[id]->target_dist, args[id]->delosBuffer, args[id]->delisBuffer, args[id]->delusBuffer, args[id]->delfsBuffer);
			// end of the main training function

  			args[id]->loss += loss;
  		}

  		gettimeofday(&k_end_1, NULL);
  		
		Real temp_time = ((k_end_1.tv_sec-k_start_1.tv_sec)*1000000+(k_end_1.tv_usec-k_start_1.tv_usec))/1000.0;
  		k_time_1 += temp_time;
		std::cout << "for one minibatch: " << temp_time << std::endl << std::endl;
		// serial part
		gettimeofday(&k_start_2, NULL); // record the time used for serial part
  		
		for (int id = 0; id < numThreads; ++id){
  			grad += args[id]->grad;
			args[id]->grad.init();
  			//args[id]->grad.init_qiao();
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

  		gettimeofday(&k_end_2, NULL); // record the time used for serial part
  		k_time_2 += ((k_end_2.tv_sec-k_start_2.tv_sec)*1000000+(k_end_2.tv_usec-k_start_2.tv_usec))/1000.0;
  	} // end of for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it)

  	std::cout << std::endl;
  	gettimeofday(&end, 0);
  	//std::cout << "Training time for this epoch: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
  	std::cout << "Training time for this epoch: " << ((end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec))/1000.0 << " ms." << std::endl;
  	std::cout << "Time for parallel part: " << k_time_1 << " ms." << std::endl;
  	std::cout << "Time for seq part: " << k_time_2 << " ms." << std::endl;

  	std::cout << "Training Loss (/sentence):    " << lossTrain/this->trainData.size() << std::endl;
  	
	int sum_iter_counter = 0;
  	for (int ii = 0; ii < numThreads; ii ++)
  	{
  		std::cout << "thread id  = " << ii << ", count = " << iter_counter[ii]+1 << std::endl;
  		sum_iter_counter += iter_counter[ii]+1;
  	}
  	std::cout << "sum = " << sum_iter_counter << std::endl;

	std::cout << std::endl;

	// Evaluation part of trainOpenMP() function
  	gettimeofday(&start, 0); // used to record the time used for the evaluation part

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
	std::cout << std::endl << "The end of trainOpenMP() function" << std::endl << std::endl;

	return;
} // end of trainOpenMp_new_v1

void EncDec::trainOpenMP_new_v2(const Real learningRate, const int miniBatchSize, const int numThreads){
	
	static std::vector<EncDec::ThreadArg_3*> args;
	static std::vector<std::pair<int, int> > miniBatch;
	static EncDec::Grad grad;
	Real lossTrain = 0.0, perpDev = 0.0, denom = 0.0;
	Real gradNorm, lr = learningRate;
	const Real clipThreshold = 3.0;
	
	struct timeval start, end;
	std::cout << "size = " << sizeof(Real) << std::endl;
	
	struct timeval k_start, k_end;
	struct timeval k_start_1, k_end_1;
	struct timeval k_start_2, k_end_2;
	
	Real k_time_1 = 0.0;
	Real k_time_2 = 0.0;

    // for smart cache usage and get the size of tgt_voc and max_num_terms;
    int max_num_terms = 60; // for full data set max_num_terms would be 50
    int tgt_voc_size = this->targetVoc.tokenList.size(); 
	int hidden_dim = this->zeros.size();
	if (args.empty())
    {
		for (int i = 0; i < numThreads; ++i)
		{
			args.push_back(new EncDec::ThreadArg_3(*this));

      		for (int j = 0; j < 200; ++j){    //<??> why j < 200 here
      			args[i]->encState.push_back(new LSTM::State);
      			args[i]->encState[0]->h = this->zeros;
      			args[i]->encState[0]->c = this->zeros;
      			args[i]->decState.push_back(new LSTM::State);
      		}

            for (int j = 0; j < max_num_terms; j ++)
            {
                args[i]->target_dist.push_back(VecD::Zero(tgt_voc_size));
            }
			args[i]->delosBuffer = MatD(hidden_dim, max_num_terms);
			args[i]->delisBuffer = MatD(hidden_dim, max_num_terms);
			args[i]->delfsBuffer = MatD(hidden_dim, max_num_terms);
			args[i]->delusBuffer = MatD(hidden_dim, max_num_terms);


            for (int j =0; j < max_num_terms; j ++)
            {
                args[i]->deltaFeatureBuffer;
                args[i]->gradWeightBuffer;
            }
      	}

      	for (int i = 0, step = this->trainData.size()/miniBatchSize; i < step; ++i){
      		miniBatch.push_back(std::pair<int, int>(i*miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*miniBatchSize-1)));
      	}

      	grad.lstmSrcGrad = LSTM::Grad(this->enc);
      	grad.lstmTgtGrad = LSTM::Grad(this->dec);
      	grad.softmaxGrad = SoftMax::Grad(this->softmax);

	}

    std::cout << "number of miniBatch = " << miniBatch.size() << std::endl;
    std::cout << "first pair is " << miniBatch[0].first << " and " << miniBatch[0].second << std::endl;

	//this->rnd.shuffle(miniBatch);
  	this->rnd.shuffle(this->trainData); // <??> this part can be faster??
  	int iter_counter[numThreads];

  	for (int ii = 0; ii < numThreads; ii ++)
  	{
  		iter_counter[ii] = -1;
  	}

  	gettimeofday(&start, 0);

  	int count = 0;
  	k_time_1 = 0.0;
  	k_time_2 = 0.0;

  	// int max_batch_count = 4;
  	// int batch_count = 0;

  	for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it)
	{
		gettimeofday(&k_start_1, NULL);  

		// parallel part
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(args)
  		for (int i = it->first; i <= it->second; ++i)
		{
  			int id = omp_get_thread_num();
  			Real loss;
  			iter_counter[id] ++;

			// the main training function
			this->train_new_v2(this->trainData[i], args[id]->encState, args[id]->decState, args[id]->grad, loss, args[id]->target_dist, args[id]->delosBuffer, args[id]->delisBuffer, args[id]->delusBuffer, args[id]->delfsBuffer);
			// end of the main training function

  			args[id]->loss += loss;
  		}

  		gettimeofday(&k_end_1, NULL);
  		
		Real temp_time = ((k_end_1.tv_sec-k_start_1.tv_sec)*1000000+(k_end_1.tv_usec-k_start_1.tv_usec))/1000.0;
  		k_time_1 += temp_time;
		std::cout << "for one minibatch: " << temp_time << std::endl << std::endl;
		// serial part
		gettimeofday(&k_start_2, NULL); // record the time used for serial part
  		
		for (int id = 0; id < numThreads; ++id){
  			grad += args[id]->grad;
			args[id]->grad.init();
  			//args[id]->grad.init_qiao();
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

  		gettimeofday(&k_end_2, NULL); // record the time used for serial part
  		k_time_2 += ((k_end_2.tv_sec-k_start_2.tv_sec)*1000000+(k_end_2.tv_usec-k_start_2.tv_usec))/1000.0;
  	} // end of for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it)

  	std::cout << std::endl;
  	gettimeofday(&end, 0);
  	//std::cout << "Training time for this epoch: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
  	std::cout << "Training time for this epoch: " << ((end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec))/1000.0 << " ms." << std::endl;
  	std::cout << "Time for parallel part: " << k_time_1 << " ms." << std::endl;
  	std::cout << "Time for seq part: " << k_time_2 << " ms." << std::endl;

  	std::cout << "Training Loss (/sentence):    " << lossTrain/this->trainData.size() << std::endl;
  	
	int sum_iter_counter = 0;
  	for (int ii = 0; ii < numThreads; ii ++)
  	{
  		std::cout << "thread id  = " << ii << ", count = " << iter_counter[ii]+1 << std::endl;
  		sum_iter_counter += iter_counter[ii]+1;
  	}
  	std::cout << "sum = " << sum_iter_counter << std::endl;

	std::cout << std::endl;

	// Evaluation part of trainOpenMP() function
  	gettimeofday(&start, 0); // used to record the time used for the evaluation part

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
	std::cout << std::endl << "The end of trainOpenMP() function" << std::endl << std::endl;

	return;
} // end of trainOpenMp_new_v2

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
	const int inputDim = 512;
	const int hiddenDim = 512;
	const int miniBatchSize = 128; // <!!> modify this parameter to test
	const int numThread = 16;
	EncDec encdec(sourceVoc, targetVoc, trainData, devData, inputDim, hiddenDim);
	auto test = trainData[0]->src;

	std::cout << "# of training data:    " << trainData.size() << std::endl;
	std::cout << "# of development data: " << devData.size() << std::endl;
	std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
	std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;
  
  int break_point;

  // use one epoch for quick test
  for (int i = 0; i < 1; ++i){
  	if (i+1 >= 6){
      //learningRate *= 0.5;
  	}

  	std::cout << "\nEpoch " << i+1 << std::endl;
  	encdec.trainOpenMP(learningRate, miniBatchSize, numThread);
    
	std::cout << "### Greedy ###" << std::endl;
    encdec.translate(test, 1, 100, 1);
    std::cout << "### Beam search ###" << std::endl;
    encdec.translate(test, 20, 100, 5);

  	std::ostringstream oss;
  	oss << "model." << i+1 << "itr.bin";
	encdec.save(oss.str());
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
	int max_epoch = 1;
	for (int i = 0; i < max_epoch; ++i){
		if (i+1 >= 6){
		//learningRate *= 0.5;
		}

		std::cout << "\nEpoch " << i+1 << std::endl;
		//encdec.trainOpenMP(learningRate, miniBatchSize, numThread);
		encdec.trainOpenMP_qiao(learningRate, miniBatchSize, numThread);
		std::cout << "### Greedy ###" << std::endl;
		encdec.translate(test, 1, 100, 1);
		std::cout << "### Beam search ###" << std::endl;
		encdec.translate(test, 20, 100, 5);

		// std::ostringstream oss;
		// oss << "model." << i+1 << "itr.bin";
		// encdec.save(oss.str());
	}


	// for quick test, don't do the translation part
	std::cout << "don't do the translation part" << std::endl;
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

void EncDec::demo_qiao_2(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads)
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

	// prepare training data
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

    // prepare development data
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
	
    // establish the nerual network
    EncDec encdec(sourceVoc, targetVoc, trainData, devData, inputDim, hiddenDim);
	auto test = trainData[0]->src;

	std::cout << "# of training data:    " << trainData.size() << std::endl;
	std::cout << "# of development data: " << devData.size() << std::endl;
	std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
	std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;

	int max_epoch = 1;
	for (int i = 0; i < max_epoch; ++i){
		if (i+1 >= 6){
		//learningRate *= 0.5;
		}

		std::cout << "\nEpoch " << i+1 << std::endl;
		//encdec.trainOpenMP(learningRate, miniBatchSize, numThread);
		//encdec.trainOpenMP_qiao(learningRate, miniBatchSize, numThread);
		encdec.trainOpenMP_qiao_2(learningRate, miniBatchSize, numThread);
        std::cout << "### Greedy ###" << std::endl;
		encdec.translate(test, 1, 100, 1);
		std::cout << "### Beam search ###" << std::endl;
		encdec.translate(test, 20, 100, 5);

		std::ostringstream oss;
		oss << "model." << i+1 << "itr.bin";
		encdec.save(oss.str());
	}

	// for quick test and dont't do the translation part
	std::cout << "don't do the translation part" << std::endl;
	return;

    // just care about the training process

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
} // end of demo_qiao_2()

void EncDec::demo_qiao_3(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads)
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

	// prepare training data
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

    // prepare development data
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
	
    // establish the nerual network
    EncDec encdec(sourceVoc, targetVoc, trainData, devData, inputDim, hiddenDim);
	auto test = trainData[0]->src;

	std::cout << "# of training data:    " << trainData.size() << std::endl;
	std::cout << "# of development data: " << devData.size() << std::endl;
	std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
	std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;

	int max_epoch = 1;
	for (int i = 0; i < max_epoch; ++i){
		if (i+1 >= 6){
		//learningRate *= 0.5;
		}

		std::cout << "\nEpoch " << i+1 << std::endl;
		//encdec.trainOpenMP(learningRate, miniBatchSize, numThread);
		//encdec.trainOpenMP_qiao(learningRate, miniBatchSize, numThread);
		encdec.trainOpenMP_qiao_3(learningRate, miniBatchSize, numThread);
        std::cout << "### Greedy ###" << std::endl;
		encdec.translate(test, 1, 100, 1);
		std::cout << "### Beam search ###" << std::endl;
		encdec.translate(test, 20, 100, 5);

		std::ostringstream oss;
		oss << "model." << i+1 << "itr.bin";
		encdec.save(oss.str());
	}

	// for quick test and dont't do the translation part
	std::cout << "don't do the translation part" << std::endl;
	return;

    // just care about the training process

	encdec.load("model.1itr.bin");

	struct timeval start, end;

    //translation
    std::vector<std::vector<int>> output(encdec.devData.size());
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
} // end of demo_qiao_3()

void EncDec::demo_mf_1(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads)
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
	int max_epoch = 1;
	for (int i = 0; i < max_epoch; ++i){

		std::cout << "\nEpoch " << i+1 << std::endl;
		//encdec.trainOpenMP(learningRate, miniBatchSize, numThread);
		encdec.trainOpenMP_mf_1(learningRate, miniBatchSize, numThread);
		std::cout << "### Greedy ###" << std::endl;
		encdec.translate(test, 1, 100, 1);
		std::cout << "### Beam search ###" << std::endl;
		encdec.translate(test, 20, 100, 5);
	}


	// for quick test, don't do the translation part
	std::cout << "don't do the translation part" << std::endl;
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
} // end of demo_mf_1()

void EncDec::demo_mf_2(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads)
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
	int max_epoch = 1;
	for (int i = 0; i < max_epoch; ++i){

		std::cout << "\nEpoch " << i+1 << std::endl;
		//encdec.trainOpenMP(learningRate, miniBatchSize, numThread);
		encdec.trainOpenMP_mf_2(learningRate, miniBatchSize, numThread);
		std::cout << "### Greedy ###" << std::endl;
		encdec.translate(test, 1, 100, 1);
		std::cout << "### Beam search ###" << std::endl;
		encdec.translate(test, 20, 100, 5);
	}


	// for quick test, don't do the translation part
	std::cout << "don't do the translation part" << std::endl;
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
} // end of demo_mf_2()

void EncDec::demo_new_v1(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads)
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

	// prepare training data
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

    // prepare development data
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
	const int numThreads = argsNumThreads;
	
    // establish the nerual network
    EncDec encdec(sourceVoc, targetVoc, trainData, devData, inputDim, hiddenDim);
	auto test = trainData[0]->src;

	std::cout << "# of training data:    " << trainData.size() << std::endl;
	std::cout << "# of development data: " << devData.size() << std::endl;
	std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
	std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;

	int max_epoch = 1;
	for (int i = 0; i < max_epoch; ++i){
		if (i+1 >= 6){
		//learningRate *= 0.5;
		}

		std::cout << "\nEpoch " << i+1 << std::endl;
	
		encdec.trainOpenMP_new_v1(learningRate, miniBatchSize, numThreads);
        
		std::cout << "### Greedy ###" << std::endl;
		encdec.translate(test, 1, 100, 1);
		std::cout << "### Beam search ###" << std::endl;
		encdec.translate(test, 20, 100, 5);

		std::ostringstream oss;
		oss << "model." << i+1 << "itr.bin";
		encdec.save(oss.str());
	}

	// for quick test and dont't do the translation part
	std::cout << "don't do the translation part" << std::endl;
	return;

    // just care about the training process

	encdec.load("model.1itr.bin");

	struct timeval start, end;

    //translation
    std::vector<std::vector<int> > output(encdec.devData.size());
    gettimeofday(&start, 0);
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(output, encdec)
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
} // end of demo_new_v1()

void EncDec::demo_new_v2(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev, const Real argsLearningRate, const int argsInputDim, const int argsHiddenDim, const int argsMiniBatchSize, const int argsNumThreads)
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

	// prepare training data
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

    // prepare development data
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
	const int numThreads = argsNumThreads;
	
    // establish the nerual network
    EncDec encdec(sourceVoc, targetVoc, trainData, devData, inputDim, hiddenDim);
	auto test = trainData[0]->src;

	std::cout << "# of training data:    " << trainData.size() << std::endl;
	std::cout << "# of development data: " << devData.size() << std::endl;
	std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
	std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;

	int max_epoch = 1;
	for (int i = 0; i < max_epoch; ++i){
		if (i+1 >= 6){
		//learningRate *= 0.5;
		}

		std::cout << "\nEpoch " << i+1 << std::endl;
	
		encdec.trainOpenMP_new_v2(learningRate, miniBatchSize, numThreads);
        
		std::cout << "### Greedy ###" << std::endl;
		encdec.translate(test, 1, 100, 1);
		std::cout << "### Beam search ###" << std::endl;
		encdec.translate(test, 20, 100, 5);

		std::ostringstream oss;
		oss << "model." << i+1 << "itr.bin";
		encdec.save(oss.str());
	}

	// for quick test and dont't do the translation part
	std::cout << "don't do the translation part" << std::endl;
	return;

    // just care about the training process

	encdec.load("model.1itr.bin");

	struct timeval start, end;

    //translation
    std::vector<std::vector<int> > output(encdec.devData.size());
    gettimeofday(&start, 0);
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(output, encdec)
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
} // end of demo_new_v2()

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
