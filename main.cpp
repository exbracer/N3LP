#include "EncDec.hpp"

#include <iostream>
#include <string>
class ArgsTable
{
public:
	std::string dataset_dir;
	std::string train_src;
	std::string train_tgt;
	std::string test_src;
	std::string test_tgt;

	Real learningRate;
	int inputDim;
	int hiddenDim;
	int miniBatchSize;
	int numThreads;
	int version;
	std::string information;

	ArgsTable()
	{
		information += "Please execute command like: \n";
		information += "[executable file] -d [dataset directory] \n";
		information += "Options: \n";
		information += "  -r [learning rate] \n";
		information += "  -i [input dimension] \n";
		information += "  -h [hidden dimension] \n";
		information += "  -m [mini batch size] \n";
		information += "  -n [number of threads] \n\n";
		
		dataset_dir = "./";
		learningRate = 0.5;
		inputDim = 50;
		hiddenDim = 50;
		miniBatchSize = 1;
		numThreads = 1;
		version = 0;
	}
	int readInArgs(int argc, char ** argv);
private:
	void genDatasetPath(std::string arg);
};

int ArgsTable::readInArgs(int argc, char ** argv)
{
	if (argc < 3)
	{
		std::cout << information;
		return 1;
	}
	for (int i = 1; i < argc - 1; i ++)
	{
		std::string arg(argv[i]);
		std::string val(argv[i+1]);

		if (arg.compare("-d") == 0) 
		{
			genDatasetPath(val);
			i ++;
			continue;
		}
		else if (arg.compare("-r") == 0)
		{
			learningRate = atof(val.c_str());
			i ++;
			continue;
		}
		else if (arg.compare("-i") == 0) 
		{
			inputDim = atoi(val.c_str());
			i ++;
			continue;
		}
		else if (arg.compare("-h") == 0)
		{
			hiddenDim = atoi(val.c_str());
			i ++;
			continue;
		}
		else if (arg.compare("-m") == 0)
		{
			miniBatchSize = atoi(val.c_str());
			i ++;
			continue;
		}
		else if (arg.compare("-n") == 0)
		{
			numThreads = atoi(val.c_str());
			i ++;
			continue;
		}
		else if (arg.compare("-v") == 0)
		{
			version = atoi(val.c_str());
			i ++;
			continue;
		}
		else
		{
			std::cout << "No such a valid parameter:" << arg << std::endl;
			std::cout << information;
			return 1;
		}
	}

	return 0;
}

void ArgsTable::genDatasetPath(std::string dir)
{
	dataset_dir = dir;
	if (dataset_dir.back() == '/')
	{
		// val = val.Substring(0, val.length()-1);
		dataset_dir.pop_back();
	}

	train_src = dataset_dir + "/train_set.en";
	train_tgt = dataset_dir + "/train_set.ja";
	test_src = dataset_dir + "/test_set.en";
	test_tgt = dataset_dir + "/test_set.ja";

	return;
}


int main(int argc, char** argv){
	// const std::string src = "./corpus/sample.en";
	// const std::string tgt = "./corpus/sample.ja";
	// const std::string src = "../bigger_data/1k_train_en";
	// const std::string tgt = "../bigger_data/1k_train_ja";

	ArgsTable *args = new ArgsTable();
	int res = args->readInArgs(argc, argv);

	if (res != 0)
	{
		exit(1);
	}

	const std::string train_src = args->train_src;
	const std::string train_tgt = args->train_tgt;
	const std::string test_src = args->test_src;
	const std::string test_tgt = args->test_tgt;

	const Real learningRate = args->learningRate;
	const int inputDim = args->inputDim;
	const int hiddenDim = args->hiddenDim;
	const int miniBatchSize = args->miniBatchSize;
	const int numThreads = args->numThreads;

	const int version = args->version; // to clarify the version of training process

	Eigen::initParallel();
	
	std::cout << train_src << std::endl;
	std::cout << train_tgt << std::endl;
	std::cout << test_src << std::endl;
	std::cout << test_tgt << std::endl;
	std::cout << "learning rate = " << learningRate << std::endl;
	std::cout << "input dimention = " << inputDim << std::endl;
	std::cout << "hidden dimention = " << hiddenDim << std::endl;
	std::cout << "miniBatchSize = " << miniBatchSize <<  std::endl;
	std::cout << "number of threads = " << numThreads << std::endl;
	std::cout << "version = " << version << std::endl;

	if (version == 0)
	{
		// EncDec::demo(train_src, train_tgt, test_src, test_tgt);
		EncDec::demo(train_src, train_tgt, train_src, train_tgt);
	}
	else if (version == 1)
	{
		// EncDec::demo_qiao(train_src, train_tgt, test_src, test_tgt, learningRate, inputDim, hiddenDim, miniBatchSize, numThreads);
		EncDec::demo_qiao(train_src, train_tgt, train_src, train_tgt, learningRate, inputDim, hiddenDim, miniBatchSize, numThreads);
	}
	else if (version == 2)
	{
		//EncDec::demo_qiao_2(train_src, train_tgt, test_src, test_tgt, learningRate, inputDim, hiddenDim, miniBatchSize, numThreads);
		EncDec::demo_qiao_2(train_src, train_tgt, test_src, test_tgt, learningRate, inputDim, hiddenDim, miniBatchSize, numThreads);
	}
	else if (version == 3)
	{
		EncDec::demo_new_v1(train_src, train_tgt, test_src, test_tgt, learningRate, inputDim, hiddenDim, miniBatchSize, numThreads);
	}
	else if (version == 4)
	{
		EncDec::demo_qiao_3(train_src, train_tgt, test_src, test_tgt, learningRate, inputDim, hiddenDim, miniBatchSize, numThreads);
	}
	else if (version == 5)
	{
		EncDec::demo_new_v2(train_src, train_tgt, test_src, test_tgt, learningRate, inputDim, hiddenDim, miniBatchSize, numThreads);
	}
	else if (version == 6)
	{
		EncDec::demo_mf_1(train_src, train_tgt, test_src, test_tgt, learningRate, inputDim, hiddenDim, miniBatchSize, numThreads);
	}
	else if (version == 7)
	{
		EncDec::demo_mf_2(train_src, train_tgt, test_src, test_tgt, learningRate, inputDim, hiddenDim, miniBatchSize, numThreads);
	}
	return 0;
}
