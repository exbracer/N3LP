#include "EncDec.hpp"

#include <iostream>
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
	}
	int readInArgs(int argc, char ** argv);
private:
	void genDatasetPath();
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
			genDatasetPath();
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
		else
		{
			std::cout << "No such a valid parameter:" << arg << std::endl;
			std::cout << information;
			return 1;
		}
	}

	return 0;
}

void ArgsTable::genDatasetPath()
{
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
	std::cout << "hahaha" << std::endl;
	Eigen::initParallel();
	std::cout << train_src << std::endl;
	std::cout << train_tgt << std::endl;
	std::cout << test_src << std::endl;
	std::cout << test_tgt << std::endl;
	std::cout << learningRate << std::endl;
	std::cout << miniBatchSize <<  std::endl;
	std::cout << numThreads << std::endl;
	EncDec::demo_qiao(train_src, train_tgt, train_src, train_tgt, learningRate,
			inputDim, hiddenDim, miniBatchSize, numThreads);

	return 0;

}
