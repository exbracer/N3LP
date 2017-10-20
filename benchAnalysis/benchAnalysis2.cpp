/*************************************************************************
    > File Name: benchAnalysis.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: Mon 27 Mar 2017 05:23:02 PM JST
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <chrono>

class TripleInt
{
public:
	int first;
	int second;
	int third;

	TripleInt(int a, int b, int c)
	{
		first = a;
		second = b;
		third = c;
	}
	~TripleInt(){}
};

namespace Utils
{
	inline bool isSpace(const char& c)
	{
		return (c == ' ' || c == '\t');
	}

	inline void split(const std::string& str, std::vector<std::string>& res)
	{
		bool tok = false;
		int beg = 0;

		res.clear();
		for (int i = 0, len = str.length(); i < len; ++ i)
		{
			if (!tok && !Utils::isSpace(str[i]))
			{
				beg = i;
				tok= true;
			}

			if (tok && (i == len-1 || Utils::isSpace(str[i])))
			{
				tok = false;
				res.push_back(isSpace(str[i]) ? str.substr(beg, i-beg) : str.substr(beg, i-beg+1));
			}
		}
	}
	inline void printTripleList(std::vector<TripleInt>& tripleIntList)
	{
		for (int i = 0, len = tripleIntList.size(); i < len; i ++)
		{
			std::cout << tripleIntList[i].first << ' ';
			std::cout << tripleIntList[i].second << ' ';
			std::cout << tripleIntList[i].third << std::endl;
		}
	}
}

bool myCompare1(const TripleInt& tripleInt1, const TripleInt& tripleInt2)
{
	return tripleInt1.first < tripleInt2.first;
}

bool myCompare2(const TripleInt& tripleInt1, const TripleInt& tripleInt2)
{
	return tripleInt1.second < tripleInt2.second;
}

bool myCompare3(const TripleInt& tripleInt1, const TripleInt& tripleInt2)
{
	return tripleInt1.third < tripleInt2.third;
}

double paddingAnalysis(std::vector<TripleInt>& list, int type, int minibatch_size)
{
	if (type == 0)
	{
		//do nothing
	}
	else if (type == 1)
	{
		std::sort(list.begin(), list.end(), myCompare1);
	}
	else if (type == 2)
	{
		std::sort(list.begin(), list.end(), myCompare2);
	}
	else if (type == 3)
	{
		std::sort(list.begin(), list.end(), myCompare3);
	}
	else if (type == 4)
	{
		// std::random_shuffle(list.begin(), list.end());
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(list.begin(), list.end(), std::default_random_engine(seed));
	}
	else if (type == 5)
	{
		// do nothing
	}


	int length = list.size();

	int pad_sum_src = 0;
	int pad_sum_tgt = 0;

	int sum_src = 0;
	int sum_tgt = 0;

	for (int i = 0; i < length; i += minibatch_size)
	{
		int index_begin = i;
		int index_end = i+minibatch_size;
		if (index_end >= length)
		{
			index_end = length;
		}
		
		int max_src_len = 0;
		int max_tgt_len = 0;
		int max_pair_len = 0;
		
		for (int j = index_begin; j < index_end; j ++)
		{
			if (list[j].first > max_src_len)
			{
				max_src_len = list[j].first;
			}
			if (list[j].second > max_tgt_len)
			{
				max_tgt_len = list[j].second;
			}
			if (list[j].third > max_pair_len)
			{
				max_pair_len = list[j].third;
			}
		}

		for (int j = index_begin; j < index_end; j ++)
		{
			pad_sum_src += max_src_len-list[j].first;
			pad_sum_tgt += max_tgt_len-list[j].second;
			sum_src += list[j].first;
			sum_tgt += list[j].second;
		}
	}

	// output information
	std::cout << "For type " << type << ":"<< std::endl; 
	std::cout << pad_sum_src << ' ' << sum_src << std::endl;
	std::cout << pad_sum_tgt << ' ' << sum_tgt << std::endl;

	double padding_ratio = (double)(pad_sum_tgt+pad_sum_src) / (double)(pad_sum_src+pad_sum_tgt+sum_tgt+sum_src);
	std::cout << "Padding Ratio: " << padding_ratio << std::endl;

	return padding_ratio;
}

void paddingAnalysis2(std::vector<TripleInt>& list) // for bucketing
{
	int length = list.size();

	int pad_sum_src = 0;
	int pad_sum_tgt = 0;

	int sum_src = 0;
	int sum_tgt = 0;

	int num_odd_pairs = 0;

	for (int i = 0; i < length; i ++)
	{
		if (list[i].first <= 5 && list[i].second <= 10)
		{
			pad_sum_src += 5 - list[i].first;
			pad_sum_tgt += 10 - list[i].second;
		}
		else if (list[i].first <= 10 && list[i].second <= 15)
		{
			pad_sum_src += 10 - list[i].first;
			pad_sum_tgt += 15 - list[i].second;
		}
		else if (list[i].first <= 20 && list[i].second <= 25)
		{
			pad_sum_src += 20 - list[i].first;
			pad_sum_tgt += 25 - list[i].second;
		}
		else if (list[i].first <= 50 && list[i].second <= 50)
		{
			pad_sum_src += 40 - list[i].first;
			pad_sum_tgt += 50 - list[i].second;
		}
		else
		{
			num_odd_pairs ++;
		}
		sum_src += list[i].first;
		sum_tgt += list[i].second;
	}
	// output information
	std::cout << "For bucketing:"<< std::endl; 
	std::cout << pad_sum_src << ' ' << sum_src << std::endl;
	std::cout << pad_sum_tgt << ' ' << sum_tgt << std::endl;
	std::cout << "num of odd pairs: " << num_odd_pairs << std::endl;

	double padding_ratio = (double)(pad_sum_tgt+pad_sum_src) / (double)(pad_sum_src+pad_sum_tgt+sum_tgt+sum_src);
	std::cout << "Padding Ratio: " << padding_ratio << std::endl;

	return;
}

int main (int argc, char ** argv)
{
	std::string data_src(argv[1]);
	std::string data_tgt(argv[2]);

	int minibatch_size = atoi(argv[3]);

	std::ifstream ifsSrc(data_src.c_str());
	std::ifstream ifsTgt(data_tgt.c_str());

	int src_num_line = 0;
	int tgt_num_line = 0;

	std::vector<std::string> src_tokens;
	std::vector<std::string> tgt_tokens;

	std::vector<TripleInt> list_length;

	for (std::string line; std::getline(ifsSrc, line);)
	{
		Utils::split(line, src_tokens);
		list_length.push_back(TripleInt(src_tokens.size(), 0, 0));
		src_num_line ++;
	}

	for (std::string line; std::getline(ifsTgt, line);)
	{
		if (tgt_num_line > src_num_line)
		{
			std::cout << "Error: Source data and Target data don't have the same length" << std::endl;
			return 0;
		}
		Utils::split(line, tgt_tokens);
		list_length[tgt_num_line].second = tgt_tokens.size();
		list_length[tgt_num_line].third = list_length[tgt_num_line].first + list_length[tgt_num_line].second;

		tgt_num_line ++;
	}

	std::cout << "tgt_num_line = " << tgt_num_line << std::endl;
	//Utils::printTripleList(list_length);

	// paddingAnalysis(list_length, 0, minibatch_size);
	
	double padding_ratio = 0.0;
	padding_ratio += paddingAnalysis(list_length, 4, minibatch_size);
	padding_ratio += paddingAnalysis(list_length, 4, minibatch_size);
	padding_ratio += paddingAnalysis(list_length, 4, minibatch_size);
	padding_ratio += paddingAnalysis(list_length, 4, minibatch_size);
	padding_ratio += paddingAnalysis(list_length, 4, minibatch_size);
	padding_ratio /= 5.0;
	std::cout << "The average padding ratio is " << padding_ratio << std::endl;
	paddingAnalysis2(list_length);
	// paddingAnalysis(list_length, 1, minibatch_size);
	// paddingAnalysis(list_length, 2, minibatch_size);
	
	
	paddingAnalysis(list_length, 3, minibatch_size);

	return 0;
}
