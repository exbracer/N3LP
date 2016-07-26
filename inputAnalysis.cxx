/*************************************************************************
    > File Name: calNumTerms.cpp
    > Author: qiao_yuchen
    > Mail: qiaoyc14@mails.tsinghua.edu.cn 
    > Created Time: Mon Jul 25 20:27:09 2016
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <vector>

#define Real double

namespace Utils
{
	inline Real max(const Real& x, const Real& y)
	{
		return x > y ? x : x;
	}
	inline Real min(const Real& x, const Real& y)
	{
		return x > y ? y : x;
	}

	inline bool isSpace(const char& c)
	{
		return (c == ' ' || c == '\t');
	}

	inline void split(const std::string& str, std::vector<std::string>& res)
	{
		bool tok = false;
		int beg = 0;

		res.clear();

		for (int i = 0, len = str.length(); i < len; ++i)
		{
			if (!tok && str[i] != Utils::isSpace(str[i]))
			{
				beg = i;
				tok = true;
			}

			if (tok && (i == len-1 || Utils::isSpace(str[i]))) 
			{
				tok = false;
				res.push_back(isSpace(str[i]) ? str.substr(beg, i-beg) : str.substr(beg, i-beg+1));
			}
		}
	}

	inline void split(const std::string& str, std::vector<std::string>& res, const char sep)
	{
		bool tok = false;
		int beg = 0;

		res.clear();

		for (int i = 0, len = str.length(); i < len; ++i)
		{
			if (!tok && str[i] != sep)
			{
				beg = i;
				tok = true;
			}

			if (tok && (i == len-1 || Utils::isSpace(str[i])))
			{
				tok = false;
				res.push_back((str[i] == sep) ? str.substr(beg, i-beg) : str.substr(beg, i-beg+1));
			}
		}
	}
}

int main(int argc, char **argv)
{
	std::string src_train_file(argv[1]);
	std::string tgt_train_file(argv[2]);

	int maxNumLines = atoi(argv[3]);
	std::ifstream ifsSrcTrain(src_train_file.c_str());
	std::ifstream ifsTgtTrain(tgt_train_file.c_str());

	std::vector<std::string> tokens;

	int numLine = 0;

	int numSrcTokens = 0;
	int numTgtTokens = 0;
	// read training daay
	for (std::string line;std::getline(ifsSrcTrain, line) && numLine < maxNumLines; numLine ++)
	{
		Utils::split(line, tokens);
		numSrcTokens += tokens.size();
	}

	std::cout << "average tokens of first " << numLine << " sentence for Source Data is: " << (Real)(numSrcTokens)/numLine << std::endl;

	numLine = 0;
	for (std::string line;std::getline(ifsTgtTrain, line) && numLine < maxNumLines; numLine ++)
	{
		Utils::split(line, tokens);
		numTgtTokens += tokens.size();
	}

	std::cout << "average tokens of first " << numLine << " sentence for Target Data is: " << (Real)(numTgtTokens)/numLine << std::endl;
	
	
	return 0;
}

