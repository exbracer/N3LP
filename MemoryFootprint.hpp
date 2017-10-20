#pragma once
#include <fstream>
class MemoryFootprint
{
	public:
		MemoryFootprint()
		{
		};
		void record(std::string str, struct timeval start, int threadId);
		/* address of parameters */
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

		/* time start point of memory access */
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

		/* time end point of memory access */
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

void MemoryFootprint::record(std::string str, struct timeval start, int threadId)
{
	std::ofstream fout_r;
	fout_r.open(str.c_str());
	fout_r << threadId << std::endl;
	
	for (int i = 0; i < time_s_LSTM_Wxi.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Wxi[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Wxi[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Whi.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Whi[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Whi[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;
										
	for (int i = 0; i < time_s_LSTM_bi.size(); i ++)
	{
		double time = (double)((time_s_LSTM_bi[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_bi[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Wxf.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Wxf[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Wxf[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Whf.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Whf[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Whf[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_bf.size(); i ++)
	{
		double time = (double)((time_s_LSTM_bf[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_bf[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Wxo.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Wxo[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Wxo[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Who.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Who[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Who[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_bo.size(); i ++)
	{
		double time = (double)((time_s_LSTM_bo[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_bo[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Wxu.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Wxu[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Wxu[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Whu.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Whu[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Whu[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_bu.size(); i ++)
	{
		double time = (double)((time_s_LSTM_bu[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_bu[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;
	
	for (int i = 0; i < time_s_LSTM_Grad_Wxi.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Grad_Wxi[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Grad_Wxi[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Grad_Whi.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Grad_Whi[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Grad_Whi[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;
										
	for (int i = 0; i < time_s_LSTM_Grad_bi.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Grad_bi[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Grad_bi[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Grad_Wxf.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Grad_Wxf[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Grad_Wxf[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Grad_Whf.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Grad_Whf[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Grad_Whf[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Grad_bf.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Grad_bf[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Grad_bf[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Grad_Wxo.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Grad_Wxo[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Grad_Wxo[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Grad_Who.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Grad_Who[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Grad_Who[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Grad_bo.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Grad_bo[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Grad_bo[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Grad_Wxu.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Grad_Wxu[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Grad_Wxu[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Grad_Whu.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Grad_Whu[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Grad_Whu[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_LSTM_Grad_bu.size(); i ++)
	{
		double time = (double)((time_s_LSTM_Grad_bu[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_LSTM_Grad_bu[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_Softmax_weight.size(); i ++)
	{
		double time = (double)((time_s_Softmax_weight[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_Softmax_weight[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_Softmax_bias.size(); i ++)
	{
		double time = (double)((time_s_Softmax_bias[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_Softmax_bias[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_Softmax_Grad_weight.size(); i ++)
	{
		double time = (double)((time_s_Softmax_Grad_weight[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_Softmax_Grad_weight[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_s_Softmax_Grad_bias.size(); i ++)
	{
		double time = (double)((time_s_Softmax_Grad_bias[i].tv_sec - start.tv_sec) * 1000000.0 + (time_s_Softmax_Grad_bias[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	// for end point
	
	for (int i = 0; i < time_e_LSTM_Wxi.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Wxi[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Wxi[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Whi.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Whi[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Whi[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;
										
	for (int i = 0; i < time_e_LSTM_bi.size(); i ++)
	{
		double time = (double)((time_e_LSTM_bi[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_bi[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Wxf.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Wxf[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Wxf[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Whf.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Whf[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Whf[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_bf.size(); i ++)
	{
		double time = (double)((time_e_LSTM_bf[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_bf[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Wxo.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Wxo[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Wxo[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Who.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Who[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Who[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_bo.size(); i ++)
	{
		double time = (double)((time_e_LSTM_bo[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_bo[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Wxu.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Wxu[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Wxu[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Whu.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Whu[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Whu[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_bu.size(); i ++)
	{
		double time = (double)((time_e_LSTM_bu[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_bu[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;
	
	for (int i = 0; i < time_e_LSTM_Grad_Wxi.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Grad_Wxi[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Grad_Wxi[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Grad_Whi.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Grad_Whi[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Grad_Whi[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;
										
	for (int i = 0; i < time_e_LSTM_Grad_bi.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Grad_bi[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Grad_bi[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Grad_Wxf.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Grad_Wxf[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Grad_Wxf[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Grad_Whf.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Grad_Whf[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Grad_Whf[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Grad_bf.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Grad_bf[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Grad_bf[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Grad_Wxo.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Grad_Wxo[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Grad_Wxo[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Grad_Who.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Grad_Who[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Grad_Who[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Grad_bo.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Grad_bo[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Grad_bo[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Grad_Wxu.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Grad_Wxu[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Grad_Wxu[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Grad_Whu.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Grad_Whu[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Grad_Whu[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_LSTM_Grad_bu.size(); i ++)
	{
		double time = (double)((time_e_LSTM_Grad_bu[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_LSTM_Grad_bu[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_Softmax_weight.size(); i ++)
	{
		double time = (double)((time_e_Softmax_weight[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_Softmax_weight[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_Softmax_bias.size(); i ++)
	{
		double time = (double)((time_e_Softmax_bias[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_Softmax_bias[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_Softmax_Grad_weight.size(); i ++)
	{
		double time = (double)((time_e_Softmax_Grad_weight[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_Softmax_Grad_weight[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	for (int i = 0; i < time_e_Softmax_Grad_bias.size(); i ++)
	{
		double time = (double)((time_e_Softmax_Grad_bias[i].tv_sec - start.tv_sec) * 1000000.0 + (time_e_Softmax_Grad_bias[i].tv_usec - start.tv_usec))/1000.0;
		fout_r << time << " ";
	}
	fout_r << std::endl;

	fout_r.close();
	return;
}
