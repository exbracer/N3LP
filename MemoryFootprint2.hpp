#pragma once
#include <fstream>
class MemoryFootprint2
{
	public:
		MemoryFootprint2();
		~MemoryFootprint2();
		void record(std::string str, struct timeval start, int threadId);
		
		// index for record
		std::vector<std::vector<Real*>*> addr_index;
		std::vector<std::vector<int>*> size_index;
		std::vector<std::vector<std::vector<struct timeval>*>*> time_r_index;
		std::vector<std::vector<std::vector<struct timeval>*>*> time
		/* address of parameters */
		std::vector<Real*> addr_LSTM;
		std::vector<Real*> addr_LSTM_Grad;
		std::vector<Real*> addr_LSTM_State;
		std::vector<Real*> addr_LSTM_TempVec;
		std::vector<Real*> addr_Softmax;
		std::vector<Real*> addr_Softmax_Grad;
		std::vector<Real*> addr_Softmax_TempVec;

		/* address of data */
		std::vector<Real*> addr_Data;
		/* memory size of parameters */
		std::vector<int> size_LSTM;
		std::vector<int> size_LSTM_Grad;
		std::vector<int> size_LSTM_State;
		std::vector<int> size_LSTM_TempVec;
		std::vector<int> size_Softmax;
		std::vector<int> size_Softmax_Grad;
		std::vector<int> size_Softmax_TempVec;

		/* memory size of data */
		std::vector<int> size_Data;

		/* tims start point of read-type memory access */
		std::vector<std::vector<struct timeval>*> time_s_r_LSTM;
		std::vector<std::vector<struct timeval>*> time_s_r_LSTM_Grad;
		std::vector<std::vector<struct timeval>*> time_s_r_LSTM_State;
		std::vector<std::vector<struct timeval>*> time_s_r_LSTM_TempVec;
		std::vector<std::vector<struct timeval>*> time_s_r_Softmax;
		std::vector<std::vector<struct timeval>*> time_s_r_Softmax_Grad;
		std::vector<std::vector<struct timeval>*> time_s_r_Softmax_TempVec;
		std::vector<std::vector<struct timeval>*> time_s_r_Data;
		
		/* time end point of read-type memory access */
		std::vector<std::vector<struct timeval>*> time_e_r_LSTM;
		std::vector<std::vector<struct timeval>*> time_e_r_LSTM_Grad;
		std::vector<std::vector<struct timeval>*> time_e_r_LSTM_State;
		std::vector<std::vector<struct timeval>*> time_e_r_LSTM_TempVec;
		std::vector<std::vector<struct timeval>*> time_e_r_Softmax;
		std::vector<std::vector<struct timeval>*> time_e_r_Softmax_Grad;
		std::vector<std::vector<struct timeval>*> time_e_r_Softmax_TempVec;
		std::vector<std::vector<struct timeval>*> time_e_r_Data;

		/* time start point of write-type memory access */
		std::vector<std::vector<struct timeval>*> time_s_w_LSTM;
		std::vector<std::vector<struct timeval>*> time_s_w_LSTM_Grad;
		std::vector<std::vector<struct timeval>*> time_s_w_LSTM_State;
		std::vector<std::vector<struct timeval>*> time_s_w_LSTM_TempVec;
		std::vector<std::vector<struct timeval>*> time_s_w_Softmax;
		std::vector<std::vector<struct timeval>*> time_s_w_Softmax_Grad;
		std::vector<std::vector<struct timeval>*> time_s_w_Softmax_TempVec;
		std::vector<std::vector<struct timeval>*> time_s_w_Data;

		/* time end point of write-type memory access */
		std::vector<std::vector<struct timeval>*> time_e_w_LSTM;
		std::vector<std::vector<struct timeval>*> time_e_w_LSTM_Grad;
		std::vector<std::vector<struct timeval>*> time_e_w_LSTM_State;
		std::vector<std::vector<struct timeval>*> time_e_w_LSTM_Tempvec;
		std::vector<std::vector<struct timeval>*> time_e_w_Softmax;
		std::vector<std::vector<struct timeval>*> time_e_w_Softmax_Grad;
		std::vector<std::vector<struct timeval>*> time_e_w_Softmax_TempVec;
		std::vector<std::vector<struct timeval>*> time_e_w_Data;

		/*
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
		*/
};
MemoryFootprint2::MemoryFootprint2()
{
	// create instance of std::vector<struct timeval> for time_s_r_LSTM
	for (int i = 0; i < 12; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_r_LSTM.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_s_r_LSTM_Grad
	for (int i = 0; i < 12; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_r_LSTM_Grad.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_s_r_LSTM_State
	for (int i = 0; i < 11; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_r_LSTM_State.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_s_r_LSTM_TempVec
	for (int i = 0; i < 4; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_r_LSTM_TempVec.push_back(vec_time);
	}
	// create instnace of std::vector<struct timeval> for time_s_r_Softmax
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_r_Softmax.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_s_r_Softmax_Grad
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_r_Softmax_Grad.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_s_r_Softmax_TempVec
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_r_Softmax_TempVec.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_s_r_Data
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_r_Data.push_back(vec_time);
	}
	
	// create instance of std::vector<struct timeval> for time_e_r_LSTM
	for (int i = 0; i < 12; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_r_LSTM.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_e_r_LSTM_Grad
	for (int i = 0; i < 12; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_r_LSTM_Grad.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_e_r_LSTM_State
	for (int i = 0; i < 11; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_r_LSTM_State.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_e_r_LSTM_TempVec
	for (int i = 0; i < 4; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_r_LSTM_TempVec.push_back(vec_time);
	}
	// create instnace of std::vector<struct timeval> for time_e_r_Softmax
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_r_Softmax.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_e_r_Softmax_Grad
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_r_Softmax_Grad.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_e_r_Softmax_TempVec
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_r_Softmax_TempVec.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_e_r_Data
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_r_Data.push_back(vec_time);
	}

	// create instance of std::vector<struct timeval> for time_s_w_LSTM
	for (int i = 0; i < 12; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_w_LSTM.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_s_w_LSTM_Grad
	for (int i = 0; i < 12; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_w_LSTM_Grad.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_s_w_LSTM_State
	for (int i = 0; i < 11; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_w_LSTM_State.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_s_w_LSTM_TempVec
	for (int i = 0; i < 4; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_w_LSTM_TempVec.push_back(vec_time);
	}
	// create instnace of std::vector<struct timeval> for time_s_w_Softmax
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_w_Softmax.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_s_w_Softmax_Grad
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_w_Softmax_Grad.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_s_w_Softmax_TempVec
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_w_Softmax_TempVec.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_s_w_Data
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_s_w_Data.push_back(vec_time);
	}
	
	// create instance of std::vector<struct timeval> for time_e_w_LSTM
	for (int i = 0; i < 12; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_w_LSTM.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_e_w_LSTM_Grad
	for (int i = 0; i < 12; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_w_LSTM_Grad.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_e_w_LSTM_State
	for (int i = 0; i < 11; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_w_LSTM_State.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_e_w_LSTM_TempVec
	for (int i = 0; i < 4; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_w_LSTM_TempVec.push_back(vec_time);
	}
	// create instnace of std::vector<struct timeval> for time_e_w_Softmax
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_w_Softmax.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_e_w_Softmax_Grad
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_w_Softmax_Grad.push_back(vec_time);
	}
	// create instance of std::vector<struct timeval> for time_e_w_Softmax_TempVec
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_w_Softmax_TempVec.push_back(vec_time);
	}
	for (int i = 0; i < 2; i ++)
	{
		std::vector<struct timeval>* vec_time = new std::vector<struct timeval>;
		time_e_w_Data.push_backk(vec_time);
	}

	// push address of vectors for record into index vector
	addr_index.push_back(&addr_LSTM);
	addr_index.push_back(&addr_LSTM_Grad);
	addr_index.push_back(&addr_LSTM_State);
	addr_index.push_back(&addr_LSTM_TempVec);
	addr_index.push_back(&addr_Softmax);
	addr_index.push_back(&addr_Softmax_Grad);
	addr_index.push_back(&addr_Softmax_TempVec);

	// push size of vectors for record into index vector
	size_index.push_back(&size_LSTM);
	size_index.push_back(&size_LSTM_Grad);
	size_index.push_back(&size_LSTM_State);
	size_index.push_back(&size_LSTM_TempVec);
	size_index.push_back(&size_Softmax);
	size_index.push_back(&size_Softmax_Grad);
	size_index.push_back(&size_Softmax_TempVec);

	// push timestamp of vectors for recording read-type memory access into the index vector
	time_r_index.push_back(&time_s_r_LSTM);
	time_r_index.push_back(&time_s_r_LSTM_Grad);
	time_r_index.push_back(&time_s_r_LSTM_State);
	time_r_index.push_back(&time_s_r_LSTM_TempVec);
	time_r_index.push_back(&time_s_r_Softmax);
	time_r_index.push_back(&time_s_r_Softmax_Grad);
	time_r_index.push_back(&time_s_r_Softmax_TempVec);

	time_r_index.push_back(&time_e_r_LSTM);
	time_r_index.push_back(&time_e_r_LSTM_Grad);
	time_r_index.push_back(&time_e_r_LSTM_State);
	time_r_index.push_back(&time_e_r_LSTM_TempVec);
	time_r_index.push_back(&time_e_r_Softmax);
	time_r_index.push_back(&time_e_r_Softmax_Grad);
	time_r_index.push_back(&time_e_r_Softmax_TempVec);

	// push timestamp of vectors for recofding write-type memory access into the index vector
	time_w_index.push_back(&time_s_w_LSTM);
	time_w_index.push_back(&time_s_w_LSTM_Grad);
	time_w_index.push_back(&time_s_w_LSTM_State);
	time_w_index.push_back(&time_s_w_LSTM_TempVec);
	time_w_index.push_back(&time_s_w_Softmax);
	time_w_index.push_back(&time_s_w_Softmax_Grad);
	time_w_index.push_back(&time_s_w_Softmax_TempVec);

	time_w_index.push_back(&time_e_w_LSTM);
	time_w_index.push_back(&time_e_w_LSTM_Grad);
	time_w_index.push_back(&time_e_w_LSTM_State);
	time_w_index.push_back(&time_e_w_LSTM_TempVec);
	time_w_index.push_back(&time_e_w_Softmax);
	time_w_index.push_back(&time_e_w_Softmax_Grad);
	time_w_index.push_back(&time_e_w_Softmax_TempVec);
	
	
}
MemoryFootprint2::~MemoryFootprint2()
{
	for (int i = 0; i < time_r_index.size(); i ++)
	{
		std::vector<std::vector<struct timeval*>* p = time_r_index[i];
		for (int j = 0; j < p->size(); j ++)
		{
			p->at(j).delete();
		}
	}
	for (int i = 0; i < time_w_index.size(); i ++)
	{
		std::vector<std::vector<struct timeval*>*> p = time_w_index[i];
		for (int j = 0; j < p->size(); j ++)
		{
			p->at(j).delete();
		}
	}
}
void MemoryFootprint2::record(std::string str, struct tivmeal start, int threadId)
{
	std::ofstream fout;
	if (threadId == 0)
	{
		fout.open(str.c_str(), std::ios::out);
	}
	else 
	{
		fout.open(str.c_str(), std::ios::app);
	}
	// write the id of the thread frist 
	fout << threadId << std::endl;

	// write the address then
	for (int i = 0; i < addr_index.size(); i ++)
	{
		std::vector<Real*>* p = addr_index[i];
		for (int j = 0; j < p->size(); j ++)
		{
			fout << p->at(j) << " ";
		}
		fout << std::endl;
	}

	// write the memory size then
	for (int i = 0; i < size_index.size(); i ++)
	{
		std::vector<int>* p = size_inde[i];
		for (int j = 0; j < p->size(); j ++)
		{
			fout << p->at(j) << " ";
		}
		fout << std::endl;
	}

	// write the timestamp then
	for (int i = 0; i < time_r_index.size(); i ++)
	{
		std::vector<std::vector<struct timeval>*>* p = time_r_index[i];
		for (int j = 0; j < p->size(): j ++)
		{
			std::vector<struct timeval>* q = p->at(j);
			for (int k = 0; k < q->size(); k ++)
			{
				fout << q->at(k) << " ";
			}
			fout << std::endl;
		}
	}
	for (int i = 0; i < time_w_index.size(); i ++)
	{
		std::vector<std::vector<struct timeval>*>* p = time_w_index[i];
		for (int j = 0; j < p->size(); j ++)
		{
			std::vector<struct timeval>* q = p->at(j);
			for (int k = 0; k < q->size(); q ++)
			{
				fout << q->at(k) << " ";
			}
			fout << std::endl;
		}
	}
	
	return;
}
/*
void MemoryFootprint2::record(std::string str, struct timeval start, int threadId)
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
*/
