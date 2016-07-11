#include "EncDec.hpp"

int main(int argc, char** argv){
  // const std::string src = "./corpus/sample.en";
  // const std::string tgt = "./corpus/sample.ja";

	const std::string src = "../bigger_data/1k_train_en";
	const std::string tgt = "../bigger_data/1k_train_ja";

  Eigen::initParallel();
  EncDec::demo(src, tgt, src, tgt);

  return 0;

}
