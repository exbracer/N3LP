################################
#
#	created by qiaoyc
#
################################

CXX=g++ # default compilier
#CXX=/home/qiao/user/bin/g++ # for magellan
#CXX=icc

EIGEN_LOCATION=./
#GPERF_LIB_LOCATION=/home/qiao/user/lib # for taura lab
#GPERF_LIB_LOCATION=/home/korchagin/user/lib # for gorgon0
GPERF_LIB_LOCATION=~/user/lib
BUILD_DIR=objs

TARGETS=
TARGETS+= n3lp
#TARGETS+= tc_n3lp

CXXFLAGS=
CXXFLAGS+= -O3
#CXXFLAGS+= -std=c++0x
CXXFLAGS+= -std=c++11
CXXFLAGS+= -funroll-loops
CXXFLAGS+= -march=native
CXXFLAGS+= -m64
CXXFLAGS+= -DEIGEN_DONT_PARALLELIZE
CXXFLAGS+= -DEIGEN_NO_DEBUG
CXXFLAGS+= -DEIGEN_NO_STATIC_ASSERT
CXXFLAGS+= -I $(EIGEN_LOCATION)
CXXFLAGS+= -fopenmp
#CXXFLAGS+= -openmp
CXXFLAGS+= -g
#CXXFLAGS+= -I${MKLROOT}/include
#CXXFLAGS+= -mkl=parallel
LDFLAGS= 
LDFLAGS+= -lm
#LDFLAGS+= -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl
#LDFLAGS+= -Wl,--no-as-needed -lgomp -lpthread -lm -ldl
#LDFLAGS+= -Wl,-start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,-end-group -lgomp -lpthread -lm -ldl
#LDFLAGS+= -liomp5 -lpthread -lm -ldl

TC_LDFLAGS=
TC_LDFLAGS+= -ltcmalloc -L $(GPERF_LIB_LOCATION) -Wl,-R$(GPERF_LIB_LOCATION)

SRCS=$(shell ls *.cpp)
OBJS=$(SRCS:.cpp=.o)

all: $(TARGETS)

n3lp: $(BUILD_DIR) $(patsubst %,$(BUILD_DIR)/%,n3lp)

$(BUILD_DIR)/%.o : %.cpp
	$(CXX) -o $@ -c $<  $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR)/n3lp : $(patsubst %, $(BUILD_DIR)/%,$(OBJS))
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)
	mv $@ ./
	rm -f ?*~

tc_n3lp: $(BUILD_DIR) $(patsubst %, $(BUILD_DIR)/%, tc_n3lp)

$(BUILD_DIR)/tc_%.o : %.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(LDFLAGS) $(TC_LDFLAGS)

$(BUILD_DIR)/tc_n3lp : $(patsubst %, $(BUILD_DIR)/tc_%, $(OBJS))
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(TC_LDFLAGS)
	mv $@ ./
	rm -f ?*~

inputAnalysis: inputAnalysis.cxx
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f $(BUILD_DIR)/* $(TARGETS) ?*~

objs_clean:
	rm -f $(BUILD_DIR)/*.o

.PHONY: all clean objs_clean
