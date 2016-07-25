################################
#
#	created by qiaoyc
#
################################
CXX=g++

EIGEN_LOCATION=./
GPERF_LIB_LOCATION=/home/qiao/user/lib
BUILD_DIR=objs

TARGETS=
TARGETS+= n3lp
TARGETS+= n3lp_tc

CXXFLAGS=
CXXFLAGS+= -O3
CXXFLAGS+= -std=c++0x
CXXFLAGS+= -funroll-loops
CXXFLAGS+= -march=native
CXXFLAGS+= -m64
CXXFLAGS+= -DEIGEN_DONT_PARALLELIZE
CXXFLAGS+= -DEIGEN_NO_DEBUG
CXXFLAGS+= -DEIGEN_NO_STATIC_ASSERT
CXXFLAGS+= -I $(EIGEN_LOCATION)
CXXFLAGS+= -fopenmp
#CXXFLAGS+= -pg
LDFLAGS= 
LDFLAGS+= -lm

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

n3lp_tc: $(BUILD_DIR) $(patsubst %, $(BUILD_DIR)/%, n3lp_tc)

$(BUILD_DIR)/tc_%.o : %.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(LDFLAGS) $(TC_LDFLAGS)

$(BUILD_DIR)/n3lp_tc : $(patsubst %, $(BUILD_DIR)/tc_%, $(OBJS))
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(TC_LDFLAGS)
	mv $@ ./
	rm -f ?*~

inputAnalysis: inputAnalysis.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f $(BUILD_DIR)/* $(TARGETS) ?*~

objs_clean:
	rm -f $(BUILD_DIR)/*.o

.PHONY: all clean objs_clean
