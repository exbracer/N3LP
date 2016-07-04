CXX=g++
#CXX=g++-6 # for Mac OS platform
#CXX=g++-5 # for Mac OS platform, lower version
#CXX=g++-4.9 # for Mac OS platform, lower version

#EIGEN_LOCATION=$$HOME/local/eigen_new #Change this line to use Eigen
EIGEN_LOCATION=./ 
BUILD_DIR=objs

CXXFLAGS=
#CXXFLAGS+=-Wall
CXXFLAGS+=-O3
CXXFLAGS+=-std=c++0x
CXXFLAGS+=-lm
CXXFLAGS+=-funroll-loops
CXXFLAGS+=-march=native
CXXFLAGS+=-m64
CXXFLAGS+=-DEIGEN_DONT_PARALLELIZE
CXXFLAGS+=-DEIGEN_NO_DEBUG
CXXFLAGS+=-DEIGEN_NO_STATIC_ASSERT
CXXFLAGS+=-I $(EIGEN_LOCATION)
CXXFLAGS+=-fopenmp

SRCS=$(shell ls *.cpp)
OBJS=$(SRCS:.cpp=.o)

PROGRAM=n3lp

all : $(BUILD_DIR) $(patsubst %,$(BUILD_DIR)/%,$(PROGRAM))

$(BUILD_DIR)/%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/$(PROGRAM) : $(patsubst %,$(BUILD_DIR)/%,$(OBJS))
	$(CXX) $(CXXFLAGS) $(CXXLIBS) -o $@ $^
	mv $(BUILD_DIR)/$(PROGRAM) ./
	rm -f ?*~

clean:
	rm -f $(BUILD_DIR)/* $(PROGRAM) ?*~
