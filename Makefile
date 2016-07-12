################################
#
#	created by qiaoyc
#
################################
CXX=g++

EIGNE_LOCATION=./
BUILD_DIR=objs
TARGET=
TARGET+= n3lp
TARGET+= n3lp_tc

CXXFLAGS=

LDFLAGS=

all: TARGET

n3lp: $(BUILD_DIR) $(patsubst %,$(BUILD_DIR)%,n3lp)

$(BUILD_DIR)/%.o : %.cpp
	$(CXX) -o $@ $< -c $(CXXFLAGS)

$(BUILD_DIR)/

n3lp_tc:

clean:
	rm -f $(BUILD_DIR)/* ?*~
	rm -r n3lp
	rm 
