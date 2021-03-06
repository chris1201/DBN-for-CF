CXX = clang++
CXXFLAGS = -Wall -Wextra -std=c++0x -DEIGEN_DEFAULT_TO_ROW_MAJOR
CPPFLAGS = -I$(HOME)/eigen/
LDFLAGS = -pthread
OBJ_DIR = obj
SRC_DIR = src

DEBUG ?= 0
ifeq ($(DEBUG), 1)
    CXXFLAGS += -DDEBUG -g
else
    CXXFLAGS += -O3 -DNDEBUG -DEIGEN_NO_DEBUG
endif

sources = $(wildcard $(SRC_DIR)/*.cpp)
objects = $(addprefix $(OBJ_DIR)/, $(notdir $(sources:.cpp=.o)))
executables = rbm

all: $(executables)

rbm: $(objects)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(CPPFLAGS) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $^ $(CPPFLAGS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(executables)
