# Compiler and Flags
CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -flto -funroll-loops -Wall -Wextra -IInclude

# Default Arguments (can be overridden)
ARG1     := graph.json
ARG2     := queries.json
ARG3     := output.json

.PHONY: all phase1 phase2 clean run1 run2

all: phase1 phase2

###############################################################################
# Phase 1
###############################################################################
PH1_DIR  := ./Phase-1
PH1_SRCS := $(PH1_DIR)/Graph.cpp $(PH1_DIR)/SampleDriver.cpp
PH1_OBJS := $(PH1_SRCS:.cpp=.o)
PH1_BIN  := phase1

phase1: $(PH1_OBJS)
	$(CXX) $(CXXFLAGS) -o $(PH1_BIN) $(PH1_OBJS)

# run1: phase1
# 	./$(PH1_BIN) $(ARG1) $(ARG2) $(ARG3)

###############################################################################
# Phase 2
###############################################################################
PH2_DIR  := ./Phase-2
PH2_SRCS := $(PH2_DIR)/Graph.cpp $(PH2_DIR)/SampleDriver.cpp
PH2_OBJS := $(PH2_SRCS:.cpp=.o)
PH2_BIN  := phase2

phase2: $(PH2_OBJS)
	$(CXX) $(CXXFLAGS) -o $(PH2_BIN) $(PH2_OBJS)

# run2: phase2
# 	./$(PH2_BIN) $(ARG1) $(ARG2) $(ARG3)

###############################################################################
# Pattern Rule (works for both Phase1 and Phase2)
###############################################################################
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

###############################################################################
# Clean
###############################################################################
clean:
	rm -f $(PH1_OBJS) $(PH2_OBJS) phase1 phase2 output.json