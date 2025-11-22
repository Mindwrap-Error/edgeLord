# Compiler and Flags
CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -flto -funroll-loops -Wall -Wextra -Iinclude
TARGET   := phase1

# Source and Object Files
SRCS     := Graph.cpp SampleDriver.cpp
OBJS     := $(SRCS:.cpp=.o)

# Default Arguments (Can be overridden via command line)
ARG1     := graph.json
ARG2     := queries.json
ARG3     := output.json

# Phony Targets (Not actual files)
.PHONY: all clean run test

# Default Target: Build the executable
all: $(TARGET)

# Link Object Files to Create Executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compile Source Files into Object Files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run the Executable with 3 Arguments
run: $(TARGET)
	./$(TARGET) $(ARG1) $(ARG2) $(ARG3)

# Execute the Shell Script
test:
	chmod +x script.sh
	./script.sh

# Clean Build Artifacts
clean:
	rm -f $(OBJS) output.json