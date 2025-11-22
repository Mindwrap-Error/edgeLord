# Compiler and Flags
CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -flto -funroll-loops -Wall -Wextra -Iinclude
TARGET   := phase2

# Source, Object, and Header Files
SRCS     := Graph.cpp SampleDriver.cpp
OBJS     := $(SRCS:.cpp=.o)
HEADERS  := Graph.hpp  # <--- Added this

# Default Arguments
ARG1     := graph.json
ARG2     := queries.json
ARG3     := output.json

.PHONY: all clean run test cleanx

# Default Target
all: $(TARGET)

# Link Object Files
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compile Source Files (Now depends on HEADERS too)
# This ensures if Graph.hpp changes, everything recompiles.
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run the Executable
run: $(TARGET)
	./$(TARGET) $(ARG1) $(ARG2) $(ARG3)

# Execute the Shell Script
test: $(TARGET)
	chmod +x script.sh
	./script.sh

# Clean Build Artifacts (Combined clean and cleanx)
clean:
	rm -f $(OBJS) output.json

cleanx:
	rm -f phase2