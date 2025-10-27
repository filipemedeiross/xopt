# Defining the constants
CPPC = g++

# Flags
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -pedantic

TARGET_EXEC := tspmed
SRC_DIRS    := src
BUILD_DIR   := build

# Find all the C++ files
# Create a list of object files
SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' -not -path '*/python/*')
OBJS := $(SRCS:$(SRC_DIRS)/%.cpp=$(BUILD_DIR)/%.o)

# The final build step
$(TARGET_EXEC): $(OBJS)
	$(CPPC) $(CXXFLAGS) $(OBJS) -o $@

# Rule to create object files
$(BUILD_DIR)/%.o: $(SRC_DIRS)/%.cpp
	$(CPPC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET_EXEC) $(OBJS)

.DEFAULT_GOAL := $(TARGET_EXEC)