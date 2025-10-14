# Defining the constants
CPPC = g++

TARGET_EXEC   := tspmed
SRC_DIRS      := src/ts
BUILD_DIR     := build

# Find all the C++ files
# Create a list of object files
SRCS      := $(shell find $(SRC_DIRS) -name '*.cpp')
OBJS      := $(SRCS:$(SRC_DIRS)/%.cpp=$(BUILD_DIR)/%.o)

# The final build step
$(TARGET_EXEC): $(OBJS)
	$(CPPC) $(OBJS) -o $@

# Rule to create object files
$(BUILD_DIR)/%.o: $(SRC_DIRS)/%.cpp
	$(CPPC) -c $< -o $@

clean:
	rm -f $(TARGET_EXEC) $(OBJS)

.DEFAULT_GOAL := $(TARGET_EXEC)