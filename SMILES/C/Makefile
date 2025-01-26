# Makefile for gen_fingerprint_mat

# Compiler
CC = gcc

# Flags
CFLAGS = -Wall -O2

# Target executable
TARGET = gen

# Source files
SRCS = gen_fingerprint_mat.c
SRCS += cJSON.c

# Object files
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean