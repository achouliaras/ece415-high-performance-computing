# Makefile.

#Add any executable you want to be created here.
EXECUTABLES	= Convolution8 Convolution1 Convolution3 Convolution4

#This is the compiler to use
CC = nvcc

#These are the flags passed to the compiler. Change accordingly
CFLAGS = -G -Xcompiler -Wall -g -O4

#These are the flags passed to the linker. Nothing in our case
LDFLAGS =


# make all will create all executables
all: $(EXECUTABLES)

# This is the rule to create any executable from the corresponding .c
# file with the same name.
%: %.cu
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# make clean will remove all executables, jpg files and the
# output of previous executions.
clean:
	rm -f $(EXECUTABLES)
