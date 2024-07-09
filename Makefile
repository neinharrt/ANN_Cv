CC=icx
CFLAGS=-std=c++11 -Ofast -w

OBJ_DIR=obj
OBJ=$(OBJ_DIR)/model_basic.o\
		$(OBJ_DIR)/model.o\

TARGET=libann

.PHONY : all clean test

all : mkobj $(TARGET)

mkobj :
	-mkdir $(OBJ_DIR)
	-mkdir lib

test : all
	$(CC) -c main_example.cpp -o $(OBJ_DIR)/main_example.o -I./inc
	$(CC) -o test $(OBJ_DIR)/main_example.o -L./lib -lann -lstdc++

$(TARGET) : $(OBJ)
	ar rscv lib/$(TARGET).a $(OBJ)

$(OBJ_DIR)/%.o : src/%.cpp
	$(CC) -c $< -o $@ $(CFLAGS) -I./src -I./inc

clean :
	-rm $(OBJ_DIR)/*.o
	-rmdir $(OBJ_DIR)
	-rm lib/$(TARGET).a
	-rmdir lib
	-rm test
