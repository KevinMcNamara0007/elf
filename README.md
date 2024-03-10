to install on OS X 
git clone 
then 
g++ -std=c++11 -pthread elf.cpp -o elf
then copy elf to the bin folder of the llama.cpp /build/bin directory
example use
./elf "{prompt}"
