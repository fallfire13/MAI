#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "diff.h"

std::vector<std::string> read_file(char* filename) {
    std::ifstream is(filename);
    std::string line;
    std::vector<std::string> text;

    while(std::getline(is, line)) {
        text.push_back(line);
    }

    return text;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout 
            << "Usage: " 
            << argv[0] 
            << " FILE1 FILE2" 
            << std::endl;
        return -1;
    }

    std::vector<std::string> text1 = read_file(argv[1]);
    std::vector<std::string> text2 = read_file(argv[2]);

    std::vector<TAction> actions( find_diff(text1, text2) );

    for (const auto& act : actions) {
        switch (act.type) {
          case TAction::ADD: {
            std::cout << "+ ";
            std::cout << text2[act.y] << std::endl;
            break;
          }
          case TAction::DEL: {
            std::cout << "- ";
            std::cout << text1[act.x] << std::endl;
            break;
          }
          case TAction::KEEP: {
            std::cout << "  ";
            std::cout << text1[act.x] << std::endl;
            break;
          }
        }
    }
}