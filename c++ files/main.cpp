#include "predictor.cpp"
#include <iostream>

namespace py = pybind11;

int main() {
    VideoInferenceTool myTool("save/cnnlstm_e5_b8ed7038fcc611f09084bab6975ba98b.pkl");

    auto result = myTool.runInference("test_split/FathersDay/FathersDayR0P0Y0_0");

    std::cout << "Detected: " << result.label << " (" << result.confidence << "%)" << std::endl;

    return 0;
}