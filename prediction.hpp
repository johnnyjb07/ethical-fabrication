#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include <string>
#include <utility>

// Structure to hold our results
struct PredictionResult {
    std::string label;
    float confidence;
};

class VideoInferenceTool {
public:
    // Constructor handles Python startup and model loading
    VideoInferenceTool(const std::string& modelPath);
    
    // Function you will call in other projects
    PredictionResult runInference(const std::string& videoPath);

private:
    // We keep these as generic objects to hide pybind11 from the header if needed
    void* predictor_instance; 
};

#endif