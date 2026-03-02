#include "prediction.hpp"
#include <pybind11/embed.h>
#include <iostream>

namespace py = pybind11;

// Global guard for the interpreter - it must stay alive for the whole program
static py::scoped_interpreter guard{}; 

VideoInferenceTool::VideoInferenceTool(const std::string& modelPath) {
    try {
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("append")("."); //

        py::module_ predictor_mod = py::module_::import("predictor"); //
        
        // Store the class instance
        py::object* obj = new py::object(predictor_mod.attr("VideoPredictor")(modelPath)); //
        predictor_instance = static_cast<void*>(obj);
    } catch (py::error_already_set &e) {
        std::cerr << "Initialization Error: " << e.what() << std::endl;
    }
}

PredictionResult VideoInferenceTool::runInference(const std::string& videoPath) {
    PredictionResult res;
    try {
        py::object* predictor = static_cast<py::object*>(predictor_instance);
        
        // Call the method from your Python script
        py::tuple result = predictor->attr("predict_video")(videoPath); //

        res.label = result[0].cast<std::string>(); //
        res.confidence = result[1].cast<float>(); //
    } catch (py::error_already_set &e) {
        std::cerr << "Inference Error: " << e.what() << std::endl;
        res.label = "Error";
        res.confidence = 0.0f;
    }
    return res;
}