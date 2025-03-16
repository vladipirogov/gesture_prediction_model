#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

/**
 * g++ GestureModel.cpp -o model_test \
  -I/home/user/tensorflow/ \
  -I/home/user/tensorflow/tensorflow/lite/tools/make/downloads/ \
  -I/home/user/tensorflow/tensorflow/lite/micro/ \
  -I/home/user/tensorflow/tensorflow/lite/core/shims/c/ \
  -L/home/user/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/ \
  -ltensorflow-lite -pthread
 */
class GestureModel {
public:
    GestureModel() {
        const std::string model_path = "gesture_model.tflite"; // Константное имя модели

        // Load the model from the file
        model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (!model) {
            throw std::runtime_error("Failed to load model");
        }

        // Create an interpreter
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        if (!interpreter) {
            throw std::runtime_error("Failed to create interpreter");
        }

        // Allocate memory for tensors
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            throw std::runtime_error("Failed to allocate tensors");
        }
        interpreter->SetAllowFp16PrecisionForFp32(true);
    }

    void Run(const std::vector<float>& input_data) {
        // Get the input tensor
        int input_index = interpreter->inputs()[0];
        float* input_ptr = interpreter->typed_tensor<float>(input_index);

        // Copy data to tensor
        std::memcpy(input_ptr, input_data.data(), input_data.size() * sizeof(float));

        // Launch the interpreter
        if (interpreter->Invoke() != kTfLiteOk) {
            throw std::runtime_error("Failed to invoke interpreter");
        }

        // Get the output data
        float* output_ptr = interpreter->typed_output_tensor<float>(0);

        std::cout << "Output values:" << std::endl;
        for (int i = 0; i < interpreter->output_tensor(0)->dims->data[1]; ++i) {
            std::cout << "Output[" << i << "]: " << output_ptr[i] << std::endl;
        }
    }

private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
};

std::vector<float> LoadCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file");
    }

    std::vector<float> data;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            data.push_back(std::stof(value));
        }
    }

    return data;
}

/**
 * ./model_test input_data.csv
 */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <csv_path>" << std::endl;
        return 1;
    }

    try {
        GestureModel model;
        std::vector<float> input_data = LoadCSV(argv[1]);
        model.Run(input_data);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
