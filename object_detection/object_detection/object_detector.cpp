#include <iostream>
#include <chrono>
#include <vector>

#include "object_detector.h"


ObjectDetector::ObjectDetector(const float score_threshold)
    : score_threshold_(score_threshold)
{

}

bool ObjectDetector::BuildInterpreter(
    const std::string& model_path,
    const unsigned int num_of_threads)
{
    // Load Model
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (model_ == nullptr)
    {
        std::cerr << "Fail to build FlatBufferModel from file: " << model_path << std::endl;
        return false;
    }

    // Build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    if (tflite::InterpreterBuilder(*model_, resolver)(&interpreter_) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter." << std::endl;
        return false;
    }

    interpreter_->SetNumThreads(num_of_threads);

    // Bind given context with interpreter.
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
    }

    // Get input tensor size.
    const auto& dimensions = interpreter_->tensor(interpreter_->inputs()[0])->dims;

    input_height_ = dimensions->data[1];
    input_width_ = dimensions->data[2];
    input_channels_ = dimensions->data[3];

    // Get output tensor
    output_locations_ = interpreter_->tensor(interpreter_->outputs()[0]);
    output_classes_ = interpreter_->tensor(interpreter_->outputs()[1]);
    output_scores_ = interpreter_->tensor(interpreter_->outputs()[2]);
    num_detections_ = interpreter_->tensor(interpreter_->outputs()[3]);

    // Display input size
    auto input_array_size = 1;
    std::vector<int> input_tensor_shape;

    input_tensor_shape.resize(dimensions->size);
    for (auto i = 0; i < dimensions->size; i++)
    {
        input_tensor_shape[i] = dimensions->data[i];
        input_array_size *= input_tensor_shape[i];
    }

    std::ostringstream input_string_stream;
    std::copy(input_tensor_shape.begin(), input_tensor_shape.end(), std::ostream_iterator<int>(input_string_stream, " "));

    std::cout << "input shape: " << input_string_stream.str() << std::endl;
    std::cout << "input array size: " << input_array_size << std::endl;

    return true;
}

std::unique_ptr<std::vector<BoundingBox>> ObjectDetector::RunInference(
    const std::vector<uint8_t>& input_data,
    std::chrono::duration<double, std::milli>& time_span)
{
    const auto& start_time = std::chrono::steady_clock::now();

    std::vector<float> output_data;
    uint8_t* input = interpreter_->typed_input_tensor<uint8_t>(0);
    std::memcpy(input, input_data.data(), input_data.size());

    interpreter_->Invoke();

    const float* locations = TensorData<float>(*output_locations_, 0);
    const float* classes = TensorData<float>(*output_classes_, 0);
    const float* scores = TensorData<float>(*output_scores_, 0);
    const int num_detections = (int)*TensorData<float>(*num_detections_, 0);
    
    auto results = std::make_unique<std::vector<BoundingBox>>();

    for (auto i = 0; i < num_detections; i++)
    {
        if (scores[i] >= score_threshold_)
        {
            auto bounding_box = std::make_unique<BoundingBox>();
            auto y0 = locations[4 * i + 0];
            auto x0 = locations[4 * i + 1];
            auto y1 = locations[4 * i + 2];
            auto x1 = locations[4 * i + 3];

            
            bounding_box->class_id = (int)classes[i];
            bounding_box->scores = scores[i];
            bounding_box->x = x0;
            bounding_box->y = y0;
            bounding_box->width = x1 - x0;
            bounding_box->height = y1 - y0;
            bounding_box->center_x = bounding_box->x + (bounding_box->width / 2.0f);
            bounding_box->center_y = bounding_box->y + (bounding_box->height / 2.0f);

#if 0
            std::cout << "class_id: " << bounding_box->class_id << std::endl;
            std::cout << "scores  : " << bounding_box->scores << std::endl;
            std::cout << "x       : " << bounding_box->x << std::endl;
            std::cout << "y       : " << bounding_box->y << std::endl;
            std::cout << "width   : " << bounding_box->width << std::endl;
            std::cout << "height  : " << bounding_box->height << std::endl;
            std::cout << "center  : " << bounding_box->center_x << ", " << bounding_box->center_y << std::endl;
            std::cout << "y       : " << bounding_box->y << std::endl;
#endif
            results->emplace_back(std::move(*bounding_box));
        }
    }

    time_span =
        std::chrono::steady_clock::now() - start_time;

    return results;
}

const int ObjectDetector::Width() const
{
    return input_width_;
}

const int ObjectDetector::Height() const
{
    return input_height_;
}

const int ObjectDetector::Channels() const
{
    return input_channels_;
}

template<>
float* ObjectDetector::TensorData(TfLiteTensor& tensor, const int index)
{
    float* result = nullptr;
    auto nelems = 1;
    for (auto i = 1; i < tensor.dims->size; i++)
    {
        nelems *= tensor.dims->data[i];
    }

    switch (tensor.type)
    {
    case kTfLiteFloat32:
        result = tensor.data.f + nelems * index;
        break;
        std::cerr << "Unmatch tensor type." << std::endl;
    default:
        break;
    }
    return result;
}


template<>
uint8_t* ObjectDetector::TensorData(TfLiteTensor& tensor, const int index)
{
    uint8_t* result = nullptr;
    auto nelems = 1;
    for (auto i = 1; i < tensor.dims->size; i++)
    {
        nelems *= tensor.dims->data[i];
    }

    switch (tensor.type)
    {
    case kTfLiteUInt8:
        result = tensor.data.uint8 + nelems * index;
        break;
        std::cerr << "Unmatch tensor type." << std::endl;
    default:
        break;
    }
    return result;
}
