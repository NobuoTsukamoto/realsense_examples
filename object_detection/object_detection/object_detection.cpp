// object_detection.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

#include "cv-helpers.hpp"
#include "object_detector.h"

const cv::String kKeys =
    "{help h usage ? |    | show help command.}"
    "{n thread       |2   | num of thread to set tf-lite interpreter.}"
    "{s score        |0.5 | score threshold.}"
    "{l label        |.   | path to label file.}"
    "{@input         |    | path to tf-lite model file.}"
    ;

const cv::String kWindowName = "Object detection example.";

const cv::Scalar kWhiteColor = cv::Scalar(246, 250, 250);
const cv::Scalar kBuleColor = cv::Scalar(255, 209, 0);


std::unique_ptr<std::vector<std::string>> ReadLabelFile(const std::string& label_path)
{
    auto labels = std::make_unique<std::vector<std::string>>();

    std::ifstream ifs(label_path);
    if (ifs.is_open())
    {
        std::string label = "";
        while (std::getline(ifs, label))
        {
            if (!label.empty())
            {

                labels->emplace_back(label);
            }
        }
    }
    else
    {
        std::cout << "Label file not found. : " << label_path << std::endl;
    }
    return labels;
}

void DrawCaption(
    cv::Mat& im,
    const cv::Point& point,
    const std::string& caption)
{
    cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
}

int main(int argc, char* argv[]) try
{
    // Argument parsing
    cv::String model_path;
    cv::CommandLineParser parser(argc, argv, kKeys);
    if (parser.has("h"))
    {
        parser.printMessage();
        return 0;
    }
    auto num_of_threads = parser.get<unsigned int>("thread");
    auto score_threshold = parser.get<float>("score");
    auto label_path = parser.get<cv::String>("label");
    if (parser.has("@input"))
    {
        model_path = parser.get<cv::String>("@input");
    }
    else
    {
        std::cout << "No model file path." << std::endl;
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }
    std::cout << "model path      : " << model_path << std::endl;
    std::cout << "label path      : " << label_path << std::endl;
    std::cout << "threads         : " << num_of_threads << std::endl;
    std::cout << "score threshold : " << score_threshold << std::endl;

    // Window setting
    cv::namedWindow(kWindowName,
        cv::WINDOW_GUI_NORMAL | cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
    cv::moveWindow(kWindowName, 100, 100);

    // Create Object detector
    auto detector = std::make_unique<ObjectDetector>(score_threshold);

    detector->BuildInterpreter(model_path, num_of_threads);
    auto width = detector->Width();
    auto height = detector->Height();

    // Load label file
    auto labels = ReadLabelFile(label_path);

    // Realsense settings
    rs2::spatial_filter spatial;                    // Define spatial filter (Edge-preserving)
    rs2::temporal_filter temporal;                  // Define temporal filter
    rs2::align align_to_depth(RS2_STREAM_DEPTH);    // Spatially align all streams to depth viewport
    rs2::align align_to_color(RS2_STREAM_COLOR);    // Spatially align all streams to color viewport

    // Declare RealSense pipline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_DEPTH); // Enable default depth
    // For the color stream, set format to RGBA
    // To allow blending of the color frame on top of the depth frame
    cfg.enable_stream(RS2_STREAM_COLOR, RS2_FORMAT_RGB8);
    
     auto profile = pipe.start(cfg);

   
    auto sensor = profile.get_device().first<rs2::depth_sensor>();

    // set the device to preset of the L515 cameras.
    if (sensor && sensor.is<rs2::depth_stereo_sensor>())
    {
        sensor.set_option(RS2_OPTION_VISUAL_PRESET, RS2_L500_VISUAL_PRESET_DEFAULT);
    }
    //auto stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();

    while (cv::getWindowProperty(kWindowName, cv::WND_PROP_AUTOSIZE) >= 0)
    {
        // Wait for the next set of frames
        auto frames = pipe.wait_for_frames();

        // First make the frames sptially aligned
        frames = align_to_color.process(frames);

        // Apply spatial filtering
        frames = spatial.process(frames);
        // Apply temporal filtering
        frames = temporal.process(frames);
        auto color_frame = frames.get_color_frame();
        auto depth_frame = frames.get_depth_frame();
        if (!color_frame || !depth_frame)
        {
            continue;
        }

        auto color_im = frame_to_mat(color_frame);

        // Create input data.
        // camera resolution  => input_im tensor size
        cv::Mat input_im;
        std::chrono::duration<double, std::milli> inference_time_span;

        cv::resize(color_im, input_im, cv::Size(width, height));
        std::vector<uint8_t> input_data(
            input_im.data, input_im.data + (input_im.cols * input_im.rows * input_im.elemSize()));

        // Run inference.
        auto result = detector->RunInference(input_data, inference_time_span);

        auto width = color_im.cols;
        auto height = color_im.rows;
        for (const auto& object : *result)
        {
            auto x = int(object.x * width);
            auto y = int(object.y * height);
            auto w = int(object.width * width);
            auto h = int(object.height * height);
            auto center_x = int(object.center_x * width);
            auto center_y = int(object.center_y * height);

            // Draw bounding box
            cv::Rect rect(x, y, w, h);
            cv::rectangle(color_im, rect, kBuleColor, 2);

            // Draw Caption
            std::ostringstream caption;
            if (labels->size() > object.class_id)
            {
                caption << (*labels)[object.class_id];
            }
            else
            {
                caption << std::to_string(object.class_id);
            }
            caption << " " << std::fixed << std::setprecision(2) << object.scores;
            DrawCaption(color_im, cv::Point(x, y), caption.str());

            // Draw Center
            auto distance = depth_frame.get_distance(center_x, center_y);
            if (distance > 0.0f)
            {
                // Center Point
                cv::circle(color_im, cv::Point(center_x, center_y), 7, kWhiteColor, -1);
                cv::circle(color_im, cv::Point(center_x, center_y), 7, kBuleColor, 2);

                std::ostringstream ss;
                ss << std::setprecision(2) << distance << " meters away";
                DrawCaption(color_im, cv::Point(center_x - 10, center_y - 10), ss.str());
            }


        }

        // Draw inference time.
        std::ostringstream elapsed_ms_ss;
        elapsed_ms_ss << std::fixed << std::setprecision(2) << inference_time_span.count() << " ms";
        DrawCaption(color_im, cv::Point(10, 60), elapsed_ms_ss.str());

        cv::imshow(kWindowName, color_im);
        // Handle the keyboard before moving to the next frame
        const int key = cv::waitKey(1);
        if (key == 27)
        {
            break;  // Escape
        }
    }
    return EXIT_SUCCESS;

}
catch (const cv::Exception& e)
{
    std::cerr << "OpenCV error calling :\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}