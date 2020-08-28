// edge_detection.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <iostream>

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

#include "cv-helpers.hpp"

#define CVUI_IMPLEMENTATION
#include "cvui.h"


int main() try
{
    // Define spatial filter (Edge-preserving)
    rs2::spatial_filter spatial;

    // Define temporal filter
    rs2::temporal_filter temporal;

    // Spatially align all streams to depth viewport
    rs2::align align_to(RS2_STREAM_DEPTH);
    
    // Declare RealSense pipline, encapsulating the actual device and sensors
    rs2::pipeline pipe;

    rs2::config cfg;

    // cfg.enable_stream(RS2_STREAM_DEPTH); // Enable default depth
    // For the color stream, set format to RGBA
    // To allow blending of the color frame on top of the depth frame
    // cfg.enable_stream(RS2_STREAM_COLOR, RS2_FORMAT_RGB8);
    auto profile = pipe.start(cfg);

    auto sensor = profile.get_device().first<rs2::depth_sensor>();

    // set the device to preset of the L515 cameras.
    if (sensor && sensor.is<rs2::depth_stereo_sensor>())
    {
        sensor.set_option(RS2_OPTION_VISUAL_PRESET, RS2_L500_VISUAL_PRESET_DEFAULT);
    }

    auto stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();

    // Create a simple OpenGL windows for rendering
    const cv::String window_name = "RealSense L515 Example";

    // Init cvui and tell it to create a OpenCV window, 
    cvui::init(window_name);

    // The width of all trackbars used in this example.
    int width = 300;
    
    double canny_threshold1 = 50.0;
    double canny_threshold2 = 50.0;
    double hough_lines_threshold = 30.0;
    double hough_lines_min = 50.0;
    double hough_lines_gap = 10.0;

    while (cv::getWindowProperty(window_name, cv::WND_PROP_AUTOSIZE) >= 0)
    {
        // Wait for the next set of frames
        auto frames = pipe.wait_for_frames();
        // First make the frames sptially aligned
        frames = align_to.process(frames);
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

        auto color_image = frame_to_mat(color_frame);
        auto image = frame_to_mat(depth_frame);
        cv::Mat gaussian_im, laplacian_im, temp_im, dst_im, gray_im;
        std::vector<cv::Vec4i> lines;

        image.convertTo(gray_im, CV_8UC1, -255.0 / 10000.0, 255.0); // Scaling
        //cv::GaussianBlur(gray_im, gaussian_im, cv::Size(5, 5), 1.0);
        cv::Canny(gray_im, temp_im, std::floor(canny_threshold1), std::floor(canny_threshold2));
        cv::HoughLinesP(temp_im, lines, 1, CV_PI / 180,
            std::floor(hough_lines_threshold), std::floor(hough_lines_min), std::floor(hough_lines_gap));
        cv::cvtColor(temp_im, dst_im, cv::COLOR_GRAY2BGR);

        for (size_t i = 0; i < lines.size(); i++)
        {
            auto line = lines[i];

            cv::line(dst_im, cv::Point(line[0], line[1]),
                cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }

        cv::Rect roi_rect;
        cv::Mat combined_img(cv::Size(dst_im.cols * 2, dst_im.rows), CV_8UC3);
        roi_rect.width = dst_im.cols;
        roi_rect.height = dst_im.rows;
        cv::Mat roi1(combined_img, roi_rect);
        dst_im.copyTo(roi1);

        roi_rect.x = dst_im.cols;
        cv::Mat roi2(combined_img, roi_rect);
        color_image.copyTo(roi2);

        cvui::text(combined_img, 20, 10, "HoughLinesP, threshold");
        cvui::trackbar(combined_img, 10, 20, width, &hough_lines_threshold, (double)0., (double)200., 1, "%.0Lf");

        cvui::text(combined_img, 20, 70, "HoughLinesP, min");
        cvui::trackbar(combined_img, 10, 80, width, &hough_lines_min, (double)0., (double)200., 1, "%.0Lf");

        cvui::text(combined_img, 20, 130, "HoughLinesP, gap");
        cvui::trackbar(combined_img, 10, 140, width, &hough_lines_gap, (double)0., (double)200., 1, "%.0Lf");


        cvui::update();

        cv::imshow(window_name, combined_img);
        // Handle the keyboard before moving to the next frame
        const int key = cv::waitKey(1);
        if (key == 27)
        {
            break;  // Escape
        }
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error& e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)

{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
