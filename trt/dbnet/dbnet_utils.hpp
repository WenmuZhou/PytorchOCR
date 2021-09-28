#ifndef DBNET_UTILS_HPP
#define DBNET_UTILS_HPP
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
typedef Point3_<float> Pixel;

const float mean1[3] = { 103.53, 116.28, 123.675 };
const float std1[3] = { 57.375, 57.12, 58.395 };

cv::Mat Array2Mat(float* array, std::vector<int64_t> t);
void _normalize(cv::Mat& img);
void normalize(Pixel &pixel);
void convertMat2pointer(cv::Mat& img, float* x);
//void convertMat2pointer(cv::Mat& img, float* x);
int sign(float x);

#endif