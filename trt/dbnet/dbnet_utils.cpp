#include "dbnet_utils.hpp"

cv::Mat Array2Mat(float* array, std::vector<int64_t> t){
	int row = t[0];
	int col = t[1];
	if (t.size() == 2) {
		cv::Mat img(row, col, CV_32F);
		for (int i = 0; i < img.rows; ++i) {
			float* cur_row = img.ptr<float>(i);
			for (int j = 0; j < img.cols; ++j) {
				*cur_row++ = array[i * col + j];
			}
		}
		return img;
	}
	else {
		int sizes[] = { t[0],t[1],t[2],t[3] };
		cv::Mat img = cv::Mat(4, sizes, CV_32F, array);
		return img;
	}
}


void _normalize(cv::Mat& img){
	img.convertTo(img, CV_32F);
	for (int i = 0; i < img.rows; ++i) {
		float* cur_row = img.ptr<float>(i);
		for (int j = 0; j < img.cols; ++j) {
			*cur_row++ = (*cur_row - mean1[0]) / std1[0];
			*cur_row++ = (*cur_row - mean1[1]) / std1[1];
			*cur_row++ = (*cur_row - mean1[2]) / std1[2];
		}
	}
}

void normalize(Pixel &pixel){
    pixel.x = (pixel.x - mean1[0]) / std1[0];
    pixel.y = (pixel.y - mean1[1]) / std1[1];
    pixel.z = (pixel.z - mean1[2]) / std1[2];
}

void convertMat2pointer(cv::Mat& img, float* x){
	for (int i = 0; i < img.rows; ++i) {

		float* cur_row = img.ptr<float>(i);
		for (int j = 0; j < img.cols; ++j) {
            x[img.rows * img.cols + i * img.cols + j] = *cur_row++;
            
            
            x[img.rows * img.cols * 2 + i * img.cols + j] = *cur_row++;
		
	    x[i * img.cols + j] = *cur_row++;
            //std::cout << x[i * img.cols + j] << std::endl;
            //std::cout << x[img.rows * img.cols + i * img.cols + j] << std::endl;
            //std::cout << x[img.rows * img.cols * 2 + i * img.cols + j] << std::endl;
		}
	}
}


int sign(float x) {
	int w = 0;
	if (x > 0) {
		w = 1;
	}
	else if (x == 0) {
		w = 0;
	}
	else {
		w = -1;
	}
	return w;
}
