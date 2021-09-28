#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include "cuda_utils.h"
#include "logging.h"

#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include "dbnet_utils.hpp"
#include "utils.h"
#include "calibrator.h"
#include "clipper.hpp"
#include "Config.h"


static int INPUT_H = SHORT_INPUT;
static int INPUT_W = SHORT_INPUT;
static int OUTPUT_S = INPUT_W * INPUT_H;
const char* INPUT_BLOB_NAME = "input_1";
const char* OUTPUT_BLOB_NAME = "output_1";
static Logger gLogger;

using namespace nvinfer1;
using namespace ClipperLib;

void get_coordinates_of_rotated_box(cv::RotatedRect _rotated_box, int _height, int _width, cv::Point2f (&order_rect)[4])
{
    float center_x = _rotated_box.center.x;
    float center_y = _rotated_box.center.y;

    cv::RotatedRect tmp_rect = RotatedRect(Point2f(center_x * _width,center_y * _height),
    Size2f(_rotated_box.size.width * _width,_rotated_box.size.height * _height), _rotated_box.angle);
    tmp_rect.points(order_rect);
    return;
}

float paddimg(cv::Mat& In_Out_img, int shortsize = 960) {
    int w = In_Out_img.cols;
    int h = In_Out_img.rows;
    float scale = 1.f;
    if (w < h) {
        scale = (float)shortsize / w;
        h = scale * h;
        w = shortsize;
    }
    else {
        scale = (float)shortsize / h;
        w = scale * w;
        h = shortsize;
    }

    if (h % 32 != 0) {
        h = (h / 32 + 1) * 32;
    }
    if (w % 32 != 0) {
        w = (w / 32 + 1) * 32;
    }

    cv::resize(In_Out_img, In_Out_img, cv::Size(w, h));
    return scale;
}


cv::RotatedRect get_min_area_bbox(cv::Mat _image, cv::Mat _contour, float _scale_ratio=1.0)
{
    int _image_W = _image.rows;
    int _image_H = _image.cols;
    cv::Mat scaled_contour;
    if (abs(_scale_ratio -1) > 0.001)
    {
        cv::Mat reshaped_contour = _contour.reshape(-1, 2);
        float _contour_area = contourArea(_contour, false);
        float _contour_length = arcLength(_contour, true);
        float distance = _contour_area * _scale_ratio / _contour_length;
        ClipperOffset offset = ClipperOffset();
        offset.AddPath(reshaped_contour, jtRound, etClosedPolygon);

        Paths box;
        offset.Execute(box, distance);
        if (box.size() == 0 || box.size() > 1)
            return cv::RotatedRect();
        //scaled_contour = np.array(box).reshape(-1, 1, 2);
    }
    else
        scaled_contour = _contour;

    cv::RotatedRect rotated_box = cv::minAreaRect(scaled_contour);

    float to_rotate_degree = 0;
    float bbox_height = 0, bbox_width = 0;
    if (rotated_box.angle >=-90 && rotated_box.angle <= -45)
    {
        to_rotate_degree = rotated_box.angle + 90;
        bbox_height  = rotated_box.size.width;
        bbox_width = rotated_box.size.height;
    }
    else
    {
        to_rotate_degree = rotated_box.angle;
        bbox_width  = rotated_box.size.width;
        bbox_height = rotated_box.size.height;
    }

    cv::RotatedRect rotated_box_normal = cv::RotatedRect(Point2f(rotated_box.center.x / _image_W,rotated_box.center.y / _image_H),
    Size2f(bbox_width / _image_W,bbox_height / _image_H), to_rotate_degree);
    return rotated_box_normal;
}

float get_box_score(float* map, cv::Point2f rect[], int width, int height,
                    float threshold)
{

    int xmin = width - 1;
    int ymin = height - 1;
    int xmax = 0;
    int ymax = 0;

    for (int j = 0; j < 4; j++) {
        if (rect[j].x < xmin) {
            xmin = rect[j].x;
        }
        if (rect[j].y < ymin) {
            ymin = rect[j].y;
        }
        if (rect[j].x > xmax) {
            xmax = rect[j].x;
        }
        if (rect[j].y > ymax) {
            ymax = rect[j].y;
        }
    }
    float sum = 0;
    int num = 0;
    for (int i = ymin; i <= ymax; i++) {
        for (int j = xmin; j <= xmax; j++) {
            if (map[i * width + j] > threshold) {
                sum = sum + map[i * width + j];
                num++;
            }
        }
    }

    return sum / num;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string& onnx_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    if (!network)
    {
        return;
    }

    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser)
    {
        return;
    }

    auto parsed = parser->parseFromFile(onnx_name.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return;
    }

    //builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));
    config->setFlag(BuilderFlag::kFP16);

    //samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMIN, Dims4(1,3,640,640));
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kOPT, Dims4(1,3,1280,768));
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMAX, Dims4(4,3,1280,1280));
    config->addOptimizationProfile(profile);

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;
    engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_S * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

bool parse_args(int argc, char** argv, std::string& onnx, std::string& engine, std::string& img_dir) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s") {
        onnx = std::string(argv[2]);
        engine = std::string(argv[3]);
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    std::string onnx_name = "";
    std::string engine_name = "";

    std::string img_dir;
    if (!parse_args(argc, argv, onnx_name, engine_name, img_dir)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./litehrnet -s [.onnx] [.engine]  // serialize model to plan file" << std::endl;
        std::cerr << "./litehrnet -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    if (!onnx_name.empty()) {
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream, onnx_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }


    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    float *data = new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    float *prob = new float[BATCH_SIZE * OUTPUT_S];
    

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    context->setBindingDimensions(0, Dims4(1,3,640,640));

    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_S * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int fcount = 0;
    for (int f = 0; f < (int)file_names.size(); f++) {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;

        cv::Mat tmpimg = cv::imread(img_dir + "/" + file_names[f - fcount + 1]);
        if (tmpimg.empty()) continue;
        resize(tmpimg,tmpimg,Size(INPUT_H, INPUT_W),0,0, INTER_LINEAR);
        cv::Mat src_img = tmpimg.clone();
        cv::imshow("a", tmpimg);
        float scale = paddimg(tmpimg, SHORT_INPUT); // resize the image
        std::cout << "letterbox shape: " << tmpimg.cols << ", " << tmpimg.rows << std::endl;

        tmpimg.convertTo(tmpimg, CV_32F);

         tmpimg.forEach<Pixel>([](Pixel &pixel, const int * position) -> void
         {
             normalize(pixel);
         });
//        _normalize(tmpimg);
        convertMat2pointer(tmpimg, data);

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        cv::Mat map = cv::Mat::zeros(cv::Size(tmpimg.cols, tmpimg.rows), CV_8UC1);
        for (int h = 0; h < tmpimg.rows; ++h) {
            uchar *ptr = map.ptr(h);
            for (int w = 0; w < tmpimg.cols; ++w) {
                ptr[w] = (prob[h * tmpimg.cols + w] > 0.3) ? 255 : 0;
            }
        }
        cv::imshow("mask", map);
        cv::waitKey(0);
        // Extracting minimum circumscribed rectangle
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarcy;
        cv::findContours(map, contours, hierarcy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect> boundRect(contours.size());
        std::vector<cv::RotatedRect> box(contours.size());

        cv::Point2f order_rect[4];

        for (int i = 0; i < contours.size(); i++) {
            if (contours[i].size() < 4 && cv::contourArea(contours[i]) < 16)
            {
                std::cout << "area too small" <<  std::endl;
                continue;
            }
            cv::RotatedRect rotated_rect = get_min_area_bbox(tmpimg, cv::Mat(contours[i]));
            get_coordinates_of_rotated_box(rotated_rect, src_img.cols, src_img.rows, order_rect);

//            float score = get_box_score(prob, rect, src_img.cols, src_img.rows,
//                                        SCORE_THRESHOLD);

            for (int i = 0; i < 4; i++)
                cv::line(src_img, cv::Point(order_rect[i].x,order_rect[i].y), cv::Point(order_rect[(i+1)%4].x,order_rect[(i+1)%4].y), cv::Scalar(0, 255, 255), 2, 8);
            cv::rectangle(src_img, cv::Point(order_rect[0].x,order_rect[0].y), cv::Point(order_rect[2].x,order_rect[2].y), cv::Scalar(0, 0, 255), 2, 8);
        }

        std::cout << "row : " << tmpimg.rows << " los : " << tmpimg.cols << std::endl;

	    cv::imwrite("./dbnet_out/_" + file_names[f - fcount + 1], map);
	    cv::imwrite("./dbnet_out/__" + file_names[f - fcount + 1], src_img);
        fcount = 0;
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
