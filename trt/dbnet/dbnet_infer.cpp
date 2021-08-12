#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include "cuda_utils.h"
#include "logging.h"
#include "dbnet_utils.hpp"
#include "utils.h"
#include "calibrator.h"
#include "clipper.hpp"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1
#define EXPANDRATIO 1.5
#define BOX_MINI_SIZE 5
#define SCORE_THRESHOLD 0.3
#define BOX_THRESHOLD 0.7

static int SHORT_INPUT = 640;
static int INPUT_H = 640;
static int INPUT_W = 640;
static int OUTPUT_S = 409600;
const char* INPUT_BLOB_NAME = "input_1";
const char* OUTPUT_BLOB_NAME = "output_1";
static Logger gLogger;

using namespace nvinfer1;


cv::RotatedRect expandBox(cv::Point2f temp[], float ratio)
{
    ClipperLib::Path path = {
        {ClipperLib::cInt(temp[0].x), ClipperLib::cInt(temp[0].y)},
        {ClipperLib::cInt(temp[1].x), ClipperLib::cInt(temp[1].y)},
        {ClipperLib::cInt(temp[2].x), ClipperLib::cInt(temp[2].y)},
        {ClipperLib::cInt(temp[3].x), ClipperLib::cInt(temp[3].y)}};
    double area = ClipperLib::Area(path);
    double distance;
    double length = 0.0;
    for (int i = 0; i < 4; i++) {
        length = length + sqrtf(powf((temp[i].x - temp[(i + 1) % 4].x), 2) +
                                powf((temp[i].y - temp[(i + 1) % 4].y), 2));
    }

    distance = area * ratio / length;

    ClipperLib::ClipperOffset offset;
    offset.AddPath(path, ClipperLib::JoinType::jtRound,
                   ClipperLib::EndType::etClosedPolygon);
    ClipperLib::Paths paths;
    offset.Execute(paths, distance);

    std::vector<cv::Point> contour;
    for (int i = 0; i < paths[0].size(); i++) {
        contour.emplace_back(paths[0][i].X, paths[0][i].Y);
    }
    offset.Clear();
    return cv::minAreaRect(contour);
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


bool get_mini_boxes(cv::RotatedRect& rotated_rect, cv::Point2f rect[],
                    int min_size)
{

    cv::Point2f temp_rect[4];
    rotated_rect.points(temp_rect);
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (temp_rect[i].x > temp_rect[j].x) {
                cv::Point2f temp;
                temp = temp_rect[i];
                temp_rect[i] = temp_rect[j];
                temp_rect[j] = temp;
            }
        }
    }
    int index0 = 0;
    int index1 = 1;
    int index2 = 2;
    int index3 = 3;
    if (temp_rect[1].y > temp_rect[0].y) {
        index0 = 0;
        index3 = 1;
    } else {
        index0 = 1;
        index3 = 0;
    }
    if (temp_rect[3].y > temp_rect[2].y) {
        index1 = 2;
        index2 = 3;
    } else {
        index1 = 3;
        index2 = 2;
    }

    rect[0] = temp_rect[index0];  // Left top coordinate
    rect[1] = temp_rect[index1];  // Left bottom coordinate
    rect[2] = temp_rect[index2];  // Right bottom coordinate
    rect[3] = temp_rect[index3];  // Right top coordinate

    if (rotated_rect.size.width < min_size ||
        rotated_rect.size.height < min_size) {
        return false;
    } else {
        return true;
    }
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

    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));
    config->setFlag(BuilderFlag::kFP16);

    //samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

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
    float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    float prob[BATCH_SIZE * OUTPUT_S];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
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
        //cv::cvtColor(tmpimg, tmpimg, cv::COLOR_BGR2RGB);
        resize(tmpimg,tmpimg,Size(INPUT_H, INPUT_W),0,0, INTER_LINEAR);
        cv::Mat src_img = tmpimg.clone();
        float scale = paddimg(tmpimg, SHORT_INPUT); // resize the image
        std::cout << "letterbox shape: " << tmpimg.cols << ", " << tmpimg.rows << std::endl;
        //if (tmpimg.cols < MIN_INPUT_SIZE || tmpimg.rows < MIN_INPUT_SIZE) continue;

        _normalize(tmpimg);

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

        // Extracting minimum circumscribed rectangle
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarcy;
        cv::findContours(map, contours, hierarcy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect> boundRect(contours.size());
        std::vector<cv::RotatedRect> box(contours.size());
        cv::Point2f rect[4];
        cv::Point2f order_rect[4];

        for (int i = 0; i < contours.size(); i++) {
            cv::RotatedRect rotated_rect = cv::minAreaRect(cv::Mat(contours[i]));
            if (!get_mini_boxes(rotated_rect, rect, BOX_MINI_SIZE)) {
                std::cout << "box too small" <<  std::endl;
                continue;
            }

            // drop low score boxes
            float score = get_box_score(prob, rect, tmpimg.cols, tmpimg.rows,
                                        SCORE_THRESHOLD);
            if (score < BOX_THRESHOLD) {
                std::cout << "score too low =  " << score << ", threshold = " << BOX_THRESHOLD <<  std::endl;
                continue;
            }

            // Scaling the predict boxes depend on EXPANDRATIO
            cv::RotatedRect expandbox = expandBox(rect, EXPANDRATIO);
            expandbox.points(rect);
            if (!get_mini_boxes(expandbox, rect, BOX_MINI_SIZE + 2)) {
                continue;
            }

            // Restore the coordinates to the original image
            for (int k = 0; k < 4; k++) {
                order_rect[k] = rect[k];
                order_rect[k].x = int(order_rect[k].x / tmpimg.cols * src_img.cols);
                order_rect[k].y = int(order_rect[k].y / tmpimg.rows * src_img.rows);
            }

            cv::rectangle(src_img, cv::Point(order_rect[0].x,order_rect[0].y), cv::Point(order_rect[2].x,order_rect[2].y), cv::Scalar(0, 0, 255), 2, 8);
            //std::cout << "After LT =  " << order_rect[0] << ", After RD = " << order_rect[2] <<  std::endl;
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
