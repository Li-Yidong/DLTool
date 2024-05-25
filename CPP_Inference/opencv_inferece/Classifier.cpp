#include "Classifier.h"

void Classifier::read_model(std::string modelpath)
{
	//.onnxモデルを読み込む
	this->net = cv::dnn::readNetFromONNX(modelpath);

	if (this->net.empty())
	{
		std::cout << "Load model failed!!" << std::endl;
	}
	else
	{
		std::cout << "Load model successed!!" << std::endl;
	}

}

int Classifier::detect(cv::Mat src)
{
	bool is_NG = false;

	//tensor画像に変換
	cv::Mat blob = cv::dnn::blobFromImage(src, 1 / 255.0, this->input_shape, true);

	//tensor画像をモデルに渡す
	this->net.setInput(blob);

	//forward
	cv::Mat outs = this->net.forward();

	//出力を0~1に転換
	this->softmax(outs, outs);

	//信頼度が高い結果を取る
	cv:: Point classIdPoint;
	double confidence;
	minMaxLoc(outs.reshape(1, 1), 0, &confidence, 0, &classIdPoint);

	//推論できたclassId
	int classId = classIdPoint.x;

	return classId;
}

void Classifier::softmax(cv::Mat& src, cv::Mat& dst) {
	float max = 0.0;
	float sum = 0.0;

	max = *std::max_element(src.begin<float>(), src.end<float>());
	exp((src - max), dst);
	sum = cv::sum(dst)[0];
	dst /= sum;

	return;
}