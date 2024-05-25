#include "Classifier.h"

void Classifier::read_model(std::string modelpath)
{
	//.onnx���f����ǂݍ���
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

	//tensor�摜�ɕϊ�
	cv::Mat blob = cv::dnn::blobFromImage(src, 1 / 255.0, this->input_shape, true);

	//tensor�摜�����f���ɓn��
	this->net.setInput(blob);

	//forward
	cv::Mat outs = this->net.forward();

	//�o�͂�0~1�ɓ]��
	this->softmax(outs, outs);

	//�M���x���������ʂ����
	cv:: Point classIdPoint;
	double confidence;
	minMaxLoc(outs.reshape(1, 1), 0, &confidence, 0, &classIdPoint);

	//���_�ł���classId
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