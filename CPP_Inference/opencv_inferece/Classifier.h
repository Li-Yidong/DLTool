#pragma once
#include <opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

class Classifier
{
public:

	/*
	* @brief model 読み込み
	*
	* @param[string] modelpath モデルのパス
	*
	*/
	void read_model(std::string modelpath);

	/**
	 * @brief CNNモデルを用いて分類する関数である.
	 * @param src ソース画像
	 * @return 可能性が一番高いClassのインデックス
	*/
	int detect(cv::Mat src);

private:

	cv::Size input_shape = cv::Size(224, 224);
	cv::dnn::Net net;
	
	void softmax(cv::Mat& src, cv::Mat& dst);
};

