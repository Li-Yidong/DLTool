#include <iostream>
#include "Classifier.h"

int main()
{
    //*************************************************************
    // 
    // This project is a inference example of CNN .onnx model
    // We use opencv library to inference model
    // 
    //*************************************************************

    // read image
    cv::Mat src = cv::imread("");

    //Declare parameter
    std::vector<std::string> classes = { "dog", "cat" };
    Classifier Classifier_;

    // load model
    std::string model_path = "";
    Classifier_.read_model(model_path);
    
    // run 
    int classID = Classifier_.detect(src);

    // show result
    std::cout << "Detection result is: " << classes[classID] << std::endl;
}
