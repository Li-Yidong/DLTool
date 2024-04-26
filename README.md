# <center>Deep learning tool!

## 1. Intruduction
This project is tending to make deep learning easy learning and fast deployment!

## 2. Documents
**・Guide book**: [EN](doc/README/GuideBook_EN.md) | [JP](doc/README/GuideBook_JP.md)  
**・Build enviroment**: [EN](doc/README/installation_en.md) | [JP](doc/README/installation.md)  
**・UI configuration**: [EN](doc/README/readme_ui_config_en.md) | [JP](doc/README/readme_ui_config.md)  
**・Configuration detials** : [EN](doc/config_doc_en.md) | [JP](doc/config_doc.md)  

## 3. Quick start
3.1 Install Pytorch from [here](https://pytorch.org/)  

3.2 Install the other packages by runing this command:   
`pip install -r requirements.txt`  

3.3 Run application by runing this command:   
`python main_gui.py`  

3.4 Adjust gui by your customed dataset and have fun!

## 4. Something new!!
### 4.1 Heatmap viewer
Now you can see the heatmap of the model prediction just on the GUI!(This heatmap is made by the full connection layer of the model)

You just need to select `Heatmap Viewer` in `Mode`, and select fill the configuration items in the `Heatmap Viewer`, the press the `Run` button. There will be a `Heatmap Viewer` window pops up and you can check every heatmap of each image in the folder you just selected.

### 4.2 Export inference
Because `ONNX` models will not saved when you train or prune the model, so a new function is added to export the inference model to `ONNX` format.

## 5. Reference
This project is based on Pytorch  
Thanks **Naufal** for the gui application development!