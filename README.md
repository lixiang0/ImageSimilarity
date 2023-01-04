# ImageSimilarity
查找最相似图片

- inference.ipynb：基于resnet(图片embedding)+faiss（图片检索）方案
- inference1.ipynb：基于resnet(图片embedding)+faiss（图片检索）+百度的paddleocr（OCR）方案

# 环境

```
pip install scikit-learn numpy matplotlib pillow
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
pip install faiss-cpu
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```
