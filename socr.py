
import fastdeploy as fd
import cv2
import os
import numpy as np

def build_option():
    option = fd.RuntimeOption()
    # option.use_gpu(0)
    option.set_cpu_thread_num(9)
    return option
# Detection模型, 检测文字框
det_model_file = os.path.join('./models/ch_PP-OCRv3_det_infer', "inference.pdmodel")
det_params_file = os.path.join('./models/ch_PP-OCRv3_det_infer', "inference.pdiparams")
# Classification模型，方向分类，可选
cls_model_file = os.path.join('./models/ch_ppocr_mobile_v2.0_cls_infer', "inference.pdmodel")
cls_params_file = os.path.join('./models/ch_ppocr_mobile_v2.0_cls_infer', "inference.pdiparams")
# Recognition模型，文字识别模型
rec_model_file = os.path.join('./models/ch_PP-OCRv3_rec_infer', "inference.pdmodel")
rec_params_file = os.path.join('./models/ch_PP-OCRv3_rec_infer', "inference.pdiparams")
rec_label_file = './models/ppocr_keys_v1.txt'

# 对于三个模型，均采用同样的部署配置
# 用户也可根据自行需求分别配置
runtime_option = build_option()

# PPOCR的cls和rec模型现在已经支持推理一个Batch的数据
# 定义下面两个变量后, 可用于设置trt输入shape, 并在PPOCR模型初始化后, 完成Batch推理设置
cls_batch_size = 1
rec_batch_size = 6

# 当使用TRT时，分别给三个模型的runtime设置动态shape,并完成模型的创建.
# 注意: 需要在检测模型创建完成后，再设置分类模型的动态输入并创建分类模型, 识别模型同理.
# 如果用户想要自己改动检测模型的输入shape, 我们建议用户把检测模型的长和高设置为32的倍数.
det_option = runtime_option
det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                               [1, 3, 960, 960])
# 用户可以把TRT引擎文件保存至本地
# det_option.set_trt_cache_file(args.det_model  + "/det_trt_cache.trt")
det_model = fd.vision.ocr.DBDetector(
    det_model_file, det_params_file, runtime_option=det_option)

cls_option = runtime_option
cls_option.set_trt_input_shape("x", [1, 3, 48, 10],
                               [cls_batch_size, 3, 48, 320],
                               [cls_batch_size, 3, 48, 1024])
# 用户可以把TRT引擎文件保存至本地
# cls_option.set_trt_cache_file(args.cls_model  + "/cls_trt_cache.trt")
cls_model = fd.vision.ocr.Classifier(
    cls_model_file, cls_params_file, runtime_option=cls_option)

rec_option = runtime_option
rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
                               [rec_batch_size, 3, 48, 320],
                               [rec_batch_size, 3, 48, 2304])
# 用户可以把TRT引擎文件保存至本地
# rec_option.set_trt_cache_file(args.rec_model  + "/rec_trt_cache.trt")
rec_model = fd.vision.ocr.Recognizer(
    rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)
# 创建PP-OCR，串联3个模型，其中cls_model可选，如无需求，可设置为None
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model)
ppocr_v3.cls_batch_size = cls_batch_size
ppocr_v3.rec_batch_size = rec_batch_size
# # 预测图片准备
# im = cv2.imread('14.jpg')
# #预测并打印结果
# result = ppocr_v3.predict(im)
# print(result.text)
# Take in base64 string and return PIL image
from PIL import Image
import base64,re
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    import io
    img= Image.open(io.BytesIO(imgdata)).convert('RGB')
    w,h=img.size
    # print(w,h)
    n=2000/h
    img= img.resize((int(w*n),int(h*n)))
    # img.save('1.jpg')
    return img
# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
def ocr(img_b64):
    im=toRGB(stringToImage(img_b64))
    result = ppocr_v3.predict(im).text
    # print(result)
    # ocr_str= ''.join(result)上川上加教育部审定义务教育教科书2013英语三年级起点六年级下班福建教育出版社
    # re.sub('[xyz]', '1', s)
    # det.replace(' ', '').replace('(', '').replace(')', '').replace(r'上[加班川]', '上册').replace(r'下[加班川]', '下册')
    ocr_str=''.join([det for det in result])
    ocr_str=re.sub('[\ ()]', '',  ocr_str)
    ocr_str=re.sub(' ', '',  ocr_str)
    ocr_str=re.sub('[匠]', '五',  ocr_str)
    ocr_str=re.sub('上[加班川期科哥]', '上册',  ocr_str)
    ocr_str=re.sub('下[加班川期用]', '下册',  ocr_str)
    ocr_str=re.sub('第二学期', '下册',  ocr_str)
    ocr_str=re.sub('上学期', '上册',  ocr_str)
    ocr_str=re.sub('下学期', '下册',  ocr_str)
    ocr_str=re.sub('第一学期', '上册',  ocr_str)
    ocr_str=re.sub('第→册', '第一册',  ocr_str)

    r1 = re.findall('\w年级\w册', ocr_str)
    # print(ocr_str,r1,len(r1))
    if len(r1)>0:
        return r1[0],ocr_str
    r2 = re.findall('第\w册', ocr_str)
    # print(ocr_str,r2,len(r2))
    if len(r2)>0:
        return r2[0],ocr_str    
    r3 = re.findall('\w年级', ocr_str)
    r4 = re.findall('\w册', ocr_str)
    # print(ocr_str,r2,len(r2))
    if len(r3)>0 and len(r4)>0:
        return r3[0]+r4[0],ocr_str
    return 'nopattern',ocr_str