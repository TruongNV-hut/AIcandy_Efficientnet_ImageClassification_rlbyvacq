# EfficientNet and Image Classification

<p align="justify">
<strong>EfficientNet</strong> is a family of convolutional neural networks (CNNs) developed by Google that aim to achieve high accuracy with fewer computational resources. It uses a novel scaling method called compound scaling, which uniformly scales the network's depth, width, and resolution to balance performance and efficiency. EfficientNet models, ranging from EfficientNet-B0 to EfficientNet-B7, demonstrate state-of-the-art performance on various image classification tasks while being significantly more efficient than previous models like ResNet and Inception. The base model, EfficientNet-B0, is designed using neural architecture search (NAS), optimizing for both accuracy and computational cost..
</p>

## Image Classification
<p align="justify">
<strong>Image classification</strong> is a fundamental problem in computer vision where the goal is to assign a label or category to an image based on its content. This task is critical for a variety of applications, including medical imaging, autonomous vehicles, content-based image retrieval, and social media tagging.
</p>


## ❤️❤️❤️


```bash
If you find this project useful, please give it a star to show your support and help others discover it!
```

## Getting Started

### Clone the Repository

To get started with this project, clone the repository using the following command:

```bash
git clone https://github.com/TruongNV-hut/AIcandy_Efficientnet_ImageClassification_rlbyvacq.git
```

### Install Dependencies
Before running the scripts, you need to install the required libraries. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the model, use the following command:

```bash
python aicandy_efficientnet_train_xcprktlu.py --train_dir ../dataset --num_epochs 10 --batch_size 32 --model_path aicandy_model_out_vailsuom/aicandy_model_pth_ppxascdt.pth
```

### Testing the Model

After training, you can test the model using:

```bash
python aicandy_efficientnet_test_gmbpqtln.py --image_path ../image_test.jpg --model_path aicandy_model_out_vailsuom/aicandy_model_pth_ppxascdt.pth --label_path label.txt
```

### Converting to ONNX Format

To convert the model to ONNX format, run:

```bash
python aicandy_efficientnet_convert_onnx_laghxeeb.py --model_path aicandy_model_out_vailsuom/aicandy_model_pth_ppxascdt.pth --onnx_path aicandy_model_out_vailsuom/aicandy_model_onnx_expbpybk.onnx --num_classes 2
```

### More Information

To learn more about this project, [see here](https://aicandy.vn/ung-dung-mang-efficientnet-vao-phan-loai-hinh-anh).

To learn more about knowledge and real-world projects on Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL), visit the website [aicandy.vn](https://aicandy.vn/).

❤️❤️❤️




