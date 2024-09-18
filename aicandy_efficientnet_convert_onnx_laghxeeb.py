"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
from aicandy_model_src_ybklivqn.aicandy_efficientnet_model_mqialtec import CustomEfficientNet
import sys
import argparse


# python aicandy_efficientnet_convert_onnx_laghxeeb.py --model_path aicandy_model_out_vailsuom/aicandy_model_pth_ppxascdt.pth --onnx_path aicandy_model_out_vailsuom/aicandy_model_onnx_expbpybk.onnx --num_classes 2

def convert_to_onnx(model_path, onnx_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = CustomEfficientNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Input tensor
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Convert to ONNX
    torch.onnx.export(model,               # mô hình đang chạy
                      dummy_input,         # input tensor
                      onnx_path,           # nơi lưu file onnx 
                      export_params=True,  # lưu các trọng số đã training
                      opset_version=11,    # phiên bản ONNX
                      do_constant_folding=True,  # tối ưu hoá mô hình
                      input_names = ['input'],   # tên của input
                      output_names = ['output'], # tên của output
                      dynamic_axes={'input' : {0 : 'batch_size'},    # kích thước batch có thể thay đổi
                                    'output' : {0 : 'batch_size'}})

    print(f"Model đã được chuyển đổi và lưu tại {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the PyTorch model (.pth)')
    parser.add_argument('--onnx_path', type=str, default='model.onnx', help='Path to save the ONNX model')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the model')

    args = parser.parse_args()

    convert_to_onnx(args.model_path, args.onnx_path, args.num_classes)