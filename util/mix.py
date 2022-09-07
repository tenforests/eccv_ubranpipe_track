#mix multiModel
import json
import torch

def output(output_dir, tmp):
    with open(str(output_dir), 'w') as f:
        json.dump(tmp, f)


def getInput(input_dir):
    with open(str(input_dir), 'r') as f:
        json_data = json.load(f)
    return json_data

# give the result Path
paths = ["/home/data1/qxh/super_image/super-image/output_convnext_20E_quanliang_erase/e9_test_result.json", # 1
         "/home/data1/qxh/super_image/super-image/output_convnext_20E_5split_erase/weight2_test_result.json", # 2
         "/home/data1/qxh/super_image/super-image/output_convnext_20E_5split_drop/test_result_weight.json",
            "/home/data1/qxh/super_image/New Folder/xh5_test_result_spilt1_5_all_FL_32len__67.464.json", # 3 4
         "/home/data1/qxh/super_image/super-image/output_sifar_base_20e_quliang_ersase/crop3_test_result.json", # 5
        "/home/data1/qxh/super_image/super-image/output_5split_sifar_base_20e_quliang_ersase/weight_model_best_test_result.json", # 6
         "/home/data1/qxh/super_image/super-image/output_20e_quliang_nfnet_f3_224/crop3_test_result.json", # 7
         "/home/data1/qxh/super_image/super-image/output_20e_5split_nfnet_f3_224/weight_model_best_test_result.json", # 8
         ]

# weights should added by 1
# weights = [0.2,0.2,0.3,0.1,0.2]
# weights = [0.2,0.2,0.3,0.05,0.05,0.1,0.1]
# weights = [0.125,0.125,0.25,0.125,0.125,0.125,0.125]
# last
weights = [0.1,0.2,0.05,0.25,0.05,0.1,0.1,0.15]
jsonDataList = []
tmp = {}
for path in paths:
    json_data = getInput(path)
    jsonDataList.append(json_data)
for idx,json_data in enumerate(jsonDataList):
    if idx == 0:
        filePaths = list(json_data.keys())
    for filePath in filePaths:
        if tmp.get(filePath) is None:
            tmp[filePath] = torch.tensor([0.0] * 17)
        tmp[filePath] += weights[idx] * torch.tensor(json_data[filePath])
        if idx+1 == len(jsonDataList):
            tmp[filePath] = tmp[filePath].tolist()
output("/home/data1/qxh/super_image/New Folder/9.last_conv_all+5sp+drop+new_swinv+swint1crop+nfnet3crop+swin5split_crop+nfnet5split_crop_trick.json",tmp)
