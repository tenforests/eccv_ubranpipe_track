import json
import torch


def getInput(input_dir):
    with open(str(input_dir) + '/model_best_crop_test_result.json', 'r') as f:
        json_data = json.load(f)
    return json_data


def output(output_dir, tmp):
    with open(str(output_dir) + '/weight_crop_model_best_test_result.json', 'w') as f:
        json.dump(tmp, f)


def loadLogger(input_dir):
    AP=[]
    # with open(str(input_dir) + '/log.txt', 'r') as f:
    #     json_data = json.load(f)
    # AP = []
    # for i in range(0,17):
    #     if i >= 0 and i <= 16:
    #         AP.append(json_data[9]["test_AP"+str(i)])
    return AP


# get output from jsonFile
def getFoldOutput(input_dir):
    jsonDataList = []
    APs = []
    tmp = input_dir
    for i in range(1, 6):
        input_dir = tmp + "split" + str(i)
        json_data = getInput(input_dir)
        AP = loadLogger(input_dir)
        jsonDataList.append(json_data)
        APs.append(AP)
    return jsonDataList, APs


# direct mix 5fold
def foldMix(jsonDataList):
    x = 0
    filePaths = []
    tmp = {}
    for json_data in jsonDataList:
        # print(type(json_data))
        if x == 0:
            filePaths = list(json_data.keys())
        for filePath in filePaths:
            if x == 0:
                tmp[filePath] = torch.tensor(json_data[filePath])
            elif x < 4:
                tmp[filePath] += torch.tensor(json_data[filePath])
            else:
                tmp[filePath] += torch.tensor(json_data[filePath])
                tmp[filePath] = (tmp[filePath] / 5).tolist()
        x +=1
    return tmp


# use the weights(weights should be added by 1)
# weights = [0.2,0.1,0.3,0.1,0.3]
def weightFoldMix(jsonDataList, weights):
    x = 0
    filePaths = []
    tmp = {}
    for json_data in jsonDataList:
        if x == 0:
            filePaths = list(json_data.keys())
        for filePath in filePaths:
            if tmp.get(filePath) is None:
                tmp[filePath] = torch.tensor([0.0] * 17)
            tmp[filePath] += weights[x] * torch.tensor(json_data[filePath])
            if x==4:
                tmp[filePath] = tmp[filePath].tolist()
        x+=1
    return tmp


# use the ap to get more powerful result
# need load logger
def apFoldMix(jsonDataList, APs):
    maxAP = [0] * 17
    x = 0
    filePaths = []
    tmp = {}
    for json_data, AP in zip(jsonDataList, APs):
        if x == 0:
            filePaths = list(json_data.keys())
        for i in range(len(AP)):
            if AP[i] >= maxAP[i]:
                maxAP[i] = AP[i]
        for filePath in filePaths:
            t = json_data[filePath]
            t2 = []
            for i in range(len(t)):
                if (AP[i] == maxAP[i]):
                    t2.append(t[i])
                else:
                    t2.append(tmp[filePath][i])
            tmp[filePath] = t2
    return tmp


input_dir = "/home/data1/qxh/super_image/super-image/output_20e_5split_nfnet_f3_224/"
jsonData,APs = getFoldOutput(input_dir)
# tmp = foldMix(jsonData)
weights=[0.2,0.2,0.2,0.2,0.2]
weights_nfnet_f3 = [0.2,0.3,0.1,0.15,0.25]
weights_convnext=[0.3,0.2,0.1,0.15,0.25]
weights_sifar = [0.25,0.2,0.1,0.15,0.3]
tmp = weightFoldMix(jsonData,weights_nfnet_f3)
# tmp = apFoldMix(jsonData,APs)
output_dir = "/home/data1/qxh/super_image/super-image/output_20e_5split_nfnet_f3_224/"
output(output_dir,tmp)
