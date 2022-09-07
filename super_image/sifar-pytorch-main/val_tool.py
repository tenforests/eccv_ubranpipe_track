
models = []

class preTrainModelSet():
    def __init__(self,pth_path,dataSetPath,crop3,is_split5):
        self.pth_path = pth_path
        self.dataSetPath = dataSetPath
        if crop3 == True:
            self.crop3 = 3
        if is_split5 == True:
            self.a =1
