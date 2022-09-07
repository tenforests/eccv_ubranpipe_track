import json

news = {}
with open("/home/data1/qxh/super_image/New Folder/9.last_conv_all+5sp+drop+new_swinv+swint1crop+nfnet3crop+swin5split_crop+nfnet5split_crop_trick.json", "r") as olds:
    news = json.load(olds)

# for new in news:
#     flag = False
#     if news[new][0] >= 0.9:
#         flag = True
#         news[new][0] = 1.0
#     for i in range(1, 17):
#         if flag:
#             news[new][i] = 0.0

news = dict(sorted(news.items(),key=lambda item:item[1][0],reverse=True))

i = 0
for new in news:
    if i<800 :
        news[new][0] = 1.0
        for j in range(1,17):
            news[new][j] = 0.0
    i+=1

with open("/home/data1/qxh/super_image/New Folder/9.last_conv_all+5sp+drop+new_swinv+swint1crop+nfnet3crop+swin5split_crop+nfnet5split_crop_trick.json", "w") as f:
    json.dump(news, f)
