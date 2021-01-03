import torch,pdb

def L_c(feature, label):
    #输入feature map和label,返回聚类方差
    center = torch.zeros(label.size()[1], feature.size()[1])
    #label = torch.argmax(label, dim=1)
    num_cnt = torch.zeros(label.size()[0])
    center_loss = 0
    for i in range(feature.size()[0]):
        center[label[i],:] += feature[i,:]
        num_cnt[label[i]] += 1
    for i in range(center.size()[0]):
        if num_cnt[i] >0 :
            center[i,:] = center[i,:] / num_cnt[i].data
    for i in range(feature.size()[0]):
        center_loss += torch.sum((feature[i,:] - center[label[i],:])**2)

    return center_loss

def L_t(v, trunk_out, label, beta=0.008):
    #计算Lt损失
    #v为Fig.1中Trunk Network的v
    #trunk_out为Trunk Network中softmax层的输出
    #label为数据标签，size为[batch, label]
    soft_loss = - torch.sum(torch.log(trunk_out * label))
    center_loss = L_c(v, label)

    return soft_loss + beta * center_loss

def L_s(HR_pred, LR_pred, label):
    #输入HR和LR两个branch network中softmax层的输出
    #label为人脸标签，size为[batch, label]
    return - torch.sum(torch.log(HR_pred * label)) - torch.sum(torch.log(LR_pred * label))
    
def L_e(x, z):
    #输入Fig.1中branch net的x、z返回eq.4中的Le
    return torch.sum((x-z)**2)
