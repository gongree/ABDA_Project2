import torch


# def inner_loss(label, matrixs):

#     loss = 0

#     if torch.sum(label == 0) > 1:
#         loss += torch.mean(torch.var(matrixs[label == 0], dim=0))

#     if torch.sum(label == 1) > 1:
#         loss += torch.mean(torch.var(matrixs[label == 1], dim=0))

#     return loss

def intra_loss(label, matrixs):
    """
    label: (batch,)
    matrixs: (batch, node, node)
    """
    loss = 0

    mask0 = (label == 0).float().view(-1, 1, 1)  # (batch, 1, 1)
    mask1 = (label == 1).float().view(-1, 1, 1)  # (batch, 1, 1)

    count0 = torch.sum(mask0)  # label 0인 샘플 개수
    count1 = torch.sum(mask1)  # label 1인 샘플 개수

    if count0 > 1:
        mean0 = torch.sum(matrixs * mask0, dim=0) / count0  # (node, node)
        var0 = torch.sum(mask0 * (matrixs - mean0) ** 2, dim=0) / (count0 - 1)  # (node, node)
        loss += torch.mean(var0)  # 스칼라
        # print(1, loss)

    if count1 > 1:
        mean1 = torch.sum(matrixs * mask1, dim=0) / count1  # (node, node)
        var1 = torch.sum(mask1 * (matrixs - mean1) ** 2, dim=0) / (count1 - 1)  # (node, node)
        loss += torch.mean(var1)  # 스칼라
        # print(2, loss)

    return loss



# def intra_loss(label, matrixs):
#     a, b = None, None

#     if torch.sum(label == 0) > 0:
#         a = torch.mean(matrixs[label == 0], dim=0)

#     if torch.sum(label == 1) > 0:
#         b = torch.mean(matrixs[label == 1], dim=0)

#     if a is not None and b is not None:
#         return 1 - torch.mean(torch.pow(a-b, 2))
#     else:
#         return 0

def inter_loss(label, matrixs):
    """
    label: (batch,)
    matrixs: (batch, node, node)
    Returns:
        intra-class loss (scalar)
    """
    # print(label)
    mask0 = (label == 0).float().view(-1, 1, 1)  # (batch, 1, 1)
    mask1 = (label == 1).float().view(-1, 1, 1)  # (batch, 1, 1)

    count0 = torch.sum(mask0)  # label 0인 샘플 개수
    count1 = torch.sum(mask1)  # label 1인 샘플 개수
    # print(count0, count1)

    if count0 > 0:
        a = torch.sum(matrixs * mask0, dim=0) / count0  # (node, node)
    else:
        a = None

    if count1 > 0:
        b = torch.sum(matrixs * mask1, dim=0) / count1  # (node, node)
    else:
        b = None

    if a is not None and b is not None:
        # print(a,b)
        return 1 - torch.norm(a - b, p='fro') ** 2 # torch.mean(torch.pow(a - b, 2))  # 스칼라 값 반환
    else:
        # print("여기?")
        return torch.tensor(0.0, device=matrixs.device)


def sparse_loss(matrixs):
    return torch.norm(matrixs, p=1)

def mixup_cluster_loss(matrixs, y_a, y_b, lam, intra_weight=2):

    y_1 = lam * y_a.float() + (1 - lam) * y_b.float()

    y_0 = 1 - y_1

    bz, roi_num, _ = matrixs.shape
    matrixs = matrixs.reshape((bz, -1))
    sum_1 = torch.sum(y_1)
    sum_0 = torch.sum(y_0)
    loss = 0.0

    if sum_0 > 0:
        center_0 = torch.matmul(y_0, matrixs)/sum_0
        diff_0 = torch.norm(matrixs-center_0, p=1, dim=1)
        loss += torch.matmul(y_0, diff_0)/(sum_0*roi_num*roi_num)
    if sum_1 > 0:
        center_1 = torch.matmul(y_1, matrixs)/sum_1
        diff_1 = torch.norm(matrixs-center_1, p=1, dim=1)
        loss += torch.matmul(y_1, diff_1)/(sum_1*roi_num*roi_num)
    if sum_0 > 0 and sum_1 > 0:
        loss += intra_weight * \
            (1 - torch.norm(center_0-center_1, p=1)/(roi_num*roi_num))

    return loss

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)