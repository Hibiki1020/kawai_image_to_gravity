import numpy as np
import math

import torch

def getTriangularMatrix(outputs):
    elements = outputs[:, 3:9]
    L = torch.zeros(outputs.size(0), elements.size(1)//2, elements.size(1)//2)
    L[:, 0, 0] = torch.exp(elements[:, 0])
    L[:, 1, 0] = elements[:, 1]
    L[:, 1, 1] = torch.exp(elements[:, 2])
    L[:, 2, 0] = elements[:, 3]
    L[:, 2, 1] = elements[:, 4]
    L[:, 2, 2] = torch.exp(elements[:, 5])
    return L

def originalCriterion(outputs, labels, device):
    mu = outputs[:, :3]
    L = getTriangularMatrix(outputs)
    L = L.to(device)
    LL = getCovMatrix(outputs)
    # Ltrans = Ltrans.to(device)

    # ratio = 10.0
    # mu = ratio * mu
    # labels = ratio * 
    
    #print(mu)
    #print(labels)

    dist = torch.distributions.MultivariateNormal(mu, scale_tril=L)
    loss = -1 * dist.log_prob(labels)
    # for i in range(loss.size(0)):
    #     if torch.isnan(loss[i]):
    #         print("torch.isnan(loss[i]) = ", torch.isnan(loss[i]))
    #         print("outputs[i] = ", outputs[i])
    #         print("L[i] = ", L[i])
    #         print("LL[i] = ", LL[i])
    loss = loss.mean()
    # print("loss = ", loss)

    return loss

def getCovMatrix(outputs):
    L = getTriangularMatrix(outputs)
    Ltrans = torch.transpose(L, 1, 2)
    LL = torch.bmm(L, Ltrans)
    return LL

# def computeDet(m):
#     det = \
#         m[:, 0, 0] * (m[:, 1, 1] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 1]) \
#         - (m[:, 0, 1] * (m[:, 1, 0] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 0])) \
#         + (m[:, 0, 2] * (m[:, 1, 0] * m[:, 2, 1] - m[:, 1, 1] * m[:, 2, 0]))
#     return det

##### test #####
# outputs = np.array([
#     [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.5, 0.5, 0.5],
#     [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.5, 0.5, 0.5]
# ]).astype(np.float32)
# outputs = torch.from_numpy(outputs)
# print("outputs.size() = ", outputs.size())
# print("outputs = ", outputs)
# labels = np.array([
#     [2.1, 3.2, 4.3],
#     [2.1, 3.2, 4.3]
# ]).astype(np.float32)
# labels = torch.from_numpy(labels)
# print("labels.size() = ", labels.size())
# print("labels = ", labels)
#
# loss = originalCriterion(outputs, labels)
# print("loss.size() = ", loss.size())
# print("loss = ", loss)
