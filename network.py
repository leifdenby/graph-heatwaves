from operator import mod
import torch.nn as nn
import torch,pickle
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GATv2Conv
from torch_geometric.loader import DataLoader as batchLoader
from torch_geometric_temporal import GConvGRU
import matplotlib.pyplot as plt
torch.set_printoptions(precision=7)

def MinMaxNorm(x, min, max):
    return (x-min)/(max-min)

def deNormMinMax(x, min, max):
    return x*(max-min)+min

def MeanStdNorm(x, mean, std):
    return (x-mean)/std

def deNormMeanStd(x, mean, std):
    return (x*std+mean)

class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float64)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float64)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float64)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float64)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float64)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

class GAT(torch.nn.Module):

    def __init__(self,modelPara):
        super(GAT, self).__init__()
        self.nMLPLayers = modelPara['nMLP']
        self.nGATLayers = modelPara['nGAT']
        self.coe = 2
        if modelPara['BN']:     self.coe += 1
        if modelPara['Dropout']:  self.coe += 1
        _HLD = modelPara['HLD']
        _AF = ifAF(modelPara['AF'],_HLD)
        _gat = []
        _encode = []
        _decode = []

        for _n in range(self.nMLPLayers):
            _inputChannel = modelPara['iDim'] if _n == 0 else _HLD
            _outputChannel = _HLD
            _encode.append(torch.nn.Linear(_inputChannel,_outputChannel))
            _encode.append(_AF)
        for _n in range(self.nGATLayers):
            _inputChannel = _HLD
            _outputChannel = _HLD
            _gat.append(GATv2Conv(_inputChannel,_outputChannel,heads=modelPara['nHeads'],concat=False,add_self_loops=False,share_weights=True))
            _gat.append(_AF)
            if modelPara['BN']:_gat.append(nn.BatchNorm1d(_HLD))
            if modelPara['Dropout']: _gat.append(nn.Dropout(p=0.5))
        for _n in range(self.nMLPLayers):
          _inputChannel = _HLD
          _outputChannel = modelPara['oDim'] if _n == self.nMLPLayers-1 else _HLD
          _decode.append(torch.nn.Linear(_inputChannel,_outputChannel))

        self.GATLayers = torch.nn.ModuleList(_gat)
        self.encodingLayers = torch.nn.ModuleList(_encode)
        self.decodingLayers = torch.nn.ModuleList(_decode)

    def forward(self, x, edgeIdx, edgeAttr):
        for _f in self.encodingLayers:
            x = _f(x)
        for _n in range(int(len(self.GATLayers)/self.coe)):
            x, (_idx,_alpha) = self.GATLayers[self.coe*_n](x, edgeIdx, None, True)
            for _m in range(1,self.coe):
                x = self.GATLayers[self.coe*_n+_m](x)
        for _f in self.decodingLayers:
            x = _f(x)
        return x, torch.mean(_alpha,axis=1)

def modelSelect(modelPara):
    _modelName = modelPara['model'] 
    if _modelName == "GAT":
        _model = GAT(modelPara).double()
    else:
        print ("Model not recognized.")
    return _model

def modelName(paraDict,modelPara):
    _name = modelPara['model']    
    _name += '_Cin'+str(paraDict['Cin'])
    _name += '_Cout'+str(paraDict['Cout'])
    _name += '_'+str(modelPara['nMLP'])+'MLP_'+str(modelPara['nConv'])+'Conv_'+str(modelPara['nGAT'])+'GAT_'
    _name += modelPara['AF']
    _name += str(modelPara['HLD'])+'_'
    _name += "K"+str(modelPara['K'])+'_'
    _name += "nH"+str(modelPara['nHeads'])
    return _name

def ifAF(AF,HLD):
    if AF == 'ReLU':
        return nn.ReLU()
    elif AF == "LeakyReLU":
        return nn.LeakyReLU()
    elif AF == "SiLU":
        return nn.SiLU()
    elif AF == "CELU":
        return nn.CELU()
    elif AF == 'GELU':
        return nn.GELU()
    elif AF == 'SELU':
        return nn.SELU()
    elif AF == "sigmoid":
        return nn.Sigmoid()
    elif AF == "PReLU":
        return nn.PReLU()
    elif AF == "PReLUMulti":
        return nn.PReLU(HLD)
    elif AF == "ELU":
        return nn.ELU()
    elif AF == 'tanh':
        return nn.Tanh()
    else:
        return lambda x: x

def train(model,opt,trainLoader,nNodes,scheduler,nClass,lossFn,weights):
    model.train()
    _lossList,_accuList,_recallList,_precList,_f1List = [],[],[],[],[]
    # _loss = 0
    for _batch in trainLoader:
        _yHat,_ = model(_batch.x,_batch.edge_index,_batch.edge_attr)
        _yHat = _yHat.view(-1,nClass)
        _pred = torch.argmax(_yHat,axis=-1)
        _truth = _batch.y.view(-1)
        _correct = _pred == _truth
        _tP = _truth==1
        _tN = _truth==0
        _pP = _pred==1
        _pN = _pred==0
        _TP = torch.sum(_tP & _pP)
        _FN = torch.sum(_tP & _pN)
        _FP = torch.sum(_tN & _pP)
        _TN = torch.sum(_tN & _pN)
        _recall = _TP/(_TP+_FN)
        _prec   = _TP/(_TP+_FP)
        _f1     = 2*_TP/(2*_TP+_FP+_FN)
        _recall[torch.isnan(_recall)] = 0.
        _prec[torch.isnan(_prec)] = 0.
        _f1[torch.isnan(_f1)] = 0.

        _loss = lossFn(_yHat,_truth)
        _loss.backward(retain_graph=False)
        opt.step()
        opt.zero_grad()

        _recallList.append(_recall.item())
        _precList.append(_prec.item())
        _f1List.append(_f1.item())
        _lossList.append(_loss.item())
        _accuList.append(int(_correct.sum())/len(_truth))
    scheduler.step()
    
    return np.mean(_lossList),np.mean(_accuList),[np.mean(_recallList),np.mean(_precList),np.mean(_f1List)]

def oneStepEval(model,validLoader,nNodes,nClass,lossFn,weights):
    model.eval()
    _lossList,_accuList,_recallList,_precList,_f1List = [],[],[],[],[]
    for _batch in validLoader:
        _yHat,_ = model(_batch.x,_batch.edge_index,_batch.edge_attr)
        _yHat = _yHat.view(-1,nClass)
        _pred = torch.argmax(_yHat,axis=-1)
        _truth = _batch.y.view(-1)
        _correct = _pred == _truth
        
        _tP = (_truth==1).reshape(-1,nNodes)
        _tN = (_truth==0).reshape(-1,nNodes)
        _pP = (_pred==1).reshape(-1,nNodes)
        _pN = (_pred==0).reshape(-1,nNodes)
        
        _TP = torch.sum(_tP & _pP,axis=0)
        _FN = torch.sum(_tP & _pN,axis=0)
        _FP = torch.sum(_tN & _pP,axis=0)
        _TN = torch.sum(_tN & _pN,axis=0)

        _recall = _TP/(_TP+_FN)
        _prec   = _TP/(_TP+_FP)
        _f1     = 2*_TP/(2*_TP+_FP+_FN)
        _recall[torch.isnan(_recall)] = 0.
        _prec[torch.isnan(_prec)] = 0.
        _f1[torch.isnan(_f1)] = 0.

        _loss = lossFn(_yHat,_truth)#+nn.CrossEntropyLoss(weights)(_yHat,_truth)
        print (f"Max recall {torch.max(_recall)*100:1.2f}% at station {torch.argmax(_recall).item()}. Min TPR {torch.min(_recall)*100:1.2f}% at station {torch.argmin(_recall).item()}.")
        print (f"TPR worst 10 stations {torch.argsort(_recall)[:10]}.")
        with open('./log','a') as f:
            print (f"Max recall {torch.max(_recall)*100:1.2f}% at station {torch.argmax(_recall).item()}. Min TPR {torch.min(_recall)*100:1.2f}% at station {torch.argmin(_recall).item()}.",file=f)
            print (f"Recall worst 10 stations {torch.argsort(_recall)[:10]}.",file=f)

        _recallList.append(torch.mean(_recall).item())
        _precList.append(torch.mean(_prec).item())
        _f1List.append(torch.mean(_f1).item())
        _lossList.append(_loss.item())
        _accuList.append(int(_correct.sum())/len(_truth))

    with open('./results.pkl','wb') as f:
        pickle.dump([_pred.cpu().detach().numpy(),_truth.cpu().detach().numpy()],f)
    
    return np.mean(_lossList),np.mean(_accuList),[np.mean(_recallList),np.mean(_precList),np.mean(_f1List)]


def plotHist(name):
    Iter,trLoss,vLoss,teLoss = [],[],[],[]
    trAcc, vAcc, teAcc = [],[],[]
    trRecall, vRecall, teRecall = [],[],[]
    trPrec, vPrec, tePrec = [],[],[]
    trF1, vF1, teF1 = [],[],[]

    logFile = open('./log','r')
    log = logFile.readlines()
    for line in log:
        if "Iter" in line:
            line = line.split(' ')
            Iter.append(int(line[1]))
            trLoss.append(float(line[4][:-1]))
            vLoss.append(float(line[7][:-1]))
            teLoss.append(float(line[10][:-1]))
        if "accuracy" in line:
            line = line.split(' ')
            trAcc.append(float(line[3][:4]))
            vAcc.append(float(line[7][:4]))
            teAcc.append(float(line[-1][:4]))
        if "Train recall" in line:
            line = line.split(' ')
            trRecall.append(float(line[2][:4]))
            vRecall.append(float(line[5][:4]))
            teRecall.append(float(line[-1][:4]))
        if "Train precision" in line:
            line = line.split(' ')
            # trPrec.append(float(line[2][:4]))
            vPrec.append(float(line[5][:4]))
            tePrec.append(float(line[-1][:4]))
        if "Train f1" in line:
            line = line.split(' ')
            # trF1.append(float(line[2][:4]))
            vF1.append(float(line[5][:4]))
            teF1.append(float(line[-1][:4]))
    # f,axes = plt.subplots(1,2,constrained_layout=True,figsize=(10,4))
    # axes[0].loglog(trainingLoss)
    fig,axes = plt.subplots(1,5,figsize=(20,4))
    axes[0].semilogy(Iter,trLoss,'-k',label='Training Loss')
    axes[0].semilogy(Iter,vLoss,'-r',label='Validation Loss')
    axes[0].semilogy(Iter,teLoss,'-b',label='Test Loss')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Cross Entropy')
    axes[0].legend(loc=0)

    axes[1].plot(Iter,trAcc,'-k',label='Training')
    axes[1].plot(Iter,vAcc,'-r',label='Validation')
    axes[1].plot(Iter,teAcc,'-b',label='Test')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Percentage(%)')
    axes[1].set_title('Accuracy')
    axes[1].legend(loc=0)

    axes[2].plot(Iter,trRecall,'-k',label='Training')
    axes[2].plot(Iter,vRecall,'-r',label='Validation')
    axes[2].plot(Iter,teRecall,'-b',label='Test')
    axes[2].set_xlabel('Iterations')
    axes[2].set_ylabel('Percentage(%)')
    axes[2].set_title('Recall')
    axes[2].legend(loc=0)

    # axes[3].plot(Iter,trPrec,'-k',label='Training')
    axes[3].plot(Iter,vPrec,'-r',label='Validation')
    axes[3].plot(Iter,tePrec,'-b',label='Test')
    axes[3].set_xlabel('Iterations')
    axes[3].set_ylabel('Percentage(%)')
    axes[3].set_title('Precision')
    axes[3].legend(loc=0)

    # axes[4].plot(Iter,trF1,'-k',label='Training')
    axes[4].plot(Iter,vF1,'-r',label='Validation')
    axes[4].plot(Iter,teF1,'-b',label='Test')
    axes[4].set_xlabel('Iterations')
    axes[4].set_ylabel('Percentage(%)')
    axes[4].set_title('F1 score')
    axes[4].legend(loc=0)
    fig.savefig(f'./histPlots/{name}.png')
