from pathlib import Path

import numpy as np
import torch
import torch_geometric.utils as utils
from scipy.io import loadmat

# from sklearn import neighbors as nb
from torch_geometric.data import Data

DATA_ROOT = Path(__file__).parent.parent / "datapackage"


# Util functions
def MinMaxNorm(x, v_min, v_max):
    return (x - v_min) / (v_max - v_min)


def deNormMinMax(x, v_min, v_max):
    return x * (v_max - v_min) + v_min


def MeanStdNorm(x, mean, std):
    return (x - mean) / std


def deNormMeanStd(x, mean, std):
    return x * std + mean


def loadData(dir):
    _data = loadmat(dir)
    _location = [_data["StationLon"].squeeze(), _data["StationLat"].squeeze()]
    # _feature = _data['Temperature']
    _feature = []
    _feature.append(_data["HWFlag2"].flatten())
    _feature.append(_data["Tmax"].squeeze())
    _feature.append(_data["Tref"].squeeze())
    _feature.append(_data["Temp"].squeeze())
    _feature.append(_data["Tmin"].squeeze())
    _feature.append(_data["Tdew"].squeeze())
    _feature.append(_data["Wind"].squeeze())
    _feature.append(_data["Prcp"].squeeze())
    _feature.append(_data["PA"].squeeze())
    _feature.append(_data["Psea"].squeeze())
    _feature.append(_data["DOY"].squeeze())
    return np.array(_location), np.array(_feature)


def loadONI(dir):
    _data = loadmat(dir)
    return np.array(_data["ONI_d"].squeeze())


def createGraph(sequence, selfConn=True, thres=0.4, scaledWeights=True):
    _length, _nNodes = sequence.shape
    _corrMat = np.zeros((_nNodes, _nNodes))
    for _m in range(_nNodes):
        for _n in range(_nNodes):
            _mean1, _std1 = np.mean(sequence[:, _m]), np.std(sequence[:, _m])
            _mean2, _std2 = np.mean(sequence[:, _n]), np.std(sequence[:, _n])
            _corrMat[_m, _n] = np.correlate(
                sequence[:, _m] - _mean1, sequence[:, _n] - _mean2
            ) / (_std1 * _std2 * _length)
    _corrMat[_corrMat < thres] = 0.0
    _scale = np.mean(np.sum(_corrMat, axis=1))
    if scaledWeights:
        _corrMat /= _scale
    _edgeIdx, _edgeAttr = utils.dense_to_sparse(torch.DoubleTensor(_corrMat))
    if not selfConn:
        _edgeIdx, _edgeAttr = utils.remove_self_loops(_edgeIdx, _edgeAttr)
    return _edgeIdx, _edgeAttr


# Data generator
class FTGenerator(object):
    def __init__(self, parameterDict, metadata=None, nodeSelection=None):
        self.device = parameterDict["device"]
        self.Cin = parameterDict["Cin"]
        self.Cout = parameterDict["Cout"]
        self.selfConn = parameterDict["selfConn"]
        self.scaledWeights = parameterDict["scaledWeights"]
        self.featureIdx = parameterDict["featureIdx"]
        self.norm = parameterDict["norm"]
        self.nHistYear = parameterDict["nHistYear"]
        self.graphCorrIdx = parameterDict["graphCorrIdx"]
        self.thres = parameterDict["thres"]
        self.nodeSelection = nodeSelection
        self.radius = 20
        self.avgNN = 4
        if metadata is None:
            self._extractMetadata()
        else:
            self.metadata = metadata
        assert self.metadata is not None

    def _extractMetadata(self):
        self.metadata = {}
        _rawState, _ = self._constructRawData()
        _hwFlag = _rawState[:, :, 0].flatten()
        _mask = _hwFlag == 1
        _weights = np.sum(_mask) / (len(_mask) * 0.4)
        self.metadata["weights"] = [_weights, 1 - _weights]
        _nFeature = _rawState.shape[-1]
        _minState = np.min(_rawState.reshape(-1, _nFeature), axis=0)
        _maxState = np.max(_rawState.reshape(-1, _nFeature), axis=0)
        _minState[-1], _maxState[-1] = 0.0, 1.0  # Not normalizing Tmax classification
        self.metadata["varMin"] = _minState
        self.metadata["varMax"] = _maxState
        _meanState = np.mean(_rawState.reshape(-1, _nFeature), axis=0)
        _stdState = np.std(_rawState.reshape(-1, _nFeature), axis=0)
        _meanState[-1], _stdState[-1] = 0.0, 1.0  # Not normalizing Tmax classification
        self.metadata["mean"] = _meanState
        self.metadata["std"] = _stdState
        self.metadata["nClass"] = int(np.max(_rawState[:, :, 0] + 1))
        self.nClass = self.metadata["nClass"]

    def _constructRawData(self):
        _location, _raw = [], []
        file_paths = sorted(list(DATA_ROOT.glob("*.mat")))
        for _path in file_paths:
            _loc, _feat = loadData(_path)
            _location.append(_loc)
            _raw.append(_feat)
        _raw, _location = np.array(_raw), np.array(_location)
        _raw = np.transpose(np.array(_raw), (2, 0, 1))

        # Adding station locations as features
        _nSteps, _nNodes, _nFeatures = _raw.shape
        _rawNew = np.zeros((_nSteps, _nNodes, _nFeatures + 2))
        _rawNew[:, :, :_nFeatures] = _raw.copy()
        _rawNew[:, :, _nFeatures:] = _location.copy()
        _raw = _rawNew.copy()

        # Adding ONI as a global influence feature
        _oni = loadONI(DATA_ROOT / "ONI.mat")
        _nSteps, _nNodes, _nFeatures = _raw.shape
        _rawNew = np.zeros((_nSteps, _nNodes, _nFeatures + 1))
        _rawNew[:, :, :-1] = _raw.copy()
        _rawNew[:, :, -1] = _oni.reshape(-1, 1)
        _raw = _rawNew.copy()

        if self.nodeSelection is not None:
            return _raw[:, self.nodeSelection], _location[self.nodeSelection]
        else:
            return _raw, _location

    def _createHistFT(self):
        _rawData, _location = self._constructRawData()
        if self.norm == "MinMax":
            _normData = MinMaxNorm(
                _rawData, self.metadata["varMin"], self.metadata["varMax"]
            )
        elif self.norm == "MeanStd":
            _normData = MeanStdNorm(
                _rawData, self.metadata["mean"], self.metadata["std"]
            )
        else:
            print("Data is not normalized.")
            _normData = _rawData

        # create edges
        _edgeIdx, _edgeAttr = createGraph(
            _rawData[:, :, self.graphCorrIdx],
            self.selfConn,
            self.thres,
            self.scaledWeights,
        )

        # Assemble FT with sliding window
        _nSteps, _nNodes, _ = _normData.shape
        _totalYear = int(_nSteps / 365)
        assert self.nHistYear < _totalYear
        _startIdx = self.nHistYear * 365
        _nFeatures = ((self.Cin + self.Cout) * self.nHistYear + self.Cin) * len(
            self.featureIdx
        )
        _nSteps -= self.Cin + self.Cout + self.nHistYear * 365
        _FT = np.zeros([_nSteps, _nNodes, _nFeatures])
        _TG = np.zeros([_nSteps, _nNodes, self.Cout])
        for _n in range(_nSteps):
            _window = [
                np.arange(_n + _m * 365, _n + _m * 365 + (self.Cin + self.Cout))
                for _m in range(self.nHistYear)
            ]
            _window = np.array(_window).flatten()
            _window = np.hstack(
                [_window, np.arange(_startIdx + _n, _startIdx + _n + self.Cin)]
            )
            _feature = _normData[_window]
            _feature = np.transpose(_feature[:, :, self.featureIdx], (1, 0, 2)).reshape(
                _nNodes, -1
            )
            _target = _normData[
                _startIdx + _n + self.Cin : _startIdx + _n + self.Cin + self.Cout, :, 0
            ].T
            _FT[_n], _TG[_n] = _feature, _target

        return _FT, _TG, _edgeIdx, _edgeAttr

    def _createFT(self):
        _rawData, _location = self._constructRawData()
        if self.norm == "MinMax":
            _normData = MinMaxNorm(
                _rawData, self.metadata["varMin"], self.metadata["varMax"]
            )
        elif self.norm == "MeanStd":
            _normData = MeanStdNorm(
                _rawData, self.metadata["mean"], self.metadata["std"]
            )
        else:
            print("Data is not normalized.")
            _normData = _rawData

        # create edges
        _edgeIdx, _edgeAttr = createGraph(
            _rawData[:, :, self.graphCorrIdx], self.selfConn, self.thres
        )

        # Assemble FT with sliding window
        _nSteps, _nNodes, _ = _normData.shape
        _nSteps -= self.Cin + self.Cout - 1
        _FT = np.zeros([_nSteps, _nNodes, len(self.featureIdx) * self.Cin])
        _TG = np.zeros([_nSteps, _nNodes, self.Cout])

        for _n in range(_nSteps):
            _feature = np.transpose(
                _normData[_n : _n + self.Cin, :, self.featureIdx], (1, 0, 2)
            ).reshape(_nNodes, -1)
            _target = _normData[_n + self.Cin : _n + self.Cin + self.Cout, :, 0].T
            _FT[_n], _TG[_n] = _feature, _target
        return _FT, _TG, _edgeIdx, _edgeAttr

    def createDataset(self):
        _Dataset = []
        _FT, _TG, _edgeIdx, _edgeAttr = (
            self._createFT() if self.nHistYear == 0 else self._createHistFT()
        )
        for _n in range(len(_FT)):
            _Dataset.append(
                Data(
                    x=torch.DoubleTensor(_FT[_n]),
                    y=torch.LongTensor(_TG[_n]),
                    edge_index=_edgeIdx,
                    edge_attr=_edgeAttr,
                ).to(self.device)
            )
        return _Dataset, self.metadata
