import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# import torch.utils as utils
from absl import app, flags

# from sklearn.utils import shuffle
from torch_geometric.loader import DataLoader

from . import data, network

DATAROOT_TRAINED_MODELS = Path(__file__).parent.parent / "trained_model"

flags.DEFINE_integer("num_steps", int(1000), help="Number of steps of training.")
flags.DEFINE_string("AF", "PReLUMulti", help="Choice of activation function.")
flags.DEFINE_string("device", "GPU", help="Choice of CPU or GPU.")
flags.DEFINE_string("modelName", "tp", help="Model name")
flags.DEFINE_string("lossFn", "F1", help="Cross-Entropy(CE), Weighted CE(WCE), F1.")
flags.DEFINE_integer("nMLP", int(2), help="Number of MLP layers.")
flags.DEFINE_integer("nConv", int(0), help="Number of Conv layers.")
flags.DEFINE_integer("nGAT", int(2), help="Number of GAT layers.")
flags.DEFINE_integer("nHeads", int(1), help="Attention heads.")
flags.DEFINE_integer("HLD", int(32), help="Hidden layer dimension")
flags.DEFINE_integer("Cin", int(4), help="Input steps")
flags.DEFINE_integer("Cout", int(3), help="Output steps")
flags.DEFINE_integer(
    "graphCorrIdx", int(7), help="Use which feature to create graph correlation"
)
flags.DEFINE_float("thres", 0.05, help="correlation threshold")
flags.DEFINE_integer("batch_size", int(512), help="Training batch size.")
flags.DEFINE_float("lr", 1e-3, help="Learning rate.")
flags.DEFINE_integer("K", int(1), help="Chebconv K.")
flags.DEFINE_string("norm", "MinMax", help="Choose the type of data normalization")
flags.DEFINE_integer("nPrintOut", int(10), help="Number of iterations per validation")
flags.DEFINE_bool("scaledWeights", True, help="Whether to scaled correlation weights")
flags.DEFINE_bool(
    "selfConn", True, help="Whether to keep self-connection in graph or not."
)
flags.DEFINE_bool("Tdiff", False, help="Whether to use Tdiff target or not.")
flags.DEFINE_integer(
    "nHistYear", 0, help="Number of previous year data to include as features"
)
flags.DEFINE_bool("saveModel", True, help="Save model or not.")
flags.DEFINE_bool("shuffleBatch", True, help="Shuffle batch or not.")
flags.DEFINE_bool("BN", True, help="To use batch norm or not.")
flags.DEFINE_bool("Dropout", False, help="To use dropout or not.")
FLAGS = flags.FLAGS


def main(_):
    if FLAGS.device == "CPU":
        device = torch.device("cpu")
    if FLAGS.device == "GPU":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            print("GPU not available, using CPU instead.")
    # ----------------------------- Set up network parameters -----------------------------#
    # raw data feature : [HWFlag,Tmax,Tref,Tavg,Tmin,Tdew,Wind,Prcp,PA,Psea,DOY,ONI,StatLon,StatLat]
    featureIdx = np.arange(0, 14)
    iDim = (
        FLAGS.Cin * len(featureIdx)
        if FLAGS.nHistYear == 0
        else len(featureIdx) * ((FLAGS.Cin + FLAGS.Cout) * FLAGS.nHistYear + FLAGS.Cin)
    )

    # Select model based on # of layers
    if FLAGS.nGAT > 0 and FLAGS.nConv > 0:
        _model = "Hybrid"
    elif FLAGS.nGAT > 0 and FLAGS.nConv == 0:
        _model = "GAT"
    elif FLAGS.nConv > 0 and FLAGS.nGAT == 0:
        _model = "GNN"
    else:
        _model = None
        print("Model NOT recognized!")

    modelPara = {
        "nMLP": FLAGS.nMLP,
        "nConv": FLAGS.nConv,
        "nGAT": FLAGS.nGAT,
        "nHeads": FLAGS.nHeads,
        "iDim": iDim,
        "oDim": FLAGS.Cout,
        "BN": FLAGS.BN,
        "Dropout": FLAGS.Dropout,
        "HLD": FLAGS.HLD,
        "AF": FLAGS.AF,
        "K": FLAGS.K,
        "model": _model,
    }

    paraDict = {
        "device": device,
        "Cin": FLAGS.Cin,
        "Cout": FLAGS.Cout,
        "selfConn": FLAGS.selfConn,
        "scaledWeights": FLAGS.scaledWeights,
        "norm": FLAGS.norm,
        "Tdiff": FLAGS.Tdiff,
        "nHistYear": FLAGS.nHistYear,
        "featureIdx": featureIdx,
        "graphCorrIdx": FLAGS.graphCorrIdx,
        "thres": FLAGS.thres,
    }
    DATAROOT_TRAINED_MODELS.mkdir(parents=True, exist_ok=True)

    modelPath = DATAROOT_TRAINED_MODELS / f".{FLAGS.modelName}.pt"
    metaPath = DATAROOT_TRAINED_MODELS / f"{FLAGS.modelName}_metadata.pkl"
    paraPath = DATAROOT_TRAINED_MODELS / f"{FLAGS.modelName}_para.pkl"
    checkpointPath = DATAROOT_TRAINED_MODELS / "checkpoint.pt"
    # ----------------------------- Load all datasets -----------------------------#
    metadata = None
    if os.path.exists(metaPath):
        with open(metaPath, "rb") as f:
            metadata = pickle.load(f)

    FTG = data.FTGenerator(paraDict, metadata)
    Dataset, metadata = FTG.createDataset()
    nNodes = Dataset[0].x.shape[0]
    nClass = metadata["nClass"]
    print(f"Number of edges {len(Dataset[0].edge_attr)}.")
    print("Dataset generated.")
    # ------------------------- Generate training dataset -----------------------------#
    # Split dataset to training, validation, and testing
    validSetLength = 365
    testSetLength = 365
    trainSetLength = len(Dataset) - validSetLength - testSetLength
    # lengths = [trainSetLength,validSetLength,testSetLength]
    # trainSet,validSet,testSet = utils.data.random_split(dataset=Dataset,lengths=lengths,generator=torch.Generator().manual_seed(10))
    trainSet, validSet, testSet = (
        Dataset[:trainSetLength],
        Dataset[trainSetLength : trainSetLength + validSetLength],
        Dataset[trainSetLength + validSetLength :],
    )
    trainLoader = DataLoader(
        trainSet, batch_size=FLAGS.batch_size, shuffle=FLAGS.shuffleBatch
    )
    validLoader = DataLoader(validSet, batch_size=len(validSet), shuffle=False)
    testLoader = DataLoader(testSet, batch_size=len(testSet), shuffle=False)
    print(f"Dataset length {trainSetLength,validSetLength,testSetLength}.")

    modelPara["oDim"] = FLAGS.Cout * nClass
    model = network.modelSelect(modelPara).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    lossFnWeights = torch.DoubleTensor(metadata["weights"]).to(device)
    # lossFnWeights = torch.DoubleTensor([0.05,0.95]).to(device)
    nParameters = 0
    for thisKey in model.state_dict():
        nParameters += torch.numel(model.state_dict()[thisKey])
    print(f"Number of parameters: {nParameters}")
    # Load existing model parameters if any
    if os.path.exists(checkpointPath) and os.path.exists(metaPath):
        checkpoint = torch.load(checkpointPath, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["opt_state_dict"])
        # accuBest,TPRBest,EPOCH = checkpoint['accuBest'],checkpoint['TPRBest'],checkpoint['epoch']
        bestScore, EPOCH = checkpoint["score"], checkpoint["epoch"]
        print("Loading from existing model parameters as initialization.")
        with open("log", "a") as f:
            print("Loading from existing model parameters as initialization.", file=f)
            print("Model generated", file=f)
            print(model, file=f)
            print(modelPath, file=f)
    else:
        # accuBest,TPRBest,EPOCH = 0,0,0
        bestScore, EPOCH = 100, 0
        print("No checkpoint found, starting with new model.")
        with open("log", "w") as f:
            print("No checkpoint found, starting with new model.", file=f)
            print("Model generated", file=f)
            print(model, file=f)
            print(modelPath, file=f)

    with open("log", "a") as f:
        print(f"Dataset length {trainSetLength,validSetLength,testSetLength}.", file=f)
        print(metadata, file=f)

    # Loss function
    if FLAGS.lossFn == "CE":
        lossFn = nn.CrossEntropyLoss().to(device)
        print("Using unweighted Cross-Entropy loss function.")
    elif FLAGS.lossFn == "WCE":
        lossFn = nn.CrossEntropyLoss(weight=lossFnWeights).to(device)
        print(f"Using weighted Cross-Entropy loss function, weights:{lossFnWeights}.")
    elif FLAGS.lossFn == "F1":
        lossFn = network.F1_Loss().to(device)
        print("Using F1 loss function.")
    else:
        print("Loss function not available, using unweighted Cross-Entropy")
        with open("log", "a") as f:
            print("Loss function not available, using unweighted Cross-Entropy", file=f)
        lossFn = nn.CrossEntropyLoss().to(device)

    # ----------------------------- Train the network -----------------------------#
    nEpoch = FLAGS.num_steps
    history = torch.zeros(nEpoch)
    timeStart = time.time()
    if FLAGS.saveModel:
        with open(metaPath, "wb") as f:
            pickle.dump(metadata, f)
        with open(paraPath, "wb") as f:
            pickle.dump([modelPara, paraDict], f)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.998)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,milestones=np.arange(1000,11000,1000),gamma=0.5)

    for epoch in range(EPOCH, nEpoch):
        printOut = (epoch + 1) % FLAGS.nPrintOut if epoch != 0 else 0
        loss, accuTrain, [recallTrain, precTrain, f1Train] = network.train(
            model, opt, trainLoader, nNodes, scheduler, nClass, lossFn, lossFnWeights
        )
        history[epoch] = loss
        timer = time.time() - timeStart
        if not printOut:
            # Perform validation and test
            (
                lossValid,
                accuValid,
                [recallValid, precValid, f1Valid],
            ) = network.oneStepEval(
                model, validLoader, nNodes, nClass, lossFn, lossFnWeights
            )
            lossTest, accuTest, [recallTest, precTest, f1Test] = network.oneStepEval(
                model, testLoader, nNodes, nClass, lossFn, lossFnWeights
            )
            print("LR =", scheduler.get_last_lr())
            print(
                f"Iter {epoch+1} training loss: {loss}, validation loss: {lossValid}, test loss: {lossTest} time:{timer:1.2f}"
            )
            print(
                f"Mean training accuracy {accuTrain*100:1.2f}%, mean validation accuracy {accuValid*100:1.2f}%, mean test accuracy {accuTest*100:1.2f}%"
            )
            print(
                f"Train recall: {recallTrain*100:1.2f}%. Validation recall: {recallValid*100:1.2f}%, Test recall: {recallTest*100:1.2f}%"
            )
            print(
                f"Train precision: {precTrain*100:1.2f}%. Validation precision: {precValid*100:1.2f}%, Test precision: {precTest*100:1.2f}%"
            )
            print(
                f"Train f1: {f1Train*100:1.2f}%. Validation f1: {f1Valid*100:1.2f}%, Test f1: {f1Test*100:1.2f}%"
            )

            with open("log", "a") as f:
                print("LR =", scheduler.get_last_lr(), file=f)
                print(
                    f"Iter {epoch+1} training loss: {loss}, validation loss: {lossValid}, test loss: {lossTest} time:{timer:1.2f}",
                    file=f,
                )
                print(
                    f"Mean training accuracy {accuTrain*100:1.2f}%, mean validation accuracy {accuValid*100:1.2f}%, mean test accuracy {accuTest*100:1.2f}%",
                    file=f,
                )
                print(
                    f"Train recall: {recallTrain*100:1.2f}%. Validation recall: {recallValid*100:1.2f}%, Test recall: {recallTest*100:1.2f}%",
                    file=f,
                )
                print(
                    f"Train precision: {precTrain*100:1.2f}%. Validation precision: {precValid*100:1.2f}%, Test precision: {precTest*100:1.2f}%",
                    file=f,
                )
                print(
                    f"Train f1: {f1Train*100:1.2f}%. Validation f1: {f1Valid*100:1.2f}%, Test f1: {f1Test*100:1.2f}%",
                    file=f,
                )
                print("", file=f)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "opt_state_dict": opt.state_dict(),
                    # 'accuBest':         accuBest,
                    # 'TPRBest':          TPRBest,
                    "score": bestScore,
                },
                checkpointPath,
            )
            print(lossValid, bestScore, accuValid, recallValid, precValid, f1Valid)
            if lossValid < bestScore and FLAGS.saveModel:
                bestScore = lossValid
                record = [
                    loss,
                    lossValid,
                    lossTest,
                    accuTrain,
                    accuValid,
                    accuTest,
                    recallTrain,
                    recallValid,
                    recallTest,
                    precTrain,
                    precValid,
                    precTest,
                    f1Train,
                    f1Valid,
                    f1Test,
                    epoch,
                ]
                # if (accuValid>accuBest or np.abs(accuValid-accuBest)<=2e-2) and TPRValid>=TPRBest and epoch>=100:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "opt_state_dict": opt.state_dict(),
                        "score": bestScore,
                        # 'accuBest':         accuBest,
                        # 'TPRBest':          TPRBest,
                    },
                    checkpointPath,
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "opt_state_dict": opt.state_dict(),
                    },
                    modelPath,
                )
                print(f"Model saved, loss: {lossValid:1.7e}")
                with open("log", "a") as f:
                    print(f"Model saved, loss: {lossValid:1.7e}", file=f)
            print("")
            with open("log", "a") as f:
                print("", file=f)
    print("Training completed.")
    print(record)
    with open("record", "a") as f:
        print(record, file=f)
    # Plot covergence hist
    network.plotHist(FLAGS.modelName)


if __name__ == "__main__":
    app.run(main)
