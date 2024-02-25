import datetime
import matplotlib.pyplot as plt
import os
import ruamel
import time
import torch

from data import make_batched_dataset, LidarDataset, LIVES_DATA_PATH
from models import SimpleNet, LabelNet, LabelPoseNet

def main(cfg_path: str = "settings.yaml") -> None:
    now = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    cfg = ruamel.yaml.load(cfg_path, Loader=ruamel.yaml.RoundTripLoader)
    device = cfg["system"]["device"]

    data_cfg_path = cfg["data"]["relative_data_path"] + "dataset_status.yaml"
    data_cfg = ruamel.yaml.load(data_cfg_path, Loader=ruamel.yaml.RoundTripLoader)

    if cfg["data"]["history_length"] != data_cfg["HISTORY_LENGTH"] and \
        cfg["data"]["batch_size"] != data_cfg["BATCH_SIZE"]:
        make_batched_dataset(
            batch_size=cfg["data"]["batch_size"],
            k=cfg["data"]["history_length"],
            clear=True
            )
        data_cfg["HISTORY_LENGTH"] = cfg["data"]["history_length"]
        data_cfg["BATCH_SIZE"] = cfg["data"]["batch_size"]
        data_cfg["CREATION_TIME"] = now
        ruamel.yaml.dump(data_cfg, data_cfg_path, Dumper=ruamel.yaml.RoundTripDumper)

    dataset = LidarDataset(LIVES_DATA_PATH + "/batched", device=device)
    train_data, test_data = torch.utils.data.random_split(
        dataset, [1-cfg["data"]["data_test_split_fraction"], cfg["data"]["data_test_split_fraction"]]
        )

    train = torch.utils.data.DataLoader(train_data, shuffle=True)
    test = torch.utils.data.DataLoader(test_data)

    net = cfg["train"]["model_fn"](
        k=cfg["data"]["history_length"],
        filters=cfg["train"]["filters"]
        ).to(device)
    if cfg["load"]["model_path"] is not None:
        net.load_state_dict(torch.load(cfg["load"]["model_path"]))
    criterion = cfg["train"]["loss_fn"]()
    optimizer = cfg["train"]["optimizer_fn"](net.parameters())

    weights = torch.tensor([
        [(i/cfg["data"]["history_length"])**cfg["train"]["exp_decay_factor"]] for i in range(1,cfg["data"]["history_length"])
        ])
    weights /= torch.sum(weights)

    train_loss, train_accuracy, test_loss, test_accuracy, test_idx = [], [], [], [], []

    print("[Training]")
    for epoch in range(cfg["train"]["epochs"]):
        net.train()
        start = time.time()
        for i, data in enumerate(train):
            p = data[0].view(-1, cfg["data"]["history_length"], 3)
            x = data[1].view(-1, cfg["data"]["history_length"], cfg["data"]["scan_length"])
            y = data[2].view(-1, cfg["data"]["history_length"], cfg["data"]["scan_length"])
            y_test = y[:, -1, :]
            y_train = y.clone() 
            corruption = torch.where(
                torch.rand(
                    y.size()[0],
                    cfg["data"]["history_length"],
                    cfg["data"]["scan_length"]
                ) >= cfg["train"]["train_data_corruption_fraction"],
                1, -1)
            y_train *= corruption
            y_train[:,-1,:] = torch.einsum("ijk,jl->ik", y_train[:, :-1, :], weights)
            optimizer.zero_grad()
            pred = net(p, x, y_train)
            y_hat = torch.where(pred>=0, 1, -1)
            loss = criterion(pred, y_test)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            accuracy = torch.mean(torch.mean(torch.where(y_test==y_hat, 1., 0.), dim=1), dim=0)
            train_accuracy.append(accuracy.item())
            print("Batch {} train loss: {} train accuracy: {}".format(i, loss.item(), accuracy.item()))
        print("Epoch {} train loss: {} train accuracy: {}".format(epoch, loss.item(), accuracy.item()))
        post_train = time.time()
        print("Training loop {}s".format(post_train-start))

        net.eval()
        running_test_loss, running_test_accuracy = [], []
        for i, data in enumerate(test):
            p = data[0].view(-1, cfg["data"]["history_length"], 3)
            x = data[1].view(-1, cfg["data"]["history_length"], cfg["data"]["scan_length"])
            y = data[2].view(-1, cfg["data"]["history_length"], cfg["data"]["scan_length"])
            y_test = y[:, -1, :]
            y_train = y.clone() 
            corruption = torch.where(
                torch.rand(
                    y.size()[0],
                    cfg["data"]["history_length"],
                    cfg["data"]["scan_length"]
                ) >= cfg["train"]["test_data_corruption_fraction"],
                1, -1)
            y_train *= corruption
            y_train[:,-1,:] = torch.einsum("ijk,jl->ik", y_train[:, :-1, :], weights)
            pred = net(p, x, y_train)
            y_hat = torch.where(pred>=0, 1, -1)
            loss = criterion(pred, y_test)
            running_test_loss.append(loss.item())
            accuracy = torch.mean(torch.mean(torch.where(y_test==y_hat, 1., 0.), dim=1), dim=0)
            running_test_accuracy.append(accuracy.item())

        test_loss.append(sum(running_test_loss)/len(running_test_loss))
        test_accuracy.append(sum(running_test_accuracy)/len(running_test_accuracy))
        test_idx.append(len(train_loss))
        print("Epoch {} test loss: {} test accuracy: {}".format(epoch, test_loss[-1], test_accuracy[-1]))
        print("Validation loop {}s".format(time.time()-post_train))
        print()

    print("[Training Complete]")
    print("[Saving Logs]")
    cwd = os.path.dirname(os.path.realpath(__file__))
    logging_path = os.path.join(cwd, "logs/{}".format(now))
    os.mkdir(logging_path)
    ruamel.yaml.dump(cfg, logging_path + "/settings.yaml", Dumper=ruamel.yaml.RoundTripDumper)
    torch.save(net.state_dict(), logging_path + "/model.pt")

    fig, ax = plt.subplots()
    ax.plot(train_loss, color="blue", label="train")
    ax.plot(test_idx, test_loss, color="red", label="test")
    ax.set_xlabel("Iteration") 
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    plt.savefig(logging_path + "loss.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(train_accuracy, color="blue", label="train")
    ax.plot(test_idx, test_accuracy, color="red", label="test")
    ax.set_xlabel("Iteration") 
    ax.set_ylabel("Accuracy [%]")
    ax.set_title("Accuracy")
    plt.savefig(logging_path + "accuracy.png")
    plt.close()

    print("[Done]")

if __name__ == "__main__":
    main()