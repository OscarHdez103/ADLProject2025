#!/usr/bin/env python3
from modulefinder import test
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import os
import argparse
from pathlib import Path

from dataloader import ProgressionDataset
# from siamese_progression_net import SPN
from spn import CNN


torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(
    description="Train the coursework Siamese Progression Net on the HD-EPIC Dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
## TODO: CHANGE FOR OUR DATASET
default_dataset_dir = Path() / "dataset"
parser.add_argument("--dataset-root", default=default_dataset_dir)

parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-4, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--epoch-size",
    default=2000,
    type=int,
    help="The size of each epoch",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--weight-decay",
    default=1e-4,
    type=float,
    help="Weight decay (L2 regularization factor)",
)
parser.add_argument(
    "--load",
    default="",
    type=str,
    help="Path to a saved model file to load instead of creating a new model",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="If set, skip training and only run validation and test on the model",
)




if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")





def main(args):
    ## TODO: DON'T KNOW WHAT THIS IS FOR
    transform_SPN = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
    ])

    args.dataset_root.mkdir(parents=True, exist_ok=True)

    ## TODO: CHANGE TO OUR TRAIN AND TEST DATASETS
    train_root = os.path.join(args.dataset_root,"train")
    test_root = os.path.join(args.dataset_root, "test")
    val_root = os.path.join(args.dataset_root, "val")
    recipe_ids = [i for i in os.listdir(train_root)]
    train_dataset_SPN = ProgressionDataset(
      root_dir= train_root,
      transform= transform_SPN,
      mode= "train",
      recipe_ids_list= recipe_ids,
      epoch_size= args.epoch_size
    )
    test_dataset_SPN = ProgressionDataset(
      root_dir= test_root,
      transform= transform_SPN,
      mode= "test",
      label_file= os.path.join(args.dataset_root, "test_labels.txt")
    )
    val_dataset_SPN = ProgressionDataset(
      root_dir= val_root,
      transform= transform_SPN,
      mode= "val",
      label_file= os.path.join(args.dataset_root, "val_labels.txt")
    )
    train_loader_SPN = torch.utils.data.DataLoader(
      train_dataset_SPN,
      shuffle=True,
      batch_size=args.batch_size,
      pin_memory=True,
      num_workers=args.worker_count
    )
    val_loader_SPN = torch.utils.data.DataLoader(
        val_dataset_SPN,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True
    )
    test_loader_SPN = torch.utils.data.DataLoader(
        test_dataset_SPN,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    # train_dataset = torchvision.datasets.CIFAR10(
    #     args.dataset_root, train=True, download=True, transform=transform
    # )
    # test_dataset = torchvision.datasets.CIFAR10(
    #     args.dataset_root, train=False, download=False, transform=transform
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     batch_size=args.batch_size,
    #     pin_memory=True,
    #     num_workers=args.worker_count,
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     shuffle=False,
    #     batch_size=args.batch_size,
    #     num_workers=args.worker_count,
    #     pin_memory=True,
    # )

    if args.load and args.load != "":
        print(f"Loading model from '{args.load}'")
        model_SPN = torch.load(args.load, map_location=DEVICE)
    else:
        # model_SPN = SPN(num_classes=3)
        model_SPN = CNN(height=224, width=224, channels=3, class_count=3)

    criterion_SPN = nn.CrossEntropyLoss()
    optimizer_adam = torch.optim.Adam(
        params=model_SPN.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
      str(log_dir),
      flush_secs=5
    )

    ## TODO: Implement Trainer
    trainer_SPN = Trainer_SPN(
      model=model_SPN,
      train_loader=train_loader_SPN,
      val_loader=val_loader_SPN,
      test_loader=test_loader_SPN,
      criterion=criterion_SPN,
      optimizer=optimizer_adam,
      summary_writer=summary_writer,
      device=DEVICE
    )

    if args.test:
        print("Test-only mode: skipping training and running validation + test.")
        # Validation
        trainer_SPN.validate()
        # Test
        test_results = trainer_SPN.test()
        print(f"Test-only results: accuracy={test_results['accuracy']}%, loss={test_results['loss']}")
    else:
        # Normal training loop
        trainer_SPN.train(
          epochs=args.epochs,
          val_frequency=args.val_frequency,
          print_frequency=args.print_frequency,
          log_frequency=args.log_frequency
        )

    summary_writer.close()






class Trainer_SPN:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader : DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        # label_count = [0,0,0]
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch0, batch1, labels in self.train_loader:
                batch0 = batch0.to(self.device)
                batch1 = batch1.to(self.device)

                # import matplotlib.pyplot as plt
                # img0 = batch0.squeeze(0).permute(1, 2, 0)  # [224, 224, 3]
                # plt.imsave("batch0.png", img0.cpu().numpy())
                # img1 = batch1.squeeze(0).permute(1, 2, 0)  # [224, 224, 3]
                # plt.imsave("batch1.png", img1.cpu().numpy())

                labels = labels.to(self.device)
                data_load_end_time = time.time()

                
                # print(f"SHAPE: {labels.shape}")
                # for l in labels:
                #   label_count[int(l)] += 1
                # continue

                # print(f"Batch0: \n{batch0.shape}")
                # print(f"Batch1: \n{batch1.shape}")
                # print(f"Labels: \n{labels}")
                # import sys; sys.exit(1)
                logits = self.model.forward(batch0, batch1)
                
                loss = self.criterion(logits, labels)
                # print(f"Loss:{loss}")

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()
            
            # print(f"The label_count is: {label_count}")
            # with open(f"{self.summary_writer.log_dir}/test.txt", "a") as f:
            #     f.write(f"Data Labels: {label_count}\n")

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()
        t_acc, t_loss = self.test()
        self.model.train()

        idx = self.summary_writer.log_dir.find("CNN_")
        model_name = self.summary_writer.log_dir[idx:]

        i = 0
        while i < 10:
          name = os.path.join(Path("model"),f"model='{model_name}'_loss={t_loss}_acc={t_acc}%_run_{str(i)}")
          if os.path.isfile(name):
            i += 1
            continue
          else:
            torch.save(self.model, name)
            return



    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )
    
    def test(self):
        self.model.eval()
        results = {"preds": [], "labels": []}
        total_loss = 0

        with torch.no_grad():
            for batch0, batch1, labels in self.test_loader:
                batch0 = batch0.to(self.device)
                batch1 = batch1.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch0, batch1)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))
        
        # test_labels = [0,0,0]
        # for l in results["labels"]:
        #   test_labels[l] += 1
        # with open(f"{self.summary_writer.log_dir}/test.txt", "a") as f:
        #     f.write(f"Test Labels: {test_labels}\n")
        # return

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.test_loader)

        print(f"Test accuracy: {accuracy * 100}; Average loss: {average_loss}")
        with open(f"{self.summary_writer.log_dir}/test.txt", "a") as f:
            f.write(f"Test accuracy: {accuracy * 100}; Average loss: {average_loss}\n")
        return {"accuracy": accuracy * 100, "loss": average_loss}


    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch0, batch1, labels in self.val_loader:
                batch0 = batch0.to(self.device)
                batch1 = batch1.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch0, batch1)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        # val_labels = [0,0,0]
        # for l in results["labels"]:
        #   val_labels[l] += 1
        # with open(f"{self.summary_writer.log_dir}/test.txt", "a") as f:
        #     f.write(f"Val Labels: {val_labels}\n")
        # return


        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")




def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())















