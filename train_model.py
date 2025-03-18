"""
Train the fully convolutional neural network model with PyTorch.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import numpy as np
import time
import torch

from tensorboardX import SummaryWriter
from typing import Any

from utils.checkpointing import CheckpointManager
from utils.datasets import InjectionDataset
from utils.models import FCNN
from utils.training import AverageMeter, get_log_dir, update_lr

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_arguments() -> argparse.Namespace:
    """
    Set up an ArgumentParser to get the command line arguments.

    Returns:
        A Namespace object containing all the command line arguments
        for the script.
    """

    # Set up parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--batch-size',
                        default=64,
                        type=int,
                        metavar='N',
                        help='Size of the mini-batches during training. '
                             'Default: 64.')
    parser.add_argument('--epochs',
                        default=64,
                        type=int,
                        metavar='N',
                        help='Total number of training epochs. Default: 64.')
    parser.add_argument('--learning-rate',
                        default=3e-4,
                        type=float,
                        metavar='LR',
                        help='Initial learning rate. Default: 3e-4.')
    parser.add_argument('--log-interval',
                        default=32,
                        type=int,
                        metavar='N',
                        help='Logging interval during training. Default: 32.')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        metavar='PATH',
                        help='Path to checkpoint to be used when resuming '
                             'training. Default: None.')
    parser.add_argument('--tensorboard',
                        action='store_true',
                        default=True,
                        help='Use TensorBoard to log training progress? '
                             'Default: True.')
    parser.add_argument('--use-cuda',
                        action='store_true',
                        default=True,
                        help='Train on GPU, if available? Default: True.')
    parser.add_argument('--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='Number of workers for DataLoaders. Default: 4.')

    # Parse and return the arguments (as a Namespace object)
    arguments = parser.parse_args()
    return arguments

def train(dataloader: torch.utils.data.DataLoader,
          model: torch.nn.Module,
          loss_func: Any,
          optimizer: torch.optim.Optimizer,
          epoch: int,
          args: argparse.Namespace):
    """
    Train the given model for a single epoch using the given dataloader.

    Args:
        dataloader: The dataloader containing the training data.
        model: Instance of the model that is being trained.
        loss_func: A loss function to compute the error between the
            actual and the desired output of the model.
        optimizer: An optimizer that is used to compute
            and perform the updates to the weights of the network.
        epoch: The current training epoch.
        args: Namespace object containing global variables.
    """

    # Activate training mode
    model.train()

    # Track time, loss, and accuracy
    batch_times = AverageMeter()
    batch_losses = AverageMeter()
    batch_accuracies = AverageMeter()  # ✅ Add accuracy tracking

    for batch_idx, (data, target) in enumerate(dataloader):
        batch_start = time.time()

        # Move data to GPU if available
        data, target = data.to(args.device), target.to(args.device)
        target = target.squeeze()

        # Zero out gradients
        optimizer.zero_grad()

        # Forward pass
        output = model.forward(data).squeeze()

        # Compute loss
        loss = loss_func(output, target)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        # Convert outputs to binary predictions (threshold at 0.5)
        predicted = (output > 0.5).float()
        
        # Compute accuracy: Compare predicted vs. target
        correct = (predicted == target).float().mean().item()

        # Update meters
        batch_losses.update(loss.item())
        batch_accuracies.update(correct)  # ✅ Store accuracy
        batch_times.update(time.time() - batch_start)

        # Log loss & accuracy to TensorBoard
        if args.tensorboard:
            global_step = ((epoch - 1) * args.n_train_batches + batch_idx) * args.batch_size
            args.logger.add_scalar('loss/train', loss.item(), global_step)
            args.logger.add_scalar('accuracy/train', correct, global_step)  # ✅ Log accuracy

        # Console logging
        if batch_idx % args.log_interval == 0:
            percent = 100. * batch_idx / args.n_train_batches
            print(f'Epoch: {epoch:>3}/{args.epochs} | Batch: {batch_idx:>3}/{args.n_train_batches}'
                  f' ({percent:>4.1f}%) | Loss: {loss.item():.6f} | Acc: {correct:.4f} | Time: {batch_times.value:.3f}s')


def validate(dataloader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             loss_func: Any,
             epoch: int,
             args: argparse.Namespace) -> float:
    """
    Run the model on the validation dataset and compute loss & accuracy.

    Args:
        dataloader: The dataloader containing the validation data.
        model: Instance of the model that is being trained.
        loss_func: Loss function to compute the error.
        epoch: The current training epoch.
        args: Namespace object containing global variables.

    Returns:
        The average validation loss.
    """

    # Activate evaluation mode
    model.eval()

    validation_loss = 0
    validation_correct = 0  # ✅ Track accuracy
    total_samples = 0

    # Ensure loss function sums over batch
    reduction = loss_func.reduction
    loss_func.reduction = 'sum'

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(args.device), target.to(args.device)
            target = target.squeeze()

            # Forward pass
            output = model.forward(data).squeeze()

            # Compute loss
            validation_loss += loss_func(output, target).item()

            # Convert to binary predictions
            predicted = (output > 0.5).float()

            # Compute accuracy
            validation_correct += (predicted == target).float().sum().item()
            total_samples += target.numel()

    # Compute average loss and accuracy
    validation_loss /= total_samples
    validation_accuracy = validation_correct / total_samples  # ✅ Compute accuracy

    print(f'\nValidation Loss: {validation_loss:.5f} | Validation Accuracy: {validation_accuracy:.4f}\n')

    # Log accuracy to TensorBoard
    if args.tensorboard:
        global_step = epoch * args.n_train_batches * args.batch_size
        args.logger.add_scalar('loss/validation', validation_loss, global_step)
        args.logger.add_scalar('accuracy/validation', validation_accuracy, global_step)  # ✅ Log accuracy

    # Restore original reduction method
    loss_func.reduction = reduction

    return validation_loss


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    print('')
    print('TRAIN A FULLY CONVOLUTIONAL NEURAL NETWORK')
    print('')

    # Start the stopwatch
    script_start = time.time()

    # Read in command line arguments
    args = get_arguments()

    print('Preparing the training process:')
    print(80 * '-')

    # -------------------------------------------------------------------------
    # Set up CUDA for GPU support
    # -------------------------------------------------------------------------

    if torch.cuda.is_available() and args.use_cuda:
        args.device = 'cuda'
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f'device: \t\t GPU ({device_count} x {device_name})')
    else:
        args.device = 'cpu'
        print('device: \t\t CPU [CUDA not requested or unavailable]')

    # -------------------------------------------------------------------------
    # Set up the network model
    # -------------------------------------------------------------------------

    # Create a new instance of the model we want to train
    model = FCNN()

    print('model: \t\t\t', model.__class__.__name__)

    # DataParallel will divide and allocate batch_size to all available GPUs
    if args.device == 'cuda':
        model = torch.nn.DataParallel(model)

    # Move model to the correct device
    model.to(args.device)

    # -------------------------------------------------------------------------
    # Instantiate an optimizer, a loss function and a LR scheduler
    # -------------------------------------------------------------------------

    # Instantiate the specified optimizer
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.learning_rate,
                                 amsgrad=True)
    print('optimizer: \t\t', optimizer.__class__.__name__)

    # Define the loss function (we use Binary Cross-Entropy)
    loss_func = torch.nn.BCELoss().to(args.device)
    print('loss_function: \t\t', loss_func.__class__.__name__)

    # Reduce the LR by a factor of 0.5 if the validation loss did not
    # go down for at least 10 training epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           factor=0.5,
                                                           patience=5,
                                                           min_lr=1e-6)

    # -------------------------------------------------------------------------
    # Instantiate a CheckpointManager and load checkpoint (if desired)
    # -------------------------------------------------------------------------
    
    # Instantiate a new CheckpointManager
    checkpoint_manager = CheckpointManager(model=model,
                                           optimizer=optimizer,
                                           scheduler=scheduler,
                                           mode='min',
                                           step_size=-1)

    # Check if we are resuming training, and if so, load the checkpoint
    if args.resume is not None:
        
        # Load the checkpoint from the provided checkpoint file
        checkpoint_manager.load_checkpoint(args.resume)
        args.start_epoch = checkpoint_manager.last_epoch + 1

        # Print which checkpoint we are using and where we start to train
        print(f'checkpoint:\t\t {args.resume} '
              f'(epoch: {checkpoint_manager.last_epoch})')

    # Other, simply print that we're not using any checkpoint
    else:
        args.start_epoch = 1
        print('checkpoint: \t\t None')

    # -------------------------------------------------------------------------
    # Load datasets for training and validation and create DataLoader objects
    # -------------------------------------------------------------------------

    # Load the training and the validation dataset
    training_dataset = InjectionDataset(mode='training')
    validation_dataset = InjectionDataset(mode='validation')

    # Compute size of training / validation set and number of training batches
    args.train_size = len(training_dataset)
    args.validation_size = len(validation_dataset)
    args.n_train_batches = int(np.ceil(args.train_size / args.batch_size))
    print('train_set_size: \t', args.train_size)
    print('validation_set_size: \t', args.validation_size)
    print('n_training_batches: \t', args.n_train_batches)

    # Create DataLoaders for training and validation
    training_dataloader = \
        torch.utils.data.DataLoader(dataset=training_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.workers,
                                    pin_memory=True)
    validation_dataloader = \
        torch.utils.data.DataLoader(dataset=validation_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    pin_memory=True)

    # -------------------------------------------------------------------------
    # Create a TensorBoard logger and log some basics
    # -------------------------------------------------------------------------

    if args.tensorboard:

        # Create TensorBoard logger
        args.logger = SummaryWriter(log_dir=get_log_dir())

        # Add all args to as text objects (to epoch 0)
        for key, value in dict(vars(args)).items():
            args.logger.add_text(tag=key,
                                 text_string=str(value),
                                 global_step=0)

    # -------------------------------------------------------------------------
    # Train the network for the given number of epochs
    # -------------------------------------------------------------------------

    print(80 * '-' + '\n\n' + 'Training the Model:\n' + 80 * '-')

    for epoch in range(args.start_epoch, args.epochs):

        print('')
        epoch_start = time.time()

        # ---------------------------------------------------------------------
        # Train the model for one epoch
        # ---------------------------------------------------------------------

        train(dataloader=training_dataloader,
              model=model,
              loss_func=loss_func,
              optimizer=optimizer,
              epoch=epoch,
              args=args)

        # ---------------------------------------------------------------------
        # Evaluate on the validation set
        # ---------------------------------------------------------------------

        validation_loss = validate(dataloader=validation_dataloader,
                                   model=model,
                                   loss_func=loss_func,
                                   epoch=epoch,
                                   args=args)

        # ---------------------------------------------------------------------
        # Take a step with the CheckpointManager
        # ---------------------------------------------------------------------

        # This will create checkpoint if the current model is the best we've
        # seen yet, and also once every `step_size` number of epochs.
        checkpoint_manager.step(metric=validation_loss,
                                epoch=epoch)

        # ---------------------------------------------------------------------
        # Update the learning rate of the optimizer (using the LR scheduler)
        # ---------------------------------------------------------------------

        # Take a step with the LR scheduler; print message when LR changes
        current_lr = update_lr(scheduler, optimizer, validation_loss)

        # Log the current value of the LR to TensorBoard
        if args.tensorboard:
            args.logger.add_scalar(tag='learning_rate',
                                   scalar_value=current_lr,
                                   global_step=epoch)

        # ---------------------------------------------------------------------
        # Print epoch duration
        # ---------------------------------------------------------------------

        print(f'Total Epoch Time: {time.time() - epoch_start:.3f}s\n')

        # ---------------------------------------------------------------------

    print(80 * '-' + '\n\n' + 'Training complete!')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print('')
    print(f'This took {time.time() - script_start:.1f} seconds!')
    print('')
