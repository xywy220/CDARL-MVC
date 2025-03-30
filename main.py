import argparse
import collections
import random
from model import Model
from get_indicator_matrix_A import get_mask
from datasets import *
from configure import get_default_config

def main(MR):
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    print("GPU: " + str(use_cuda))
    #device = torch.device('cuda:0' if use_cuda else 'cpu')
    device = 'cpu'

    # Configure
    config = get_default_config(dataset)
    config['dataset'] = dataset
    print("Data set: " + config['dataset'])
    logger = get_logger()

    # Load data
    seed = config['training']['seed']
    X_list, Y_list = load_data(config)
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]

    accumulated_metrics = collections.defaultdict(list)
    alp = config['training']['alpha']
    print('--------------------Missing rate = ' + str(MR) + '--------------------')
    # Set random seeds for model initialization
    np.random.seed(seed)
    mask = get_mask(2, x1_train_raw.shape[0], MR)
    # mask the data
    x1_train = x1_train_raw * mask[:, 0][:, np.newaxis]
    x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]
    x1_train = torch.from_numpy(x1_train).float().to(device)
    x2_train = torch.from_numpy(x2_train).float().to(device)

    random.seed(seed + 1)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.backends.cudnn.deterministic = True

    # Build the model
    CDARL = Model(config)
    CDARL.to_device(device)

    # Training
    acc, nmi, ari = CDARL.train(alp, config, x1_train, x2_train, Y_list, mask, device)
    accumulated_metrics['acc'].append(acc)
    accumulated_metrics['nmi'].append(nmi)
    accumulated_metrics['ari'].append(ari)
    print('------------------------Training over------------------------')
    cal_std(logger, accumulated_metrics['acc'], accumulated_metrics['nmi'], accumulated_metrics['ari'])


if __name__ == '__main__':
    dataset = {
               0: "CUB",
               1: "Reuters_dim10",
               }
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=str(0), help='dataset id')
    parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
    parser.add_argument('--mr', type=str, default=0.5, help='missing rate')
    args = parser.parse_args()
    dataset = dataset[args.dataset]
    main(args.mr)