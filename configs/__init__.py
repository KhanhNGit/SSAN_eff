import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # build dirs
    parser.add_argument('--data_dir', type=str, default="", help='YOUR_Data_Dir')
    parser.add_argument('--result_path', type=str, default='./results', help='root result directory')
    parser.add_argument('--num_dataset_train', type=int, default=3, help='quantity of train datasets')
    # training settings
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='num workers')
    parser.add_argument('--img_size', type=int, default=300, help='img size')
    parser.add_argument('--protocol', type=str, default="all", help='protocal')
    parser.add_argument('--device', type=str, default='0', help='device id, format is like 0,1,2')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='base learning rate')
    parser.add_argument('--num_epochs', type=int, default=500, help='total training epochs')
    parser.add_argument('--print_freq', type=int, default=30, help='print frequency')
    parser.add_argument('--trans', type=str, default="I", help="different pre-process")
    # optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    # debug
    parser.add_argument('--debug_subset_size', type=int, default=None)
    return parser.parse_args()

def parse_args_pred():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', type=str, default="live.jpg", help='image name in images folder')
    return parser.parse_args()

def str2bool(x):
    return x.lower() in ('true')
    