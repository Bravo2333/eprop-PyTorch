import random

import torch
import torchaudio
import numpy as np


class SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(self, args, split='train'):
        self.dt = 1e-3  # 1ms time step
        # self.n_mels = 40  # Number of Mel frequency bins
        self.n_mels = 128  # Number of Mel frequency bins

        # 加载完整的 SpeechCommands 数据集
        self.full_dataset = torchaudio.datasets.SPEECHCOMMANDS('./', download=True)

        # 数字命令类别
        numeric_commands = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'}
        # numeric_commands = {'zero', 'one', 'two', 'three', 'four'}

        # 筛选出数字命令类别的数据
        filtered_dataset = [item for item in self.full_dataset if item[2] in numeric_commands]

        # 创建类别到索引的映射
        self.class_to_idx = {label: idx for idx, label in enumerate(sorted(numeric_commands))}
        self.n_classes = len(self.class_to_idx)

        # 准备梅尔频谱图转换
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=self.n_mels
        )

        # 划分数据集
        all_indices = list(range(len(filtered_dataset)))
        random.shuffle(all_indices)
        split_idx = int(len(all_indices) * 0.8)

        if split == 'train':
            self.indices = all_indices[:split_idx]
        else:  # 'test'
            self.indices = all_indices[split_idx:]

        # 处理数据集
        self.x = []
        self.y = []

        for i in self.indices:
            waveform, sample_rate, label, _, _ = filtered_dataset[i]

            # 计算梅尔频谱图
            mel = self.mel_spec(waveform)

            # 转换为类似脉冲的表示
            spike_prob = (mel - mel.min()) / (mel.max() - mel.min())
            spikes = torch.bernoulli(spike_prob)

            self.x.append(spikes.squeeze(0).t())  # 转置得到 (time, features)
            self.y.append(self.class_to_idx[label])

        # 转换为张量
        self.x = torch.nn.utils.rnn.pad_sequence(self.x, batch_first=True)
        self.y = torch.tensor(self.y, dtype=torch.long)

        self.n_time_bins = self.x.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    # args.train_len = float('inf')  # 这将确保使用所有训练数据
    # args.test_len = float('inf')  # 这将确保使用所有测试数据
    print("=== Loading SpeechCommands dataset...")
    train_loader, traintest_loader, test_loader = load_dataset_speechcommands(args, kwargs)

    print("Training set length: " + str(args.full_train_len))
    print("Test set length: " + str(args.full_test_len))

    return device, train_loader, traintest_loader, test_loader


def load_dataset_speechcommands(args, kwargs):
    trainset = SpeechCommandsDataset(args, "train")
    testset = SpeechCommandsDataset(args, "test")

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    args.n_classes = trainset.n_classes
    args.n_steps = trainset.n_time_bins
    args.n_inputs = trainset.n_mels
    args.dt = trainset.dt
    args.classif = True
    args.full_train_len = len(trainset)
    args.full_test_len = len(testset)
    args.delay_targets = 0  # No delay in targets for speech commands
    args.skip_test = False

    return train_loader, traintest_loader, test_loader