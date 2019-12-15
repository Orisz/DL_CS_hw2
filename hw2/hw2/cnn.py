import torch
import itertools as it
import torch.nn as nn


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        kernel_size = 3
        pool_size = 2
        padding_zeros = 1
        cur_in_chan = in_channels
        for i, out_channels in enumerate(self.channels):
            layers.append(nn.Conv2d(cur_in_chan, out_channels, kernel_size, padding=padding_zeros))
            cur_in_chan = out_channels
            layers.append(nn.ReLU())
            if (i+1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(pool_size))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        N = len(self.channels)#num of conv->relu blocks
        P = self.pool_every
        num_of_pools = N // P
        in_h_features = in_h / (2**num_of_pools)
        in_w_features = in_w / (2**num_of_pools)
        in_chanels_features = self.channels[-1]
        in_features = in_h_features * in_w_features * in_chanels_features
        for hid_dim in self.hidden_dims:
            layers.append(nn.Linear(int(in_features), hid_dim))
            in_features = hid_dim
            layers.append(nn.ReLU())
        #add final linear layer for classes
        layers.append(nn.Linear(in_features, self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        features = self.feature_extractor(x)
        col_vec_features = features.view(features.shape[0], -1)
        out = self.classifier(col_vec_features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=False, dropout=0.):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  the main_path, which should contain the convolution, dropout,
        #  batchnorm, relu sequences, and the shortcut_path which should
        #  represent the skip-connection.
        #  Use convolutions which preserve the spatial extent of the input.
        #  For simplicity of implementation, we'll assume kernel sizes are odd.
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        main_layers = []
        shortcut_layers = []
        
        cur_in_chan = in_channels
        for out_channels, ker_size in zip(channels, kernel_sizes):
            padding_zeros = int((ker_size - 1) / 2)
            main_layers.append(nn.Conv2d(cur_in_chan, out_channels, ker_size, padding=padding_zeros))
            if dropout > 0:
                main_layers.append(nn.Dropout2d(dropout))
            if batchnorm:
                main_layers.append(nn.BatchNorm2d(out_channels))
            main_layers.append(nn.ReLU())
            cur_in_chan = out_channels
        
        #removing last unnececary drop, bn, relu:
        if(dropout>0 and batchnorm):#we have both bn and drop
            main_layers = main_layers[:-3]
        elif dropout==0 and batchnorm==False:#we dont have either, bn or drop
            main_layers = main_layers[:-1]
        else:#we have 1, drop or bn
            main_layers = main_layers[:-2]
        if in_channels != channels[-1]:
            #shortcut identitiy initialization
            conv1x1 = nn.Conv2d(in_channels, channels[-1], kernel_size=(1,1), bias=False)
            shortcut_layers.append(conv1x1)
        self.main_path = nn.Sequential(*main_layers)
        self.shortcut_path = nn.Sequential(*shortcut_layers)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ReLU)*P -> MaxPool]*(N/P)
        #   \------- SKIP ------/
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs (with a skip over them) should exist at the end,
        #  without a MaxPool after them.
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        N = len(self.channels)#num of conv->relu blocks
        P = self.pool_every
        num_of_P_res_blocks = N // P
        last_res_clock_len = N % P
            
        kernel_size = 3
        pool_size = 2
        cur_in_chan = in_channels
        start_channel = 0
        res_blocks_added = 0
        for i, out_channels in enumerate(self.channels):
            if (i+1) % P == 0 and res_blocks_added < num_of_P_res_blocks:
                layers.append(ResidualBlock(in_channels=cur_in_chan,
                                            channels=self.channels[start_channel:start_channel+P],
                                            kernel_sizes=[kernel_size]*P))
                cur_in_chan = out_channels
                layers.append(nn.MaxPool2d(pool_size))
                res_blocks_added += 1
                start_channel += P
            if res_blocks_added == num_of_P_res_blocks and last_res_clock_len != 0:
                #add (conv->relu)*(#<mod division>)
                layers.append(ResidualBlock(in_channels=cur_in_chan,
                                            channels=self.channels[start_channel:start_channel+last_res_clock_len],
                                            kernel_sizes=[kernel_size]*last_res_clock_len))
                break #break for loop
        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        #  [(CONV -> BN -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV-> BN -> ReLUs should exist at the end, without a MaxPool after them.
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        kernel_size = 3
        pool_size = 2
        padding_zeros = 1
        cur_in_chan = in_channels
        for i, out_channels in enumerate(self.channels):
            layers.append(nn.Conv2d(cur_in_chan, out_channels, kernel_size, padding=padding_zeros))
            if (i+1) % 2 == 0:
                layers.append(nn.BatchNorm2d(out_channels))
            cur_in_chan = out_channels
            layers.append(nn.ReLU())
            if (i+1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(pool_size))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        #  (Linear -> ReLU -> Dropout)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        N = len(self.channels)#num of conv->relu blocks
        P = self.pool_every
        num_of_pools = N // P
        in_h_features = in_h / (2**num_of_pools)
        in_w_features = in_w / (2**num_of_pools)
        in_chanels_features = self.channels[-1]
        in_features = in_h_features * in_w_features * in_chanels_features
        for hid_dim in self.hidden_dims:
            layers.append(nn.Linear(int(in_features), hid_dim))
            in_features = hid_dim
            layers.append(nn.ReLU())
            layers.append(nn.Dropout())
        #add final linear layer for classes
        layers.append(nn.Linear(in_features, self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq
    # ========================
