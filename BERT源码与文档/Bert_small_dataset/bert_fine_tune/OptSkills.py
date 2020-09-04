# --------------------------------****************************-----------------------------------------------------
# ---------------The Engineering Source Code Of The BERT Fine Tune In Intention Recognition------------------------
# Copyright (C) 2020 Chongqing Normal university and Ma Shang Xiao Fei Financial
# This file is the code of EMA and averagemeter
# EMA is expounded in the link https://blog.csdn.net/mikelkl/article/details/85227053
# the source code of EMA can found in https://www.jianshu.com/p/f99f982ad370
# averagemeter is expounded in the https://blog.csdn.net/rytyy/article/details/105944813
# the source code of averagemeter can found in https://blog.csdn.net/rytyy/article/details/105944813
# --------------------------------****************************-----------------------------------------------------


class EMA:
    def __init__(self, gamma, model):
        super(EMA, self).__init__()
        self.gamma = gamma
        self.shadow = {}
        self.model = model
        self.setup()

    def setup(self):
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                self.shadow[name] = para.clone()

    def cuda(self):
        for k, v in self.shadow.items():
            self.shadow[k] = v.cuda()

    def update(self):
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                self.shadow[name] = (
                    1.0 - self.gamma
                ) * para + self.gamma * self.shadow[name]

    def swap_parameters(self):
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                temp_data = para.data
                para.data = self.shadow[name].data
                self.shadow[name].data = temp_data

    def state_dict(self):
        return self.shadow


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count