import torch
class RepMLPBlock(torch.nn.Module):
    def __init__(self, C, O, h, w, deploy = False):
        super(RepMLPBlock, self).__init__()
        self.h = h
        self.w = w
        self.O = O
        self.C = C
        self.deploy = deploy
        self.fc = torch.nn.Linear(C*h*w, O*h*w)
        self.conv = torch.nn.Conv2d(C, O, 1, 1, 0, bias=False)
        self.bn = torch.nn.BatchNorm2d(num_features=O)


    def forward(self, inputs):
        c, h, w = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        if self.deploy:
            return self.fc(inputs.reshape(-1,c*h*w)).reshape(-1, self.O, h, w)
        else:
            rs_inputs = inputs.reshape(-1, c*h*w)
            fc_result = self.fc(rs_inputs).reshape(-1, self.O,h,w)
            conv_result = self.bn(self.conv(inputs))
            return conv_result + fc_result

    def fuse_bn(self,conv_or_fc, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = bn.weight / std
        t = t.reshape(-1, 1, 1, 1)

        if len(t) == conv_or_fc.weight.size(0):
            return conv_or_fc.weight * t, bn.bias - bn.running_mean * bn.weight / std
        else:
            repeat_times = conv_or_fc.weight.size(0) // len(t)
            repeated = t.repeat_interleave(repeat_times, 0)
            return conv_or_fc.weight * repeated, (bn.bias - bn.running_mean * bn.weight / std).repeat_interleave(
                repeat_times, 0)
      
    def convert_conv_fc(self, conv_weight, conv_bias):
        I = torch.eye(self.C*self.w*self.h).reshape(self.C*self.w*self.h, self.C,self.h,self.w)
        fc_k = torch.nn.functional.conv2d(I, conv_weight, padding=0)
        fc_k = fc_k.reshape(self.C*self.h * self.w, self.O * self.h * self.w).t()
        fc_bias = conv_bias.repeat_interleave(self.h * self.w)
        return fc_k, fc_bias

    def get_equivalent_fc3(self):
        conv_weight,conv_bias = self.fuse_bn(self.conv, self.bn)

        rep_weight, rep_bias = self.convert_conv_fc(conv_weight, conv_bias)

        return self.fc.weight + rep_weight, self.fc.bias + rep_bias
    def switch(self):
        self.deploy = True
        if self.deploy:
            fc_weight, fc_bias = self.get_equivalent_fc3()
            self.fc.weight.data = fc_weight
            self.fc.bias.data = fc_bias
if __name__ == "__main__":
    x = torch.randn(1,3,20,20)
    repMlp = RepMLPBlock(3,4,x.shape[-2], x.shape[-1])
    repMlp.eval()
    result1 = repMlp(x)

    repMlp.switch()
    result2 = repMlp(x)
    print((result2-result1).abs().sum())
