import torch

class RepMlp_no_Bn(torch.nn.Module):
    def __init__(self, C, O, h, w, deploy = False):
        super(RepMlp_no_Bn, self).__init__()
        self.h = h
        self.w = w
        self.O = O
        self.C = C
        self.deploy = deploy
        self.fc = torch.nn.Linear(C*h*w, O*h*w)
        self.conv = torch.nn.Conv2d(C, O, 1, 1, 0, bias = False)

    def forward(self, inputs):
        c, h, w = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        if self.deploy:
            return self.fc(inputs.reshape(-1,c*h*w)).reshape(-1, self.O, h, w)
        else:
            rs_inputs = inputs.reshape(-1, c*h*w)
            fc_result = self.fc(rs_inputs).reshape(-1, self.O,h,w)
            conv_result = self.conv(inputs)
            return conv_result + fc_result

    def switch(self):
        self.deploy = True
        I = torch.eye(self.C*self.w*self.h).reshape(self.C*self.w*self.h, self.C,self.h,self.w)
        fc_k = torch.nn.functional.conv2d(I, self.conv.weight, padding=0)
        fc_k = fc_k.reshape(self.C*self.h * self.w, self.O * self.h * self.w).t()
        final_weight = self.fc.weight.data + fc_k
        final_bias = self.fc.bias
        self.__delattr__('fc')
        self.fc = torch.nn.Linear(self.C*self.h*self.w, self.O*self.h*self.w)
        self.fc.weight.data = final_weight
        self.fc.bias.data = final_bias

if __name__ == "__main__":
    x = torch.randn(1,3,20,20)
    repMlp = RepMlp_no_Bn(3,4,x.shape[-2], x.shape[-1])
    repMlp.eval()
    result1 = repMlp(x)

    repMlp.switch()

    result2 = repMlp(x)
    print((result2-result1).abs().sum())