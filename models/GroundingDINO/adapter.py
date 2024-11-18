import torch

class ElementWiseLayer(torch.nn.Module):
    def __init__(self, feat_dim):
        super(ElementWiseLayer, self).__init__()
        self.mapper = torch.nn.Parameter(torch.ones(1, feat_dim))

    def forward(self, raw_logit):
        return raw_logit * self.mapper

class LinearProbe(torch.nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.mapper = torch.nn.Linear(feat_dim, feat_dim)
        self.mapper.weight.data.copy_(torch.eye(feat_dim))
        self.mapper.bias.data.fill_(0)

    def forward(self, raw_logit):
        x = self.mapper(raw_logit)
        # x = torch.nn.functional.normalize(x, dim=-1)
        return x
    
class MLPProbe(torch.nn.Module):
    def __init__(self, feat_dim, mlp_dim, output_dim, drop_out=0, num_layers=2):
        super().__init__()
        self.mapper = self._build_mlp(num_layers=num_layers, input_dim=feat_dim, mlp_dim=mlp_dim, output_dim=output_dim, drop_out=drop_out)

    def forward(self, raw_logit):
        x = self.mapper(raw_logit)
        # x = torch.nn.functional.normalize(x, dim=-1)
        return x

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, drop_out=0, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(torch.nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(torch.nn.BatchNorm1d(dim2))
                mlp.append(torch.nn.ReLU(inplace=True))
                mlp.append(torch.nn.Dropout(p=drop_out))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(torch.nn.BatchNorm1d(dim2, affine=False))

        return torch.nn.Sequential(*mlp)

def build_adapter(args):
    assert args.adapter_type in ['ElementWiseLayer', 'LinearProbe', 'MLPProbe', ''], "Unknown args.adapter_type: {}".format(args.adapter_type)
    if args.adapter_type == 'ElementWiseLayer':
        return ElementWiseLayer(args.hidden_dim), ElementWiseLayer(args.hidden_dim)
    elif args.adapter_type == 'LinearProbe':
        return LinearProbe(args.hidden_dim), LinearProbe(args.hidden_dim)
    elif args.adapter_type == 'MLPProbe':
        inp_dim = args.hidden_dim
        mlp_dim = 2*args.hidden_dim
        out_dim = args.hidden_dim
        return MLPProbe(inp_dim, mlp_dim, out_dim), MLPProbe(inp_dim, mlp_dim, out_dim)
    elif args.adapter_type == '':
        return None, None
    else:
        raise NotImplementedError("Unknown args.adapter_type: {}".format(args.adapter_type))