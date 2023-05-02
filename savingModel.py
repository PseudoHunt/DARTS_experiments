def get_network(self, genotype):
    channels = self.init_channels
    n_layers = self.layers
    stem_multiplier = 3

    C_curr = stem_multiplier*channels
    C_prev = channels
    layer = 0
    reduction_prev = False

    feature_mix_layer_index = n_layers // 3 * 2

    stem = nn.Sequential(
        nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
        nn.BatchNorm2d(C_curr)
    )

    prev_layers = nn.ModuleList()
    for i, genotype_layer in enumerate(genotype):
        if isinstance(genotype_layer, tuple):
            op, index = genotype_layer
            if reduction_prev:
                stride = 2
            else:
                stride = 1

            prev_layers.append(MixedOp(C_prev, op, stride))

            C_prev = op.multiplier * C_prev
            reduction_prev = op.reduction
            layer += 1

            if layer == feature_mix_layer_index:
                prev_layers.append(nn.AdaptiveAvgPool2d(1))
                C_prev = C_prev * 2

        else:
            reduction = False
            if reduction_prev:
                reduction = True
                stride = 2
            else:
                stride = 1

            prev_layers.append(Cell(C_prev, genotype_layer, stride, C_curr, reduction))

            C_prev = C_curr
            reduction_prev = False

            if layer == feature_mix_layer_index:
                prev_layers.append(nn.AdaptiveAvgPool2d(1))
                C_prev = C_prev * 2

    classifier = nn.Sequential(
        nn.BatchNorm2d(C_prev),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(C_prev, self.num_classes)
    )

    return nn.Sequential(stem, *prev_layers, classifier)

  
  # create DARTS model
model_darts = DARTS(num_classes=10)

# load trained weights
model_final = torch.load("model.pt")

# extract gene from final model
gene = model_darts.get_gene(model_final)

# create final model from gene
final_model = model_darts.get_network(gene)
