import torch


def extract_feature(model, loader):
    features = torch.FloatTensor()
    labels = torch.LongTensor()
    cameras = torch.LongTensor()
    clothes = torch.LongTensor()

    for (inputs, label, cams, clos, parsing) in loader:

        input_img = inputs.to('cuda')
        outputs = model(input_img)
        f1 = outputs[-1].data.cpu()

        # flip
        inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
        input_img = inputs.to('cuda')
        outputs = model(input_img)
        f2 = outputs[-1].data.cpu()
        ff = f1 + f2

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)

        labels = torch.cat((labels, label), 0)
        cameras = torch.cat((cameras, cams), 0)
        clothes = torch.cat((clothes, clos), 0)

    return features, labels, cameras, clothes
