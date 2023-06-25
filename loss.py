from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from utils.TripletLoss import TripletLoss
from utils.AdaptiveLoss import AdaptiveLoss
from utils.range_loss import RangeLoss
from utils.center_loss import CenterLoss
from utils.parsing_loss import ParsingLoss

class Loss(loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()


    def forward(self, outputs, labels, clo_labels, parsing_label):
        cross_entropy_loss = CrossEntropyLoss()
        triplet_loss = TripletLoss(margin=1.2)
        parsing_loss = ParsingLoss()
        triplet_loss1 = TripletLoss(margin=2.0)
        
        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[1:4]] #1:4
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[4:12]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        Parsing_Loss = parsing_loss(outputs[-3], parsing_label.long())

        Weighted_Triplet_Loss = [triplet_loss1(output, labels) for output in outputs[12:15]]
        Weighted_Triplet_Loss = sum(Weighted_Triplet_Loss) / len(Weighted_Triplet_Loss)
        
        loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss + Parsing_Loss + 0.5 * Weighted_Triplet_Loss

        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f Parsing_Loss:%.2f Weighted_Triplet_Loss:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()), 
            Parsing_Loss.data.cpu().numpy(),
            Weighted_Triplet_Loss.data.cpu().numpy()
            end=' ')
        return loss_sum
