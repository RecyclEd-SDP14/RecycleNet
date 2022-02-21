import torch
import torch.nn as nn
import resnet
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

TRASH_DICT = {
    '1': 'glass',
    '2': 'metal',
    '3': 'paper',
    '4': 'plastic',
    '5': 'metal',
    '6': 'trash'
}

class FlaskNet:
    def __init__(self, size, pretrain, attention, model_dir):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = nn.DataParallel(resnet.resnet152(pretrained=pretrain, use_att=attention, num_classes=len(TRASH_DICT)))

        checkpoint = torch.load(model_dir, map_location=self.device)
        state_dict = checkpoint['state_dict']

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def classify(self, img_dir):
        pred_class, confidence = self.inference(img_dir)
        return f'Prediction: {pred_class}, Confidence: {confidence}'

    def inference(self, save_dir):
        frame = Image.open(save_dir)
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        img_transforms = transforms.Compose([transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=MEAN, std=STD)])

        image_tensor = img_transforms(frame).float()
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensor.to(self.device)

        softmax = nn.Softmax(dim=1)
        output = self.model(Variable(image_tensor))
        pred = torch.Tensor.cpu(softmax(output[0].data)).numpy()
        trash_idx = str(pred.argmax() + 1)
        pred_class, confidence = TRASH_DICT[trash_idx], pred.max()

        return pred_class, confidence