import torch
import torch.nn as nn
import resnet
from webcam import inference

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
        model = nn.DataParallel(resnet.resnet152(pretrained=pretrain, use_att=attention, num_classes=len(TRASH_DICT)))

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        checkpoint = torch.load(model_dir, map_location=device)
        state_dict = checkpoint['state_dict']

        model.load_state_dict(state_dict)
        model.eval()

    def classify(self, img_dir):
        pred_class, confidence = inference(img_dir)
        return f'Prediction: {pred_class}, Confidence: {confidence}'
