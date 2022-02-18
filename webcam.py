import argparse
import torch
import torch.nn as nn
import resnet
import cv2
from utils import delimiter

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

parser = argparse.ArgumentParser(description='RecycleNet webcam inference')
parser.add_argument('--resume', default='save' + delimiter() + 'model_best.pth.tar', type=str)
parser.add_argument('--cuda', default=False, type=bool)
parser.add_argument('--save_dir', default='capture_img.jpg', type=str)
parser.add_argument('--use_att', action='store_true', help='use attention module')
parser.add_argument('--arch', type=str, default='resnet18_base', help='resnet18, 34, 50, 101, 152')
parser.add_argument('--no_pretrain', action='store_false', help='training from scratch')

args = parser.parse_args()

if torch.cuda.is_available() and args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = nn.DataParallel(resnet.resnet18(pretrained=True, use_att=args.use_att, num_classes=len(TRASH_DICT)))
if args.arch == 'resnet18_base':
    model = nn.DataParallel(resnet.resnet18(pretrained=not args.no_pretrain if not args.resume else False,
                                            use_att=args.use_att, num_classes=len(TRASH_DICT)))
elif args.arch == 'resnet34_base':
    model = nn.DataParallel(resnet.resnet34(pretrained=not args.no_pretrain if not args.resume else False,
                                            use_att=args.use_att, num_classes=len(TRASH_DICT)))
elif args.arch == 'resnet50_base':
    model = nn.DataParallel(resnet.resnet50(pretrained=not args.no_pretrain if not args.resume else False,
                                            use_att=args.use_att, num_classes=len(TRASH_DICT)))
elif args.arch == 'resnet101_base':
    model = nn.DataParallel(resnet.resnet101(pretrained=not args.no_pretrain if not args.resume else False,
                                             use_att=args.use_att, num_classes=len(TRASH_DICT)))
elif args.arch == 'resnet152_base':
    model = nn.DataParallel(resnet.resnet152(pretrained=not args.no_pretrain if not args.resume else False,
                                             use_att=args.use_att, num_classes=len(TRASH_DICT)))

checkpoint = torch.load(args.resume, map_location=device)
state_dict = checkpoint['state_dict']

model.load_state_dict(state_dict)
model.eval()


def inference(save_dir):
    frame = Image.open(save_dir)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    img_transforms = transforms.Compose([transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=MEAN, std=STD)])

    image_tensor = img_transforms(frame).float()
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor.to(device)

    softmax = nn.Softmax(dim=1)
    output = model(Variable(image_tensor))
    pred = torch.Tensor.cpu(softmax(output[0].data)).numpy()
    trash_idx = str(pred.argmax() + 1)
    pred_class, confidence = TRASH_DICT[trash_idx], pred.max()

    return pred_class, confidence


def main(save_dir):
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Unable to read camera feed")

    while True:
        ret, frame = cam.read()

        if ret:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == 32:  # Press 'space bar' to capture image
                cv2.imwrite(save_dir, frame)
                pred_class, confidence = inference(save_dir)
                print(f'Prediction: {pred_class}, Confidence: {confidence}')
            if cv2.waitKey(1) == 27:  # Press 'esc' to exit
                break
        else:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(args.save_dir)
