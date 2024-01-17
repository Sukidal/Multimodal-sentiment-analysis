import torch
import numpy as np
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from util.data_preprocess import get_data_loader, calc_metrics
from model.CatModel import CatModel
from model.AddModel import AddModel
from model.AttentionCatModel import AttentionCatModel
from model.AttentionAddModel import AttentionAddModel

import argparse

import warnings
warnings.filterwarnings("ignore") 


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_only', action='store_true', help='only use text to train model, default false')
    parser.add_argument('--image_only', action='store_true', help='only use image to train model, default false')
    parser.add_argument('--do_test', action='store_true', help='use trained model to predict test dataset, default false')
    parser.add_argument('--lr', default=3e-5, help='set the learning rate, default 3e-5', type=float)
    parser.add_argument('--weight_decay', default=1e-5, help='set weight decay, default 1e-5', type=float)
    parser.add_argument('--epochs', default=5, help='set train epochs, default 5', type=int)
    parser.add_argument('--model', default='AttentionCatModel', help='set the type of model, default AttentionCatModel', type=str)
    args = parser.parse_args()
    return args


args = init_argparse()
print('args:', args)

assert((args.text_only and args.image_only) == False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)

torch.backends.cudnn.deterministic = True

# args = {'lr': 1e-2, 'epochs': 5, 'weight_decay': 1e-5, 'text_only': False, 'image_only': False}

def model_train():

    train_data_loader, valid_data_loader, test_data_loader = get_data_loader()
    model = None
    if args.model == 'AttentionCatModel':
        model = AttentionCatModel.from_pretrained('./pre_trained/bert-base-uncased')
    elif args.model == 'AttentionAddModel':
        model = AttentionAddModel.from_pretrained('./pre_trained/bert-base-uncased')
    elif args.model == 'AddModel':
        model = AddModel.from_pretrained('./pre_trained/bert-base-uncased')
    elif args.model == 'CatModel':
        model = CatModel.from_pretrained('./pre_trained/bert-base-uncased')
    else:
        print('sorry, no such model')
        return
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(lr=args.lr, params=optimizer_grouped_parameters)
    criterion = CrossEntropyLoss()
    best_rate = 0
    print('train start')
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        target_list = []
        pred_list = []
        model.train()
        for idx, (guid, tag, image, text) in enumerate(train_data_loader):
            tag = tag.to(device)
            image = image.to(device)
            text = text.to(device)
            out = None
            if args.text_only:
                out = model(image_input=None, text_input=text)
            elif args.image_only:
                out = model(image_input=image, text_input=None)
            else:
                out = model(image_input=image, text_input=text)
            loss = criterion(out, tag)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * len(guid)
            pred = torch.max(out, 1)[1]
            total += len(guid)
            correct += (pred == tag).sum()

            target_list.extend(tag.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())

        total_loss /= total
        print('epoch {:02d}'.format(epoch + 1), end='')
        print('train - loss:{:.6f}'.format(total_loss), end='')
        rate = correct / total * 100
        print('accuracy rate:{:.2f}%'.format(rate), end='')
        metrics = calc_metrics(target_list, pred_list)
        print(' precision: {:.2f}% f1 score: {:.2f}%'.format(metrics[0] * 100,metrics[2] * 100))

        total_loss = 0
        correct = 0
        total = 0
        target_list = []
        pred_list = []
        model.eval()

        for guid, tag, image, text in valid_data_loader:
            tag = tag.to(device)
            image = image.to(device)
            text = text.to(device)
            out = None
            if args.text_only:
                out = model(image_input=None, text_input=text)
            elif args.image_only:
                out = model(image_input=image, text_input=None)
            else:
                out = model(image_input=image, text_input=text)

            loss = criterion(out, tag)

            total_loss += loss.item() * len(guid)
            pred = torch.max(out, 1)[1]
            total += len(guid)
            correct += (pred == tag).sum()

            target_list.extend(tag.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())

        total_loss /= total
        print('eval - loss:{:.6f}'.format(total_loss), end='')
        rate = correct / total * 100
        print('accuracy rate:{:.2f}%'.format(rate), end='')
        metrics = calc_metrics(target_list, pred_list)
        print('precision:{:.2f}% f1 score: {:.2f}%'.format(metrics[0] * 100,metrics[2] * 100))

        if rate > best_rate:
            best_rate = rate
            print('save the best accuracy on val dataset:{:.2f}%'.format(rate))
            torch.save(model.state_dict(), args.model + '.pth')
        print()
    print('[END_OF_TRAINING_STAGE]')

def model_test():
    """利用训练好的./model.pth对测试集进行预测，结果保存至output/test_with_label.txt"""

    train_data_list, test_data_list = get_data_list()
    train_data_list, test_data_list = data_preprocess(train_data_list, test_data_list)
    train_data_loader, valid_data_loader, test_data_loader = get_data_loader(train_data_list, test_data_list)
    if args.model == 'AttentionCatModel':
        model = AttentionCatModel.from_pretrained('./pre_trained/bert-base-uncased')
    elif args.model == 'AttentionAddModel':
        model = AttentionAddModel.from_pretrained('./pre_trained/bert-base-uncased')
    elif args.model == 'AddModel':
        model = AddModel.from_pretrained('./pre_trained/bert-base-uncased')
    elif args.model == 'CatModel':
        model = CatModel.from_pretrained('./pre_trained/bert-base-uncased')
    else:
        print('sorry, no such model')
        return
    model.load_state_dict(torch.load(args.model + '.pth'))
    model.to(device)
    print('test start')
    guid_list = []
    pred_list = []
    model.eval()

    for guid, tag, image, text in test_data_loader:
        image = image.to(device)
        text = text.to(device)

        if args.text_only:
            out = model(image_input=None, text_input=text)
        elif args.image_only:
            out = model(image_input=image, text_input=None)
        else:
            out = model(image_input=image, text_input=text)

        pred = torch.max(out, 1)[1]
        guid_list.extend(guid)
        pred_list.extend(pred.cpu().tolist())

    pred_mapped = {
        0: 'negative',
        1: 'neutral',
        2: 'positive',
    }
    
    with open('output/' + args.model + '_test_with_label.txt', 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for guid, pred in zip(guid_list, pred_list):
            f.write(f'{guid},{pred_mapped[pred]}\n')
        f.close()
        print('save prediction to output/' + args.model + '_test_with_label.txt')
    print('test end')

if __name__ == "__main__":
    if args.do_test:
        model_test()
    else:
        model_train()
