import torch
import torch.nn as nn
from transformers import ResNetModel
from transformers import BertModel, BertPreTrainedModel, BertLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AddModel(BertPreTrainedModel):

    def __init__(self, config):
        super(AddModel, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.resnet = ResNetModel.from_pretrained("./pre_trained/resnet-152")
        self.W = nn.Linear(in_features=2048, out_features=config.hidden_size)
        self.classifier_single = nn.Linear(in_features=config.hidden_size, out_features=3)

    def forward(self, image_input = None, text_input = None):
        if (image_input is not None) and (text_input is not None):
            """both image and text"""

            """提取文本特征"""
            text_features = self.bert(**text_input)
            text_hidden_state, _ = text_features.last_hidden_state.max(1)
            

            """提取图像特征"""
            t = self.resnet(**image_input).last_hidden_state
            image_features = t.view(-1, 2048, 49).permute(0, 2, 1).contiguous()
            image_pooled_output, _ = image_features.max(1)
            image_hidden_state = self.W(image_pooled_output).unsqueeze(1).view(-1, self.hidden_size)
            """将文本和图像特征相加，得到共同特征"""
            image_text_hidden_state = torch.add(image_hidden_state, text_hidden_state)

            """利用结果向量进行分类"""
            out = self.classifier_single(image_text_hidden_state)
            return out

        elif image_input is None:
            """text only"""
            assert(text_input is not None)
            text_features = self.bert(**text_input)
            text_hidden_state, _ = text_features.last_hidden_state.max(1)
            
            out = self.classifier_single(text_hidden_state)
            return out

        elif text_input is None:
            """image only"""
            assert(image_input is not None)
            image_features = self.resnet(**image_input).last_hidden_state.view(-1, 2048, 49).permute(0, 2, 1).contiguous()
            image_pooled_output, _ = image_features.max(1)
            image_hidden_state = self.W(image_pooled_output).unsqueeze(1).view(-1, self.hidden_size)
            
            out = self.classifier_single(image_hidden_state)
            return out