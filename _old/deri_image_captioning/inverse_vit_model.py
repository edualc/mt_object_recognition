import torch
from transformers import ViTFeatureExtractor, AutoTokenizer
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import VisionEncoderDecoderModel


class InverseUSR(torch.nn.Module):

    def __init__(
            self,
            device,
            height: int = 224,
            width: int = 224
    ):
        super().__init__()
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        self.height = height
        self.width = width

        self.mean = torch.tensor(self.feature_extractor.image_mean, device=device)
        self.std = torch.tensor(self.feature_extractor.image_std, device=device)

        self.image_pix = self.init_response_weight()
        self.fix_usr()

    def init_response_weight(self):

        w = torch.empty(3, self.height, self.width)
        torch.nn.init.uniform_(w, 0, 1.0)

        params = torch.nn.Parameter(w)
        params.requires_grad = True
        return params

    def fix_usr(self):
        for param in list(self.model.parameters()):
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels):
        wmin = self.image_pix.min()
        wmax = self.image_pix.max()
        #
        # image = (self.image_pix - wmin) / (wmax - wmin)

        image = torch.tanh(torch.relu(self.image_pix))
        image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        usr_out = self.model(pixel_values=image[None, :, :, :], labels=labels)
        return usr_out