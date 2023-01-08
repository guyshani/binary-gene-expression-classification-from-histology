import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import copy
import time
import lightly

from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.models.utils import batch_shuffle# use a GPU if available
from lightly.models.utils import batch_unshuffle



# configuration

num_workers = 8
batch_size = 32
memory_bank_size = 4096
seed = 1
max_epochs = 20
acc = "gpu" if torch.cuda.is_available() else "cpu"
gpu_num = 2

path_to_train = '/home/maruvka/Documents/predict_expression/images/saved_tiles/normalized_tiles'
path_to_test = ''

pl.seed_everything(seed)




# class object for the moco model
class MocoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_splits=8)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco model based on ResNet
        self.projection_head = MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(
            self.projection_head, self.projection_head_momentum, 0.99
        )
        #print("2 :  " + str(torch.cuda.memory_summary(device=None, abbreviated=False)))
        
        # get queries
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # get keys
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log("train_loss_ssl", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, max_epochs
        )
        return [optim], [scheduler]







# $$$$$$$$$$$ data augmentations

# MoCo v2 uses SimCLR augmentations, additionally, disable blur
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=256,
    gaussian_blur=0.,
)
'''
# resizing and normalizing data for moco
moco_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256)),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage()
])
'''
# We use the moco augmentations for training moco
dataset_train_moco = lightly.data.LightlyDataset(
    input_dir=path_to_train
    #transform=moco_transforms
)

# $$$$$$$$  data loaders

dataloader_train_moco = torch.utils.data.DataLoader(
    dataset_train_moco,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)



#  run script
if __name__ == '__main__':
    start = time.time()

    torch.cuda.empty_cache()
    model = MocoModel()
    trainer = pl.Trainer(max_epochs=max_epochs, devices= gpu_num, accelerator= acc,
        auto_lr_find=True, auto_scale_batch_size=True)
    trainer.fit(
        model,
        dataloader_train_moco
    )

    trainer.save_checkpoint("saved_models/moco_test.ckpt")
    torch.save(model.input_embeddings.state_dict(),"saved_models/moco_test_input_embeddings.pt")
    torch.save(model.mlp.state_dict(), "saved_models/moco_test_mlp.pt")

    end = time.time()
    print("Total training time: "+ str(end-start))
    #new_model = MyModel.load_from_checkpoint(checkpoint_path="example.ckpt")


# create the "blank" networks like they
# were created in the Lightning Module
'''
input_embeddings = MultiEmbedding(...)
mlp = FullyConnectedModule(...)

# Load the models for inference
input_embeddings.load_state_dict(
    torch.load("input_embeddings.pt")
)
input_embeddings.eval()

mlp.load_state_dict(
    torch.load("mlp.pt")
)
mlp.eval()
'''
