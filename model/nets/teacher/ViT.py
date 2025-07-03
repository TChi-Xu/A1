from model.nets.models.Transformer import *
from model.nets.models.Read import *


class ViT(nn.Module):
    def __init__(self, *, image_size=256, patch_size=16, dim=1024, depth=12, heads=16, mlp_dim=2048, pool='cls', channels=1,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        """
        image_size: 输入图像的维度 
        patch_size: 分块Patch块大小
        dim: Transformer 隐层维度
        depth: Transformer的个数
        heads: 多头的个数
        mlp_dim: Transformer中的FeedForward中第一个线性层升维后的维度，默认为768*4，先升维4倍再降维回去
        pool: 默认‘cls’，选取CLS token作为输入， 可选‘Mean’, 在patch维度做平均池化
        channel: 输入图像的特征维度，通道数
        """
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert  image_height % patch_height ==0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}

        # 步骤一：图像分块与映射。首先将图片分块，然后接一个线性层做映射
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))					# nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.Reassemble1 = Read()

    def forward(self, img):
        x = self.to_patch_embedding(img)        # [1, 256, 1024]
        b, n, _ = x.shape           # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)               # [256+1, dim]---> [257, dim]
        x += self.pos_embedding[:, :(n+1)]                  # 加位置嵌入（直接加）      (b, 257, dim)
        x = self.dropout(x)

        x1, x2, x3 = self.transformer(x)    # (b, 257 , dim)

        x1,x2,x3 = self.Reassemble1(x1, x2, x3)

        return x1, x2, x3     # x1: 64*64*128; x2: 32*32*256; x3: 16*16*512


if __name__ == '__main__':
        x = torch.randn(1, 1, 256, 256)  # 创建随机输入张量
        model = ViT(channels=1)
        x1,x2,x3 = model(x)
        print(x1.shape, x2.shape, x3.shape)
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        # 打印模型输出的形状