import numpy as np
import torch

from lateral_connections import LateralModel, VggModel, CustomImageDataset

def main():
    model_path = 'models/VGG19_normalized_avg_pool_pytorch'
    # vm = VggModel(model_path, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    vm = VggModel(model_path, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), important_layers=['relu1_1', 'pool1', 'pool2'])
    model = LateralModel(vgg_model=vm, distance=2, num_output_repetitions=4, horizon_length=512)

    def torch_transform(img):
        return img.reshape((1,) + img.shape).float()

    ds = CustomImageDataset('images/geometric_dataset/annotations.csv', image_transform=torch_transform)

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # mean_h = list()
    # mean_s = list()
    # last_h = list()
    # last_s = list()

    # img, _ = ds[0]
    # model.forward(img)

    # model.save_model('blub.h5')
    # import code; code.interact(local=dict(globals(), **locals()))


    # exit()




    for i in tqdm(range(1024)):
        img, label = ds[np.random.randint(len(ds))]
        # img, label = ds[i]

        model.forward(img)

        # import code; code.interact(local=dict(globals(), **locals()))


        # mean_h.append(torch.mean(model.H).cpu().detach().numpy())
        # mean_s.append(torch.mean(model.S).cpu().detach().numpy())
        # last_h.append(model.S[0,0,0,-1].cpu().detach().numpy())
        # last_s.append(model.H[0,0,0,-1].cpu().detach().numpy())

    # import code; code.interact(local=dict(globals(), **locals()))


    # img, label = ds[256]

    # model.

    # fig, axs = plt.subplots(2,2, figsize=(10,10))
    # axs[0, 0].plot(mean_h, label='Mean H')
    # axs[0, 1].plot(mean_s, label='Mean S')
    # axs[1, 0].plot(last_h, label='Single Cell H')
    # axs[1, 1].plot(last_s, label='Single Cell S')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig('tmp.png')


    # import code; code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()
