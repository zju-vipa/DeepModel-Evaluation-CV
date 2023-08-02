import numpy as np
import torch
from torch.autograd import Variable
import itertools

import math
from tqdm import tqdm

class EBD:

    #
    def __init__(self, model=None, platform="pytorch", clean_label_file=None, **kwargs, ):
        '''
        @description:
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        '''
        self.model = model
        self.device = next(self.model.parameters()).device
        # self.clean_labels =[int(i) for i in open(clean_label_file).read().splitlines()]
        self.direction_nums = 500
        self.max_mag = 255.5
        self.platform=platform


    def _orthogonal_vectors(self, vector_size, vector_num):
        '''

        :param vector_size
        :param vector_num
        '''
        randmat = np.random.normal(size=(vector_size, vector_num))
        q, _ = np.linalg.qr(randmat)
        vectors = q.T * np.sqrt(float(vector_size))

        return vectors
    
    def _mags(self):
        '''

        :param None
        '''
        mags = np.arange(0, self.max_mag, 4, dtype=np.float32)
        mags = mags / 255.0
        init_mags = mags.copy()
        return mags, init_mags
    
    def _measure_one_image(self, X, y, directions, mags, init_mags, direction_norm):
        '''
        @description: 
        @param {
            x:
            y:
            directions:
            mags:
            init_mags:
        } 
        @return: farthest {the worst boundary distance}
        '''
        C, H, W = X.shape[1:]
        # 对每张图片，计算所有的方向中，最小的BD值

        # print(directions.shape)

        min_mag = mags[-1]
        for mag in mags:
            shift = (mag * directions).view(self.direction_nums, C, H, W)
            # print(shift.cpu())
            shifted_X = X + shift

            with torch.no_grad():
                output = self.model(shifted_X)

            if self.platform == "huggingface":
                output = output.logits
            
            pred = torch.argmax(output, 1)
            if (pred != y).data.sum().item() != 0:
                min_mag = mag
                break

        return min_mag * direction_norm



    def apply(self, eval_loader=None):
        total = len(eval_loader)
        print("total", total)
        assert total != 0, "eval_loader is empty!"
        device = self.device
        eval_loader_dup = itertools.tee(eval_loader, 1)[0]
        X, _ = next(eval_loader_dup)
        print(X.shape)
        batch_size, channel, height, width = X.shape
        orthogonal_vectors = self._orthogonal_vectors(channel * height * width, self.direction_nums)
        direction_norm = np.linalg.norm(orthogonal_vectors[0])
        directions = torch.from_numpy(orthogonal_vectors).float().to(device)
        mags, init_mags = self._mags()
        mags = torch.from_numpy(mags).float().to(device)


        self.model.eval()
        result_list = []


        total_index = 0
        # for X, y in tqdm(eval_loader):
        for X, y in eval_loader:
            X = X.to(device)
            y = y.to(device)
            for index in range(len(X)):
                # clean_label = self.clean_labels[total_index + index]
                # if clean_label != y[index].cpu().item():
                #     print('Error predition, continue')
                #     continue
                result = self._measure_one_image(X[index: index + 1], y[index: index + 1], directions, mags, init_mags, direction_norm)
                if result.item() != 0:
                    print(result.item(), flush=True)
                result_list.append(result)

            total_index += len(X)

