import numpy as np
import torch
from torch.autograd import Variable
import itertools
# from EvalBox.Evaluation.evaluation import Evaluation
# from EvalBox.Evaluation.evaluation import MIN_COMPENSATION

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


            

            # print(mag.shape)


        # for index in range(len(directions)):
        #     direct = directions[index].view(1, C, H, W)
        #     new_x = X + mags * direct
        #     # print(new_x.shape)
        #     new_x = torch.clamp(new_x, 0, 1)
        #     with torch.no_grad():
        #         output = self.model(new_x)
        #     pred = torch.argmax(output, 1)
        #     # 如果全部的扰动都没有造成预测错误，则扰动取最大值
        #     # 若扰动产生预测错误，搜索最小预测错误扰动值
        #     epsilon = init_mags[-1]
        #     if((pred != y).data.sum().item() != 0):
        #         ind = (pred != y).cpu().numpy().tolist().index(1)
        #         epsilon = init_mags[ind]
            
        #     # 使用2-范数计算，sqrt(每个像素点扰动值^2 / 总像素点数)
        #     # torch.norm(A) = sqrt(a11^2 + a12^2 + ... + ann^2)
        #     bd = torch.norm(epsilon * direct / math.sqrt(C * H * W)).cpu().numpy()
        #     bd_list.append(bd)
            
        #     # 以方向做ranking时，取该方向所有距离取平均
        #     # direction_rst_list[index] += bd

        # 以图片做ranking时，取所有方向上最小的BD值
        # return min(bd_list)

    # def _measure_one_batch(self, X, y, directions, mags, init_mags):

    #     batch_size, C, H, W = X.shape
    #     directions = directions.view(self.direction_nums, C, H, W)
    #     for i in range(batch_size):
    #         new_X = (X[i].unsqueeze(0).expand(directions.shape) + directions)
            

    #         for j, direction in enumerate(directions):
    #             direction = direction.view(1, C, H, W)
    #             new_X = torch.clamp(X[i] + mags * direction, 0, 1)
    #             with torch.no_grad():
    #                 output = self.model(new_X)
    #             _, predicted = torch.max(output, 1)
    #             epsilon = init_mags[-1]


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




    def evaluate(self,X=None, y=None):
        '''
            Evaluate point to decision boundary
        '''
        total = len(X)
        print("total", total)
        device = self.device
        self.model.eval().to(device)
        N, C, H, W = X.shape
        assert N != 0, "X's length cann't be 0."
        vectors = self._orthogonal_vectors(C * H * W, self.direction_nums)
        directions = torch.from_numpy(vectors).float().to(device)
        mags, init_mags = self._mags()
        mags = torch.from_numpy(mags).float().to(device).view(-1, 1, 1, 1)
        
        distance = 0
        image_rst_list = []
        direction_rst_list = [0 for i in range(len(directions))]
        for i in tqdm(range(N)):
            rst = self._measure_one_image(X[i:i+1], y[i:i+1], directions, mags, init_mags, direction_rst_list)
            distance += rst
            image_rst_list.append(rst)
        
        # 数据处理部分，分段处理取min(20, k%)的数据做展示
        rate = 0.1
        dir_step = int(len(directions) // min(20, len(directions) * rate))
        img_step = int(total // min(20, total * rate))
        direction_rst_list = [float('{:.4f}'.format(x / total)) for x in direction_rst_list][::dir_step]
        image_rst_list = [float('{:.4f}'.format(x)) for x in image_rst_list][::img_step]
        
        with open("ebd.log", "a") as fp:
            fp.write("-------------------------------\n按Image排序的EBD值\n")
            image_rst_list.sort()
            fp.write("{}\n".format(image_rst_list))
            fp.write("按Direction排序的EBD值\n")
            direction_rst_list.sort()
            fp.write("{}\n".format(direction_rst_list))

        return distance / N