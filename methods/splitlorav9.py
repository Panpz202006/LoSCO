import math
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from copy import deepcopy
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from utils.losses import AugmentedTripletLoss
import schedulefree

from models.net_splitlora_R import Net
from models.vit_splitlora_R import VisionTransformer, PatchEmbed, Block, Attention_LoRA
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency
from utils.toolkit import print_args, format_elapsed_time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

class SplitLoRAV9(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        self.topk = 1
        self.network = Net(args)

        # inflora
        self.lamb = args["lamb"]
        self.lame = args["lame"]
        self.all_keys = []
        self.feature_list = []
        self.project_type = []
        self.in_dim = args['in_dim']
        self.alpha = args['alpha']
        self._protos = []
        self.rank = args['rank']
        self.atl = "true"
        # self.random_coeffs_K = nn.ParameterDict()
        # self.random_coeffs_V = nn.ParameterDict()
        self.margin_inter = 1.0
        # self.n_cur_matrix = 0
        # self.cur_matrix = torch.zeros(dim ,dim)
        # self.mds = True
        self.pca = False
        self.mds = False
        # self.pca = True
        self.lrate_head = args['lrate_head']
        self.lrate = args['lrate']



    def init_drm(self, train_loader):
        """initialzation of dimensionality reduction matrix A"""
        # 我不懂这里为啥又要再跑一遍训练集了，之前不是已经跑过一次了么？（在train函数中）
        # 并且为啥还执行了一次SVD分解。
        # for i, (_, inputs, targets) in enumerate(train_loader):
        #     inputs, targets = inputs.to(self.device), targets.to(self.device)
        #     self.network(inputs, get_cur_feat=True)  # gather the features in module.cur_matrix
        #     if self.debug: break
        
        if self.cur_task == 0:
                   
            return
        else:
            # 后续任务：使用上一个任务保存到的self.feature_list
            print("Using SplitLora Initialization for Task t(t>=1)")
            i = 0
            for module in self.network.modules():
                if isinstance(module, Attention_LoRA):
                    if self.mds:
                        # ########################   对initialized_matrix_K和initialized_matrix_V 进行重排列，避免特征冲突   #######################
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        # 对self.feature_list[i]采用PCA降低维度， 从 m,r 降维到 m,k
                        # 假设 self.feature_list[i] 是 shape (m, r) 的 tensor
                        features = self.feature_list[i].cpu().numpy()  # 转为 numpy
                        # 创建 PCA 对象，降到 k 维
                        pca = MDS(n_components=10)
                        features_reduced = pca.fit_transform(features)  # shape: (m, k)
                        initialized_matrix = torch.tensor(features_reduced, dtype=torch.float32).to(self.device)
                        # 第3行：归一化 + 缩放
                        initialized_matrix_K = initialized_matrix / initialized_matrix.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        initialized_matrix_V = initialized_matrix / initialized_matrix.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                    elif self.pca:
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        # 对self.feature_list[i]采用PCA降低维度， 从 m,r 降维到 m,k
                        # 假设 self.feature_list[i] 是 shape (m, r) 的 tensor
                        features = self.feature_list[i].cpu().numpy()  # 转为 numpy
                        # 创建 PCA 对象，降到 k 维
                        pca = PCA(n_components=10)
                        features_reduced = pca.fit_transform(features)  # shape: (m, k)
                        initialized_matrix = torch.tensor(features_reduced, dtype=torch.float32).to(self.device)
                        # 第3行：归一化 + 缩放
                        initialized_matrix_K = initialized_matrix / initialized_matrix.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        initialized_matrix_V = initialized_matrix / initialized_matrix.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                    
                    else:
                        random_coeffs_K = torch.randn(self.feature_list[i].shape[1],self.rank).to(module.lora_A_k[self.cur_task].weight.device)
                        random_coeffs_V = torch.randn(self.feature_list[i].shape[1],self.rank).to(module.lora_A_k[self.cur_task].weight.device)
                        # 第2行：用特征矩阵乘以随机系数
                        self.feature_list[i] = self.feature_list[i].to(self.device)
                        initialized_matrix_K = self.feature_list[i] @ random_coeffs_K
                        initialized_matrix_V = self.feature_list[i] @ random_coeffs_V

                        initialized_matrix_K = initialized_matrix_K / initialized_matrix_K.norm() * module.lora_A_k[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        initialized_matrix_V = initialized_matrix_V / initialized_matrix_V.norm() * module.lora_A_v[self.cur_task].weight.data.norm() * (1/(self.cur_task+1))
                        # 将UR矩阵的结果赋值给A（应该指的是 Y=X(W+AB中的A，即A<-UR)
                    
                    module.lora_A_k[self.cur_task].weight.data.copy_(initialized_matrix_K.T)   
                    module.lora_A_v[self.cur_task].weight.data.copy_(initialized_matrix_V.T)   


                    i += 1


        ########################计算权重的svd分解###########################
    def incremental_train(self, data_manager):
        self.data_manager = data_manager  #用于下面创建原型，用于ATL loss计算。
        self.build_train_loader(data_manager)
        logging.info('Task {} learning on class {}-{}'.format(self.cur_task, self.known_classes, self.total_classes))
        self._train(self.train_loader)

    def _extract_vectors(self, loader):
        self.network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self.network, nn.DataParallel):
                _vectors = tensor2numpy(self.network.module.extract_vector(_inputs.to(self.device)))
            else:
                _vectors = tensor2numpy(self.network.extract_vector(_inputs.to(self.device)))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _build_protos(self):
        print("collecting protos!")
        self.network.to(self.device)
        with torch.no_grad():
            for class_idx in range(self.known_classes, self.total_classes):
                # dataset_name, shuffle, seed, init_cls, increment,
                # data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
                # dataset_name=self.dataset_name, shuffle=self.shuffle, seed=self.seed, init_cls=self.init_cls, increment=self.increment = args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment']
                # data_manager = DataManager(dataset_name=self.dataset_name, shuffle=self.shuffle, seed=self.seed, init_cls=self.init_cls, increment=self.increment)
                idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                                           source='train',
                                                                           mode='test', ret_data=False)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)

                self._protos.append(class_mean)
                # del vectors, class_mean, data, targets, idx_dataset, idx_loader
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()        
    def _build_protos(self):
        self._protos = []
        self.network.to(self.device)
        with torch.no_grad():
            for class_idx in range(self.known_classes, self.total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                                           source='train',
                                                                           mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)

                self._protos.append(class_mean)
                
    def psv_info(self, train_loader):
        # 想要了解这个函数的功能，需要慢慢一步一步往下看。
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            #这里的forward函数中get_cur_feat=True在Attention_LoRA模块中注
            # 册hook来获取当前任务的激活矩阵（即cur_matrix）。
            self.network(inputs, get_cur_feat=True, get_weight=False)
            
        #这里的mat_list是一个列表，里面存储了每个Attention_LoRA模块的cur_matrix
        # （即当前任务的激活矩阵）。这个矩阵在前面训练过程中被更新了。
        mat_list = []
        for module in self.network.modules():
            # 判断：module 这个对象是不是 Attention_LoRA 类型的实例？
            # 返回值：True 或 False
            if isinstance(module, Attention_LoRA):
                mat_list.append(deepcopy(module.cur_matrix))
                n_cur_matrix = module.n_cur_matrix 
                print("self.n_cur_matrix:", n_cur_matrix)
                module.cur_matrix.zero_()
                module.n_cur_matrix = 0
        # 用Splitlora代替update_DualGPM(mat_list)
        for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = torch.linalg.svd(activation, full_matrices=False) #在这里接受输入激活值
                sval_total = (S).sum() #在这里计算总的能量
                sval_ratio = (S)/sval_total #在这里计算各个能量占得总的能量的比率
                k = torch.arange(1, self.in_dim+1).to(self.device)-1
                k = (self.in_dim-k.float()) / self.in_dim
                sval_ratio = sval_ratio.to(self.device)
                result = (self.cur_task+1)*(1-torch.cumsum(sval_ratio,dim=-1)) - self.alpha*k
                r = torch.argmin(result).item()

                # 这一步很多余，我感觉没必要！
                # temp = U[:,max(r,1)+1:]
                # Uf=torch.matmul(temp,temp.T)
                # self.feature_list[i] = Uf
                if i >= len(self.feature_list):
                    self.feature_list.append(U[:, max(r, 1) + 1:])
                else:
                    self.feature_list[i] = U[:, max(r, 1) + 1:]
        get_weight = False
        if get_weight:
            ########################计算权重的svd分解###########################
            for module in self.network.modules():
                # 判断：module 这个对象是不是 Attention_LoRA 类型的实例？
                # 返回值：True 或 False
                if isinstance(module, Attention_LoRA):
                    mat_list.append(deepcopy(module.cur_matrix))
                    self.weight_k = module.weight_k 
                    self.weight_v = module.weight_v

                    module.weight_k.zero_()
                    module.weight_v.zero_()


            ########################计算权重的svd分解###########################
        


                
    #这个是整个Splitlora算法的核心，其他都是参数传入、数据集载入、
    def _train(self, train_loader):
        self.network.to(self.device)
        # 保证B和分类池中的分类头是可学习的
        self.freeze_network()
        print_trainable_params(self.network)

        # Design LoRA matrix through Equation (8)
        with torch.no_grad():
            self.init_drm(train_loader)

        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)

        # 创建优化器和学习策略。
        # optimizer, scheduler = self.build_optimizer(self.network.parameters())
        trainable_params = self.get_trainable_params(self.network.parameters())

        optimizer = schedulefree.AdamWScheduleFree([
                    # trainable_params,
                    # lr=self.lrate,
                    # weight_decay=self.weight_decay,
                    # betas=(0.9,0.999)
        {'params': trainable_params[-2:], 'lr': self.lrate_head, 'weight_decay': self.weight_decay, 'betas': (0.9, 0.999)},
        {'params': trainable_params[:-2], 'lr': self.lrate,  'weight_decay': self.weight_decay, 'betas': (0.9, 0.999)}
                ])
        
        
        check_params_consistency(self.network, optimizer)

        # if self.atl:
        #     self._build_protos()

        # 从这里开始训练模型（主要是用来学习Y=X(W+AB)）中的B，A用U初试化(冻结)。
        self._train_function(train_loader, optimizer)

        #续编完成之后肯定是测试，因此为了让我们的optimizer能进入eval模式直接在这里设置就行了
        # optimizer.eval()
        
        if len(self.multiple_gpus) > 1:
            self.network = self.network.module

        # Preserve the information about the gradient of the t-th task through Splitlora
        with torch.no_grad():
            self.psv_info(train_loader)
        return

    def Weight_pri(weight, alpha=0.99, device=None):
        original_device = weight.device
        original_dtype = weight.dtype
        
        if device is not None:
            compute_device = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
            weight = weight.to(device=compute_device, dtype=torch.float32)
        else:
            weight = weight.to(dtype=torch.float32)

        # 对权重矩阵做 SVD
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        total_sum = S.sum()
        cumsum = S.cumsum(dim=0)
        r = (cumsum / total_sum >= alpha).nonzero(as_tuple=True)[0][0].item() + 1

        # print(r, end="")

        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        weight_r = torch.matmul(U_r, torch.matmul(torch.diag(S_r), Vh_r))
        
        return weight_r.to(device=original_device, dtype=original_dtype)
    
    def cosine_similarity(self, W_prev, delta):
        """计算余弦相似度，范围 [-1, 1]"""
        W_flat = W_prev.flatten()
        delta_flat = delta.flatten()
        cos_sim = torch.dot(W_flat, delta_flat) / (torch.norm(W_flat) * torch.norm(delta_flat) + 1e-8)
        return cos_sim
    
    def cosine_similarity_loss(self, W_prev, delta):
        # 展平
        W_flat = W_prev.flatten()
        delta_flat = delta.flatten()
        # 计算余弦相似度
        cos_sim = torch.dot(W_flat, delta_flat) / (torch.norm(W_flat) * torch.norm(delta_flat) + 1e-8)
        # 我们希望余弦相似度接近0（正交），所以 loss = |cos_sim|
        return torch.abs(cos_sim)
    
    def _train_function(self, train_loader, optimizer):
        #
        if self.cur_task>=1:
            ema_keys = ["Lora_shared_B_k", "Lora_shared_B_v", "Lora_shared_A_k", "Lora_shared_A_v"]
            
            # ema_keys = ["Lora_shared_B_k", "Lora_shared_B_v", "Lora_shared_A_k", "Lora_shared_A_v",
            # "lora_B_k", "lora_B_v"]
            # ema_keys = ["Lora_shared_v"]

            ema_decay = 0.9999
            ema_params = {}
            # ema2_params = {}，
            # 初始化EMA参数副本
            for name, param in self.network.named_parameters():
                if any(key in name for key in ema_keys):
                    # 冻结这些参数（不通过梯度更新）
                    # param.requires_grad_(False)
                    # 创建EMA副本
                    # .clone()：创建一个深拷贝，新张量有独立的内存空间
                    # .detach()：从计算图中分离，新张量不再与原始计算图有任何关联
                    ema_params[name] = param.data.clone().detach()
                    print(f"EMA tracking: {name}")


        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            # ######################### phase testing ##########################
            # import time
            # time_start = time.time()
            # accy, accy_with_task, accy_task = self.incremental_test(self.data_manager)
            # time_end = time.time()
            # logging.info('Evaluation time: {}'.format(format_elapsed_time(time_start, time_end)))

            # # logging
            # logging.info('Accuracy: {}'.format(accy['grouped']))
            # curve_accy, curve_accy_with_task, curve_accy_task = {'top1': []}, {'top1': []}, {'top1': []}

            # curve_accy['top1'].append(accy['top1'])
            # curve_accy_with_task['top1'].append(accy_with_task['top1'])
            # curve_accy_task['top1'].append(accy_task)
            # logging.info('Task: {}, epoch:{},  (curve) top1 Acc: {}'.format(self.cur_task, epoch ,curve_accy['top1']))  # Average Accuracy (A_t)
            # logging.info('Avergae Acc: {:.2f}'.format(np.mean(curve_accy['top1'])))  # Average Accuracy (A_t)
            
            # logging.info('(curve) top1 Acc with task: {}'.format(curve_accy_with_task['top1']))  # Average Accuracy with task id
            # logging.info('Avergae Acc: {:.2f}'.format(np.mean(curve_accy_with_task['top1'])))  # Average Accuracy (A_t)

            # logging.info('(curve) top1 Acc task: {}'.format(curve_accy_task['top1']))
            # logging.info('Avergae Acc: {:.2f}'.format(np.mean(curve_accy_task['top1'])))  # Average Accuracy (A_t)

            # logging.info('='*80)
            # ####################### phase testing ###########################

            self.network.train()
            optimizer.train()

            losses = 0.
            correct, total = 0, 0
            # mu = 0.01 * (0.95 ** epoch)  # 逐渐减小扰动
            print("\n")
            print(f"cur_task {self.cur_task}, Epoch {epoch}: "
                f"Head LR={optimizer.param_groups[0]['lr']:.6f}, "
                f"Backbone LR={optimizer.param_groups[1]['lr']:.6f}")
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                # labels = torch.index_select(targets, 0, mask)

                targets = torch.index_select(targets, 0, mask)-self.known_classes

                # # ========== ✅ 修正：只扰动可训练的B矩阵 ==========
                # # 1. 找出所有可训练参数（即B矩阵）
                # trainable_params = []
                # trainable_names = []
                # for name, param in self.network.named_parameters():
                #     if param.requires_grad:
                #         trainable_params.append(param)
                #         trainable_names.append(name)
                
                # # 2. 保存可训练参数的原始值
                # original_params = [p.data.clone() for p in trainable_params]
                
                # # # 3. 生成扰动并施加 (+μz)
                # perturbations = []
                # # for p in trainable_params:
                # #     z = torch.randn_like(p)
                # #     perturbations.append(z)
                # #     p.data.add_(mu * z)
                
                # # 5. 施加负扰动 (-μz)
                # for p, z in zip(trainable_params, perturbations):
                #     p.data.sub_(2 * mu * z)
                
                # # 6. 负扰动前向传播


                ret_minus = self.network(inputs)
                logits = (ret_minus['logits'])
                # print("logits shape and logits:", logits.shape, logits)
                loss_minus = F.cross_entropy(logits, targets)
                
                loss = loss_minus

                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.cur_task>=1:
                    # ✅ DEMA更新（改动最小，只改了这里）
                    for name, param in self.network.named_parameters():
                        if name in ema_params:
                            # 第一次EMA（标准EMA）
                            # print(self.cur_task, name, ema_params[name])
                                
                            ema_params[name] = ema_decay * ema_params[name] + (1 - ema_decay) * param.data
                            new_value, old_value = ema_params[name], param.data
                            # if not torch.allclose(new_value, old_value):
                            #     print(f"\n📊 应用EMA到 {name}:")
                            #     print(f"  new_value: {new_value.flatten()[:5].tolist()}...")  # 打印完整精度
                            #     print(f"  old_value: {old_value.flatten()[:5].tolist()}...")
                            #     print(f"  L2差异: {(new_value - old_value).norm().item():.17f}")
                            #     print(f"  相对变化: {((new_value - old_value).norm() / (old_value.norm() + 1e-8)).item():.12%}")
                            # else:
                            #     print(f"\n✅ {name}: 替换前后值相同（无变化）")

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            # scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)



        logging.info(info)
        # if self.cur_task>=1:
        #     # 可选：训练结束后将EMA参数应用到模型
        #     with torch.no_grad():
        #         for name, param in self.network.named_parameters():
        #             if name in ema_params:
        #                 param.data.copy_(ema_params[name])
        #                 print(f"Applied EMA to {name}")
        if self.cur_task>=1:
            # 可选：训练结束后将EMA参数应用到模型
            with torch.no_grad():
                for name, param in self.network.named_parameters():
                    if name in ema_params:
                        # 获取替换前的值
                        old_value = param.data.clone()
                        # 获取EMA值
                        ema_value = ema_params[name]
                        
                        # # 检查是否相等
                        # if not torch.allclose(old_value, ema_value):
                        #     print(f"\n📊 应用EMA到 {name}:")
                        #     print(f"  替换前: {old_value.flatten()[:5]}...")  # 只打印前5个元素
                        #     print(f"  替换后: {ema_value.flatten()[:5]}...")
                        #     print(f"  L2差异: {(old_value - ema_value).norm().item():.6f}")
                        #     print(f"  相对变化: {((old_value - ema_value).norm() / (old_value.norm() + 1e-8)).item():.6f}")
                        # else:
                        #     print(f"\n✅ {name}: 替换前后值相同（无变化）")
                        
                        # 执行替换
                        param.data.copy_(ema_value)


    def freeze_network(self):
        target_suffix = f".{self.cur_task}"
        if self.cur_task==0:
            unfrozen_keys = [
            f"classifier_pool{target_suffix}",
            f"lora_B_k{target_suffix}",
            f"lora_B_v{target_suffix}",
            "Lora_shared",
            

            # f"lora_M_k{target_suffix}",      # ✅ 新增
            # f"lora_M_v{target_suffix}",      # ✅ 新增
        ]
        else:
            unfrozen_keys = [
            f"classifier_pool{target_suffix}",

            f"lora_B_k{target_suffix}",
            f"lora_B_v{target_suffix}",
            "Lora_shared"
            # f"lora_M_k{target_suffix}",      # ✅ 新增
            # f"lora_M_v{target_suffix}",      # ✅ 新增

        ]
        for name, param in self.network.named_parameters():
            # print(name) 
            param.requires_grad_(any(key in name for key in unfrozen_keys))
    
    

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)




