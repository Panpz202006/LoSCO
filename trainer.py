import os
import os.path
import time
import torch
import logging

from utils.factory import get_model
from utils.toolkit import print_args, format_elapsed_time
from utils.toolkit import setup_logging, set_device, set_random
from dataloaders.data_manager import DataManager
import numpy as np

def start(args):
    data_manager, model = initialize(args)
    train(data_manager, model, args)


def initialize(args):
    # log
    args['logfilename'] = os.path.join(args['logdir'], 'seed{}'.format(args['seed']))
    setup_logging(args['logfilename'], args['save_ckp'])

    # random seed and device
    set_random(args)
    set_device(args)
    print_args(args)

    # datamanager
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], 
                               args['increment'], args)
    # model
    model = get_model(args['method'], args)

    return data_manager, model


def train(data_manager, model, args):

    curve_accy, curve_accy_with_task, curve_accy_task = {'top1': []}, {'top1': []}, {'top1': []}
    
    # ========== 打印数据集概览 ==========
    logging.info('=' * 80)
    logging.info('Dataset Information')
    logging.info('=' * 80)
    logging.info('  Dataset name     : {}'.format(args['dataset']))
    logging.info('  Data path        : {}'.format(args.get('data_path', 'N/A')))
    logging.info('  Total classes    : {}'.format(data_manager.total_class_num))
    logging.info('  Total tasks      : {}'.format(data_manager.task_num))
    logging.info('  Task increments  : {}'.format(data_manager._increments))
    logging.info('  Init classes     : {}'.format(args['init_cls']))
    logging.info('  Increment per task: {}'.format(args['increment']))
    logging.info('  Shuffle          : {}'.format(args.get('shuffle', False)))
    logging.info('  Seed             : {}'.format(args.get('seed', 'N/A')))
    logging.info('-' * 80)
    
    logging.info('Task split:')
    cumsum = 0
    for task_id in range(data_manager.task_num):
        task_size = data_manager.get_task_size(task_id)
        class_start = cumsum
        class_end = cumsum + task_size - 1
        
        # 获取该任务对应的训练集和测试集图片数量
        try:
            # 获取该任务所有类别的数据集
            train_dataset = data_manager.get_dataset(
                np.arange(class_start, class_end + 1), 
                source='train', 
                mode='test'
            )
            test_dataset = data_manager.get_dataset(
                np.arange(class_start, class_end + 1), 
                source='test', 
                mode='test'
            )
            train_count = len(train_dataset)
            test_count = len(test_dataset)
            batch_size = 256

            # 计算train_loader和test_loader的长度（batch数量）
            train_loader_len = (train_count + batch_size - 1) // batch_size  # 向上取整
            test_loader_len = (test_count + batch_size - 1) // batch_size    # 向上取整
            
        except:
            # 如果获取失败，显示占位符
            train_count = 'N/A'
            test_count = 'N/A'
            train_loader_len = 'N/A'
            test_loader_len = 'N/A'
        
        logging.info('  Task {}: classes {}-{} ({} classes) | Train: {} images ({} batches), Test: {} images ({} batches)'.format(
            task_id, class_start, class_end, task_size, 
            train_count, train_loader_len, 
            test_count, test_loader_len))
        cumsum += task_size
    logging.info('=' * 80)

    curve_accy, curve_accy_with_task, curve_accy_task = {'top1': []}, {'top1': []}, {'top1': []}
    # print(data_manager)
    # Train and Eval sequentially for N tasks
    for task in range(data_manager.task_num):
        logging.info('='*80)
        model.before_task(data_manager)

        # learning on the new task (train)
        time_start = time.time()
        model.incremental_train(data_manager)
        time_end = time.time()
        logging.info('Training time: {}'.format(format_elapsed_time(time_start, time_end)))

        # evaluate the model (eval)
        time_start = time.time()
        accy, accy_with_task, accy_task = model.incremental_test(data_manager)
        time_end = time.time()
        logging.info('Evaluation time: {}'.format(format_elapsed_time(time_start, time_end)))

        model.after_task()

        # logging
        logging.info('Accuracy: {}'.format(accy['grouped']))
        curve_accy['top1'].append(accy['top1'])
        curve_accy_with_task['top1'].append(accy_with_task['top1'])
        curve_accy_task['top1'].append(accy_task)
        logging.info('(curve) top1 Acc: {}'.format(curve_accy['top1']))  # Average Accuracy (A_t)
        logging.info('Avergae Acc: {:.2f}'.format(np.mean(curve_accy['top1'])))  # Average Accuracy (A_t)

        logging.info('(curve) top1 Acc with task: {}'.format(curve_accy_with_task['top1']))  # Average Accuracy with task id
        logging.info('Avergae Acc: {:.2f}'.format(np.mean(curve_accy_with_task['top1'])))  # Average Accuracy (A_t)

        logging.info('(curve) top1 Acc task: {}'.format(curve_accy_task['top1']))
        logging.info('Avergae Acc: {:.2f}'.format(np.mean(curve_accy_task['top1'])))  # Average Accuracy (A_t)

        logging.info('='*80)

        # save model
        torch.save(model.network.state_dict(), os.path.join(args['logfilename'], "task_{}.pth".format(int(task)))) if args['save_ckp'] else None


        # model.load_all_task_weight_merge()
