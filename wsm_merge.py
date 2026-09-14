"""
WSM (Warmup-Stable-Merge) 风格离线合并
用于持续学习中对多个任务检查点进行加权平均

使用方法:
    python wsm_merge.py --config configs/splitlorav6/imagenet-r_seed=1993.json

功能:
    1. 加载所有任务检查点
    2. 支持多种合并策略: uniform, exponential, cosine, linear, sqrt
    3. 自动评估合并后的模型
    4. 对比不同策略的性能
"""

import os
import math
import json
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.toolkit import set_device, set_random, setup_logging, print_args
from utils.factory import get_model
from dataloaders.data_manager import DataManager


def load_checkpoints(checkpoint_dir, num_tasks, device='cpu'):
    """
    加载所有任务的检查点
    
    Args:
        checkpoint_dir: 检查点目录
        num_tasks: 总任务数
        device: 目标设备
    
    Returns:
        checkpoints: list of dict, 每个dict包含模型参数
    """
    checkpoints = []
    for task_id in range(num_tasks):
        save_path = os.path.join(checkpoint_dir, f'task_{task_id}_checkpoint.pt')
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=device)
            checkpoints.append(checkpoint)
            logging.info(f"Loaded checkpoint for task {task_id} from {save_path}")
        else:
            logging.warning(f"Checkpoint for task {task_id} not found at {save_path}")
    
    return checkpoints


def compute_merge_weights(num_checkpoints, strategy='exponential', decay=0.5, temperature=1.0):
    """
    计算WSM风格的合并权重
    
    Args:
        num_checkpoints: 检查点数量
        strategy: 'uniform', 'exponential', 'cosine', 'linear', 'sqrt'
        decay: 指数衰减的衰减率（仅exponential使用）
        temperature: 温度参数，控制权重分布的陡峭程度
    
    Returns:
        weights: list of float, 归一化后的权重
    """
    if num_checkpoints == 0:
        return []
    
    weights = []
    
    if strategy == 'uniform':
        # 均匀权重：所有检查点等权
        weights = [1.0 / num_checkpoints] * num_checkpoints
        
    elif strategy == 'exponential':
        # 指数衰减：越新的检查点权重越大
        for i in range(num_checkpoints):
            # i=0 是最早的，i=num_checkpoints-1 是最新的
            w = math.exp(-decay * (num_checkpoints - 1 - i))
            weights.append(w)
        # 归一化
        total = sum(weights)
        weights = [w / total for w in weights]
        
    elif strategy == 'cosine':
        # 余弦权重：最新的检查点权重最大，最早的最小
        for i in range(num_checkpoints):
            progress = i / (num_checkpoints - 1) if num_checkpoints > 1 else 1.0
            cos_val = (1 - math.cos(math.pi * progress * temperature)) / 2
            weights.append(cos_val)
        total = sum(weights)
        weights = [w / total for w in weights]
        
    elif strategy == 'linear':
        # 线性递增：从旧到新线性增加权重
        for i in range(num_checkpoints):
            weights.append(i + 1)
        total = sum(weights)
        weights = [w / total for w in weights]
        
    elif strategy == 'sqrt':
        # 平方根递减：介于均匀和指数之间
        for i in range(num_checkpoints):
            w = 1.0 / math.sqrt(i + 1)
            weights.append(w)
        total = sum(weights)
        weights = [w / total for w in weights]
    
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")
    
    return weights


def merge_checkpoints(checkpoints, weights, device='cpu'):
    """
    执行检查点合并（仅合并共享适配器参数）
    
    Args:
        checkpoints: list of dict, 每个dict包含模型参数
        weights: list of float, 归一化后的权重
        device: 目标设备
    
    Returns:
        merged: dict, 合并后的模型参数
    """
    if len(checkpoints) != len(weights):
        raise ValueError(f"Checkpoints count ({len(checkpoints)}) != weights count ({len(weights)})")
    
    if len(checkpoints) == 0:
        return None
    
    # 只合并共享适配器参数
    merged = {}
    
    # 获取所有检查点共有的参数名
    all_keys = []
    for ckpt in checkpoints:
        for name in ckpt.keys():
            if 'Lora_shared' in name and name not in all_keys:
                all_keys.append(name)
    
    if len(all_keys) == 0:
        logging.warning("No shared adapter parameters found in checkpoints")
        # 回退：合并所有参数（排除非参数键）
        all_keys = [k for k in checkpoints[0].keys() 
                    if k not in ['task_id', 'known_classes', 'total_classes', 'ema_params']]
    
    # 加权平均
    for name in all_keys:
        # 确保所有checkpoint都有这个参数
        has_all = all(name in ckpt for ckpt in checkpoints)
        if not has_all:
            logging.warning(f"Parameter {name} not in all checkpoints, skipping")
            continue
        
        merged[name] = weights[0] * checkpoints[0][name]
        for i in range(1, len(checkpoints)):
            merged[name] += weights[i] * checkpoints[i][name]
        merged[name] = merged[name].to(device)
    
    logging.info(f"Merged {len(merged)} shared adapter parameters")
    
    return merged


def apply_merged_weights_to_network(model, merged_weights):
    """
    将合并后的权重应用到网络中
    
    Args:
        model: 模型实例
        merged_weights: dict, 合并后的模型参数
    """
    if merged_weights is None:
        return
    
    with torch.no_grad():
        for name, param in model.network.named_parameters():
            if name in merged_weights:
                param.data.copy_(merged_weights[name].to(param.device))
    
    logging.info("Applied merged weights to network")


def evaluate_model(model, data_manager, device='cuda'):
    """
    评估模型的持续学习性能
    
    Args:
        model: 模型实例
        data_manager: 数据管理器
        device: 设备
    
    Returns:
        dict: 包含各种准确率指标
    """
    # ========== 保存原始状态 ==========
    original_cur_task = model.cur_task
    original_known_classes = model.known_classes
    original_total_classes = model.total_classes
    
    # ========== 设置为最后一个任务（所有任务已学习） ==========
    num_tasks = data_manager.task_num
    model.cur_task = num_tasks - 1
    model.known_classes = model.cur_task * model.increment
    model.total_classes = (model.cur_task + 1) * model.increment
    
    # ========== 关键：确保分类头已构建 ==========
    # 调用 before_task 来初始化当前任务的分类器
    model.before_task(data_manager)
    
    model.network.to(device)
    model.network.eval()
    
    try:
        # 调用模型自身的测试方法
        accy, accy_with_task, accy_task = model.incremental_test(data_manager)
        
        result = {
            'top1': accy['top1'],
            'grouped': accy['grouped'],
            'with_task': accy_with_task['top1'],
            'task': accy_task
        }
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        # 恢复状态并重新抛出
        model.cur_task = original_cur_task
        model.known_classes = original_known_classes
        model.total_classes = original_total_classes
        raise
    
    # ========== 恢复原始状态 ==========
    model.cur_task = original_cur_task
    model.known_classes = original_known_classes
    model.total_classes = original_total_classes
    
    return result


def get_available_checkpoints(checkpoint_dir, num_tasks):
    """获取可用的检查点列表"""
    available = []
    for task_id in range(num_tasks):
        save_path = os.path.join(checkpoint_dir, f'task_{task_id}_checkpoint.pt')
        if os.path.exists(save_path):
            available.append(task_id)
    return available


def run_wsm_merge(args):
    """
    运行 WSM 风格离线合并的主函数
    """
    # ========== 处理 seed（可能是列表） ==========
    if isinstance(args.get('seed'), list):
        args['seed'] = args['seed'][0] if args['seed'] else 1993
    if isinstance(args.get('seed'), str):
        args['seed'] = int(args['seed'])
    
    # 设置设备和随机种子
    set_random(args)
    set_device(args)
    
    # 设置日志
    logfilename = os.path.join(args['logdir'], f'seed{args["seed"]}')
    setup_logging(logfilename, args.get('save_ckp', False))
    
    logging.info("=" * 60)
    logging.info("WSM Offline Merge Evaluation")
    logging.info("=" * 60)
    print_args(args)
    
    # 创建数据管理器
    data_manager = DataManager(
        args['dataset'], args['shuffle'], args['seed'], 
        args['init_cls'], args['increment'], args
    )
    
    # 加载预训练模型
    logging.info("Loading pretrained model...")
    model = get_model(args['method'], args)
    model.network.to(args['device'][0])
    
    # ========== 设置模型状态为最后一个任务 ==========
    num_tasks = data_manager.task_num
    model.cur_task = num_tasks - 1
    model.known_classes = model.cur_task * args['increment']
    model.total_classes = (model.cur_task + 1) * args['increment']
    model.increment = args['increment']
    
    # ========== 关键：构建分类头 ==========
    model.before_task(data_manager)
    
    # 如果有多个 GPU
    if len(args['device']) > 1:
        model.network = nn.DataParallel(model.network, args['device'])
    
    # 检查点目录
    checkpoint_dir = os.path.join(args['logdir'], 'task_checkpoints')
    
    # 获取可用检查点
    available_tasks = get_available_checkpoints(checkpoint_dir, num_tasks)
    logging.info(f"Available checkpoints: tasks {available_tasks}")
    
    if len(available_tasks) == 0:
        logging.error("No checkpoints found! Please train the model first.")
        return
    
    # 加载检查点
    checkpoints = load_checkpoints(checkpoint_dir, num_tasks, device='cpu')
    logging.info(f"Loaded {len(checkpoints)} checkpoints")
    
    device = args['device'][0]
    
    # ========== 评估基线：最后一个检查点 ==========
    logging.info("-" * 60)
    logging.info("Evaluating baseline (last checkpoint)...")
    
    last_ckpt = checkpoints[-1]
    merged_last = {}
    for k, v in last_ckpt.items():
        if 'Lora_shared' in k:
            merged_last[k] = v.to(device)
    
    apply_merged_weights_to_network(model, merged_last)
    baseline_results = evaluate_model(model, data_manager, device)
    logging.info(f"Last checkpoint accuracy: {baseline_results['top1']:.2f}%")
    
    # ========== 定义合并策略 ==========
    strategies = [
        {'name': 'uniform', 'strategy': 'uniform'},
        {'name': 'exp_0.3', 'strategy': 'exponential', 'decay': 0.3},
        {'name': 'exp_0.5', 'strategy': 'exponential', 'decay': 0.5},
        {'name': 'exp_0.7', 'strategy': 'exponential', 'decay': 0.7},
        {'name': 'exp_0.9', 'strategy': 'exponential', 'decay': 0.9},
        {'name': 'cosine', 'strategy': 'cosine', 'temperature': 1.0},
        {'name': 'cosine_0.5', 'strategy': 'cosine', 'temperature': 0.5},
        {'name': 'linear', 'strategy': 'linear'},
        {'name': 'sqrt', 'strategy': 'sqrt'},
    ]
    
    # ========== 评估所有策略 ==========
    logging.info("-" * 60)
    logging.info("Evaluating merge strategies...")
    
    results = {'baseline_last': baseline_results['top1']}
    
    for config in strategies:
        strategy_name = config['name']
        strategy_type = config['strategy']
        
        # 提取参数
        kwargs = {k: v for k, v in config.items() if k not in ['name', 'strategy']}
        weights = compute_merge_weights(len(checkpoints), strategy_type, **kwargs)
        
        if weights is None:
            logging.warning(f"Failed to compute weights for {strategy_name}")
            continue
        
        logging.info(f"  Testing: {strategy_name}")
        logging.info(f"    Weights: {[f'{w:.4f}' for w in weights]}")
        
        # 合并
        merged = merge_checkpoints(checkpoints, weights, device)
        
        # 应用并评估
        apply_merged_weights_to_network(model, merged)
        eval_results = evaluate_model(model, data_manager, device)
        
        results[strategy_name] = {
            'weights': weights,
            'accuracy': eval_results['top1'],
            'grouped': eval_results['grouped']
        }
        
        logging.info(f"    Accuracy: {eval_results['top1']:.2f}%")
    
    # ========== 打印结果汇总 ==========
    logging.info("=" * 60)
    logging.info("WSM Merge Results Summary")
    logging.info("=" * 60)
    logging.info(f"Baseline (last checkpoint): {results['baseline_last']:.2f}%")
    logging.info("-" * 40)
    
    # 按准确率排序
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if k != 'baseline_last'],
        key=lambda x: x[1]['accuracy'] if isinstance(x[1], dict) else x[1],
        reverse=True
    )
    
    for name, result in sorted_results:
        if isinstance(result, dict):
            acc = result['accuracy']
            logging.info(f"{name:20s} : {acc:.2f}%")
        else:
            logging.info(f"{name:20s} : {result:.2f}%")
    
    # ========== 保存结果 ==========
    result_path = os.path.join(args['logdir'], 'wsm_merge_results.json')
    # 转换数据为可序列化格式
    save_results = {}
    for k, v in results.items():
        if isinstance(v, dict) and 'weights' in v:
            save_results[k] = {
                'accuracy': v['accuracy'],
                'grouped': v.get('grouped', {}),
                'weights': [float(w) for w in v['weights']]
            }
        else:
            save_results[k] = v
    
    with open(result_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    logging.info(f"Results saved to {result_path}")
    logging.info("=" * 60)
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='WSM Offline Merge')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_ckp', action='store_true')
    
    return parser.parse_args()


def load_json(settings_path):
    with open(settings_path) as data_file:
        return json.load(data_file)


if __name__ == '__main__':
    args = parse_args()
    param = load_json(args.config)
    args = vars(args)
    args.update(param)
    
    # ========== 处理 seed ==========
    if isinstance(args.get('seed'), list):
        args['seed'] = args['seed'][0] if args['seed'] else 1993
    if isinstance(args.get('seed'), str):
        args['seed'] = int(args['seed'])
    
    # ========== 补全缺失的配置字段 ==========
    default_fields = {
        'moe_list': None,
        'ema_decay2': 0.0,
        'ema_decay_start': 0.9,
        'ema_decay_end': 0.9999,
        'ema_decay_schedule': 'cosine',
        'off_merge_knob': False,
        'grad_ema_decay': 0.99,
        'grad_direction_eta': 0.5,
    }
    for key, default_value in default_fields.items():
        if key not in args:
            args[key] = default_value
    
    # 处理 device
    if isinstance(args.get('device'), str):
        args['device'] = args['device'].split(',')
    
    # 创建日志目录
    from utils.toolkit import make_logdir
    args['logdir'] = make_logdir(args)
    
    run_wsm_merge(args)