import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.utils.data as tordata
from modeling import models
from utils import (
    config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr,
    get_attr_from, get_valid_args,
)
from utils import evaluation as eval_functions
from data.dataset_briar import DataSetBRIAR
import data.sampler as Samplers
from data.collate_fn import CollateFn
from modeling.base_model import BaseModel

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str,
                    default='config/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train', choices=['train', 'test'],
                    help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/"
                    "<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
opt = parser.parse_args()


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/',
                               cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'],
                               engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path,
                             opt.log_to_file,
                             engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if
                             isinstance(engine_cfg['restore_hint'],
                                        (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)

    return msg_mgr


def save_failure_sequences(probe_x_fail_indices, loader):
    for ii, inputs in enumerate(loader):
        inputs_list, labels, _, _, seq_length = inputs
        silhouettes = inputs[0]

        if np.isin(probe_x_fail_indices, ii):
            print(silhouettes.shape)


def run_model(cfgs, loader, msg_mgr, training):
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    model = get_ddp_module(model)
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        run_train(model, msg_mgr, loader)
    else:
        run_test(model, msg_mgr, loader)


def setup_loader(cfgs, msg_mgr, train=True):

    sampler_cfg = (cfgs['trainer_cfg']['sampler'] if
                   train else cfgs['evaluator_cfg']['sampler'])
    dataset = DataSetBRIAR(cfgs, train)

    Sampler = get_attr_from([Samplers], sampler_cfg['type'])
    vaild_args = get_valid_args(Sampler,
                                sampler_cfg,
                                free_keys=['sample_type', 'type'])
    sampler = Sampler(dataset, **vaild_args)

    loader = tordata.DataLoader(dataset=dataset,
                                batch_sampler=sampler,
                                collate_fn=CollateFn(dataset.label_set,
                                                     sampler_cfg),
                                num_workers=cfgs['data_cfg']['num_workers'])
    return loader


def run_train(model, msg_mgr, loader):
    """Accept the instance object(model) here, and then run the train loop."""
    for inputs in loader:
        ipts = model.inputs_pretreament(inputs)
        with autocast(enabled=model.engine_cfg['enable_float16']):
            retval = model(ipts)
            training_feat = retval['training_feat']
            visual_summary = retval['visual_summary']
            del retval
        loss_sum, loss_info = model.loss_aggregator(training_feat)
        ok = model.train_step(loss_sum)
        if not ok:
            continue

        visual_summary.update(loss_info)
        visual_summary['scalar/learning_rate'] = \
            model.optimizer.param_groups[0]['lr']

        msg_mgr.train_step(loss_info, visual_summary)
        if model.iteration % model.engine_cfg['save_iter'] == 0:
            # save the checkpoint
            model.save_ckpt(model.iteration)

            # run test if with_test = true
            if model.engine_cfg['with_test']:
                msg_mgr.log_info("Running test...")
                model.eval()
                result_dict = BaseModel.run_test(model)
                model.train()
                if model.cfgs['trainer_cfg']['fix_BN']:
                    model.fix_BN()
                msg_mgr.write_to_tensorboard(result_dict)
                msg_mgr.reset_time()
        if model.iteration >= model.engine_cfg['total_iter']:
            break


def run_test(model, msg_mgr, loader):
    """Accept the instance object(model) here, and then run the test loop."""

    rank = torch.distributed.get_rank()
    with torch.no_grad():
        info_dict = model.inference(rank, loader)
    if rank == 0:
        label_list = loader.dataset.label_list
        types_list = loader.dataset.types_list
        views_list = loader.dataset.views_list

        info_dict.update(
            {'labels': label_list, 'types': types_list, 'views': views_list})

        if 'eval_func' in model.cfgs["evaluator_cfg"].keys():
            eval_func = model.cfgs['evaluator_cfg']["eval_func"]
        else:
            eval_func = 'identification'
        eval_func = getattr(eval_functions, eval_func)
        valid_args = get_valid_args(
                eval_func, model.cfgs["evaluator_cfg"], ['metric'])
        try:
            dataset_name = model.cfgs['data_cfg']['test_dataset_name']
        except BaseException:
            dataset_name = model.cfgs['data_cfg']['dataset_name']

        eval_output, probe_x_fail_indices = eval_func(info_dict,
                                                      dataset_name,
                                                      **valid_args)

        if model.cfgs['evaluator_cfg']['failure_analysis']['save_failure_sequences']:
            save_failure_sequences(probe_x_fail_indices, loader)

        return eval_output


if __name__ == '__main__':
    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of availuable GPUs({}) equals to the "
                         "world size({})."
                         .format(torch.cuda.device_count(),
                                 torch.distributed.get_world_size()))
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    training = (opt.phase == 'train')
    msg_mgr = initialization(cfgs, training)

    loader = setup_loader(cfgs, msg_mgr, training)
    run_model(cfgs, loader, msg_mgr, training)
