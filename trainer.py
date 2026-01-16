import os
import os.path as path

import time
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

MASTER_NODE = 0
logging.basicConfig(level=logging.INFO)

def get_logger(logger_name, file_name):
    """
    Create logger to log on both stdout and specific file
    """
    logger = logging.getLogger(logger_name)
    consoleHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(filename=file_name)

    logger_level = logging.DEBUG
    logger.setLevel(logger_level)
    consoleHandler.setLevel(logger_level)
    fileHandler.setLevel(logger_level)

    format_str = "%(asctime)s %(name)s:%(levelname)s %(message)s"
    time_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(format_str, time_format)
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger

class StepLRScheduleWrapper(object):
    def __init__(self,
            named_params,
            lr,
            optim_conf,
            logger,
            min_lr=1e-8,
            max_grad_norm=-1,
            step_num=5,
            decay=0.9,
                 ):
        self.lr_step = 0
        self.lr = lr
        self.min_lr = min_lr
        self.logger = logger
        self.max_grad_norm = max_grad_norm
        params = [p for n, p in named_params if p.requires_grad]
        optimizer_group_params = [{'params': params}]
        if "weight_decay" in optim_conf:
            optim_conf["weight_decay"] = float(optim_conf["weight_decay"])
        self.optimizer = optim.AdamW(optimizer_group_params, self.lr, **optim_conf)
        # self.optimizer = optim.RMSprop(optimizer_group_params, self.lr, **optim_conf)
        self.cur_lr = self.lr
        self.step_num = step_num
        self.decay = decay


    def get_learning_rate(self):
        return self.cur_lr

    def adjust_learning_rate(self, lr):
        for param in self.optimizer.param_groups:
            param['lr'] = lr

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        # clip grad
        if self.max_grad_norm > 0:
            for group in self.optimizer.param_groups:
                params = group['params']
                clip_grad_norm_(params, self.max_grad_norm)
        # optimizer step
        self.optimizer.step()

    def addStep_adjustLR(self, epoch):
        self.adjust_learning_rate(self.lr * (self.decay ** ((epoch+1) // self.step_num)))
        self.cur_lr = self.lr * (self.decay ** ((epoch + 1) // self.step_num))

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict['lr_step'] = self.lr_step
        state_dict['cur_lr'] = self.lr
        state_dict['wrapper_name'] = self.__class__.__name__
        return state_dict

    def load_state_dict(self, state_dict):
        self.lr_step = state_dict['lr_step']
        self.lr = state_dict['cur_lr']
        wrapper_name = state_dict['wrapper_name']
        assert wrapper_name == self.__class__.__name__
        state_dict.pop('lr_step')
        state_dict.pop('cur_lr')
        state_dict.pop('wrapper_name')
        self.optimizer.load_state_dict(state_dict)
        self.adjust_learning_rate(self.lr)

class MetricStat(object):
    """
    Metric statistics class
    Args:
        - tags: name tag for each metric
    """
    def __init__(self, tags):
        super(MetricStat, self).__init__()
        self.tags = tags
        self.total_count = [0 for _ in tags]
        self.total_sum = [0.0 for _ in tags]
        self.log_count = [0 for _ in tags]
        self.log_sum = [0.0 for _ in tags]

    def update_stat(self, metrics, counts):
        """update count and sum"""
        for i, (m, c) in enumerate(zip(metrics, counts)):
            self.log_count[i] += c
            self.log_sum[i] += m

    def log_stat(self):
        """get recent average statistics"""
        avg = []
        for i, (m, c) in enumerate(zip(self.log_sum, self.log_count)):
            avg_stat = 0.0 if c == 0 else m / c
            avg += [avg_stat]
            self.total_sum[i] += m
            self.log_sum[i] = 0.0
            self.total_count[i] += c
            self.log_count[i] = 0
        return avg

    def summary_stat(self):
        """get total average statistics"""
        avg = []
        for i in range(len(self.tags)):
            self.total_sum[i] += self.log_sum[i]
            self.total_count[i] += self.log_count[i]
            avg_stat = 0.0
            if self.total_count[i] != 0:
                avg_stat = self.total_sum[i] / self.total_count[i]
            avg += [avg_stat]
        return avg

    def reset(self):
        for i in range(len(self.tags)):
            self.total_sum[i] = 0.0
            self.total_count[i] = 0
            self.log_sum[i] = 0.0
            self.log_count[i] = 0


class Trainer:
    def __init__(self,
                 model,
                 output_dir,
                 init_model: str =None,
                 device="cpu",
                 **train_config,
                 ):
        # ddp related
        assert device in ["cpu", "gpu", "ddp"]
        self.device = device
        # config
        self.train_conf = train_config

        # result related
        self.output_dir = path.abspath(output_dir)
        log_path = path.join(self.output_dir, "log")
        os.makedirs(log_path, exist_ok=True)
        self.logger = get_logger("logger.{}".format(path.split(self.output_dir)[-1]),
                                 path.join(log_path, "train.log"))

        # build model
        self.model = model
        # initialize
        if init_model is None:
            self.logger.info("Random initialize model")
        else:
            param_dict = torch.load(init_model, map_location='cpu')
            self.model.load_state_dict(param_dict)
            self.logger.info("Initialize model from: {}".format(init_model))

        num_param = 0
        for param in self.model.parameters():
            num_param += param.numel()
        self.logger.info("model proto: {},\tmodel_size: {} M".format(
            self.model._get_name(), num_param / 1000 / 1000))

        # put in device
        if self.device != "cpu":
            self.model.cuda(0)

        # link final model
        self.final_model = path.join(self.output_dir, "model.final")

        # build tensorboard writer
        tensorboard_dir = path.join(self.output_dir, "tensorboard", "rank")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)

        # metric statistics
        self.train_metric = MetricStat(self.model.metric_tags)
        self.valid_metric = MetricStat(self.model.metric_tags)

        # build optimizer
        named_params = self.model.named_parameters()
        lr = self.train_conf.get('lr', 1e-4)
        optim_conf = self.train_conf.get('optim_conf', {})
        schedule_conf = self.train_conf.get('schedule_conf', {})
        self.optimizer = StepLRScheduleWrapper(named_params, lr, optim_conf, self.logger, **schedule_conf)
        self.logger.info("[Optimizer] scheduler wrapper is {}".format(
            self.optimizer.__class__.__name__))

        # build checkpoint
        chkpt_path = path.join(self.output_dir, "chkpt")
        self.chkpt_path = chkpt_path
        if path.isfile(chkpt_path):
            # load checkpoint
            chkpt = torch.load(chkpt_path, map_location="cpu")
            self.start_epoch = chkpt["epoch"]
            self.best_model = chkpt["best_model"]
            self.global_step = chkpt["global_step"]
            self.best_valid_loss = chkpt["best_valid_loss"]
            self.recent_models = chkpt["recent_models"]
            self.optimizer.load_state_dict(chkpt["optim"])
            cur_lr = self.optimizer.get_learning_rate()
            self.optimizer.adjust_learning_rate(cur_lr)
            # load model
            param_dict = torch.load(self.recent_models[-1], map_location="cpu")
            self.model.load_state_dict(param_dict)
            self.logger.info("Loading checkpoint from {}, Current lr: {}".format(chkpt_path, cur_lr))
            self.logger.info("Loading most recent model from {}".format(
                self.recent_models[-1]))
        else:
            # training from scratch
            self.logger.info("no checkpoint, start training from scratch")
            self.start_epoch = 1
            self.best_model = path.join(self.output_dir, "model.epoch-0.step-0")
            self.global_step = 0
            self.best_valid_loss = float('inf')
            self.recent_models = [self.best_model]

            model_state_dict = self.model.state_dict()
            torch.save(model_state_dict, self.best_model)
            self.save_chkpt(self.start_epoch)

        self.stop_step = 0  # for early stopping
        self.num_trained = 0

        self.train_dataset = None
        self.valid_dataset = None
    def save_chkpt(self, epoch):
        optim_state = self.optimizer.state_dict()
        chkpt = {'epoch': epoch,
                 'best_model': self.best_model,
                 'best_valid_loss': self.best_valid_loss,
                 'recent_models': self.recent_models,
                 'global_step': self.global_step,
                 'optim': optim_state}
        torch.save(chkpt, self.chkpt_path)
    def save_model_state(self, epoch):
        cur_model = path.join(self.output_dir, "model.epoch-{}.step-{}".format(epoch, self.global_step))
        model_state_dict = self.model.state_dict()
        torch.save(model_state_dict, cur_model)
        self.recent_models += [cur_model]
        num_recent_models = self.train_conf.get('num_recent_models', -1)
        if num_recent_models > 0 and len(self.recent_models) > num_recent_models:
            pop_model = self.recent_models.pop(0)
            os.remove(pop_model)
    def should_early_stop(self):
        early_stop_count = self.train_conf.get("early_stop_count", 0)
        return early_stop_count > 0 and self.stop_step >= early_stop_count
    def train_one_epoch(self, epoch):
        cur_lr = self.optimizer.get_learning_rate()
        self.logger.info("Epoch {} start, lr {}".format(epoch, cur_lr))

        # schedule_type = self.train_conf['schedule']
        log_period = self.train_conf.get('log_period', 10)
        accum_grad = self.train_conf.get('accum_grad', 1)

        # record start
        start_time = time.time()
        epoch_start_time = start_time

        # train mode
        self.model.train()
        hidden = None
        for batch_data in self.train_loader:
            if self.device == "gpu":
                data, target = [tuple([dd.cuda() for dd in d]) if type(d) is list else d.cuda() for d in batch_data]
            else:
                data, target = [(dd for dd in d) if type(d) is list else d for d in batch_data]
            batch_size = data[0].size(0) if type(data) is tuple else data.size(0)

            self.global_step += 1
            if self.global_step == 276:
                pass
            res = self.model(data, hidden=hidden)
            hidden = res.get("hidden", None)
            loss, metrics, counts = self.model.cal_loss(res, target, epoch=epoch)
            loss = loss / accum_grad
            loss.backward()
            self.train_metric.update_stat(metrics, counts)
            # update
            if self.global_step % accum_grad == 0:
                self.optimizer.addStep_adjustLR(epoch)
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.num_trained += batch_size

            # log info
            if self.global_step % log_period == 0:
                log_time = time.time()
                elapsed = log_time - start_time
                avg_stat = self.train_metric.log_stat()
                avg_str = []
                for tag, stat in zip(self.train_metric.tags, avg_stat):
                    self.tensorboard_writer.add_scalar("train/%s" % tag, stat, self.global_step)
                    avg_str += ["{}: {:.6f},".format(tag, stat)]
                avg_str = '\t'.join(avg_str)
                cur_lr = self.optimizer.get_learning_rate()
                self.tensorboard_writer.add_scalar("train/lr", cur_lr, self.global_step)
                self.logger.info("Epoch: {},\tTrained sentences: {},\t"
                                 "{}\tlr: {:.8f},\telapse: {:.1f} s".format(epoch,
                                                                            self.num_trained, avg_str, cur_lr,
                                                                            elapsed))
                start_time = log_time

        # summary statistics
        train_stat = self.train_metric.summary_stat()
        self.train_metric.reset()

        # final log
        avg_str = []
        for tag, stat in zip(self.train_metric.tags, train_stat):
            avg_str += ["{}: {:.6f},".format(tag, stat)]
        avg_str = '\t'.join(avg_str)
        elapsed = time.time() - epoch_start_time
        self.logger.info("Epoch {} Done,\t{}\tTime: {:.1f} second,\t".format(epoch, avg_str, elapsed))
        # valid
        if self.valid_loader is not None:
            self.valid(epoch)
        else:
            self.save_model_state(epoch)
            self.best_model = path.join(self.output_dir, "best_valid_model")
            os.system("cp {} {}".format(self.recent_models[-1], self.best_model))
        # save checkpoint
        self.save_chkpt(epoch + 1)
    def valid(self, epoch):
        self.logger.info("Start Validation")
        self.model.eval()

        # record start
        start_time = time.time()
        valid_start_time = start_time

        # start valid
        hidden = None
        for batch_idx, batch_data in enumerate(self.valid_loader):
            if self.device == "gpu":
                data, target = [tuple([dd.cuda() for dd in d]) if type(d) is list else d.cuda() for d in batch_data]
            else:
                data, target = [(dd for dd in d) if type(d) is list else d for d in batch_data]

            with torch.no_grad():
                res = self.model(data, hidden)
                hidden = getattr(res, "hidden", None)
                loss, metrics, counts = self.model.cal_loss(res, target, epoch=epoch)

                # update state
                self.valid_metric.update_stat(metrics, counts)

        # finish validation
        valid_stat = self.valid_metric.summary_stat()  # summarize total state in one epoch
        elapsed = time.time() - valid_start_time
        avg_str = []
        for tag, stat in zip(self.valid_metric.tags, valid_stat):
            avg_str += ["{}: {:.6f},".format(tag, stat)]
        avg_str = '\t'.join(avg_str)
        self.logger.info("Validation Done,\t{}\tTime: {} s\t".format(avg_str, elapsed))
        # sync validation results
        tot_sum = self.valid_metric.total_sum
        tot_num = self.valid_metric.total_count
        loss_tensor = torch.FloatTensor([tot_sum, tot_num])
        if self.device == "gpu":
            loss_tensor = loss_tensor.cuda(0)
        # record state
        self.valid_metric.reset()
        reduced_stat = loss_tensor[0] / loss_tensor[1]
        valid_stat = reduced_stat.cpu().numpy()
        self.logger.info("reduced valid loss: {}".format(valid_stat[0]))
        for tag, stat in zip(self.valid_metric.tags, valid_stat):
            self.tensorboard_writer.add_scalar("valid/%s" % tag, stat, self.global_step)
        # save model state
        self.save_model_state(epoch)
        # check best loss
        # NOTE: state[0] must store total loss
        valid_loss = valid_stat[0]
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.best_model = path.join(self.output_dir, "best_valid_model")
            os.system("cp {} {}".format(self.recent_models[-1], self.best_model))
            self.logger.info("new best_valid_loss: {}, storing best model: {}".format(
                self.best_valid_loss, self.recent_models[-1]))
            self.stop_step = 0
        else:
            self.stop_step += 1
        # back to train mode
        self.model.train()

    def fit(self, train_dataset, val_dataset, test_dataset=None, test_batch_size=None):
        # build loader
        self.train_dataset = train_dataset
        self.valid_dataset = val_dataset
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_conf["batch_size"],
            drop_last=self.train_conf.get("drop_last", False),
            num_workers=0
        )
        if self.valid_dataset is not None:
            self.valid_loader = DataLoader(
                self.valid_dataset,
                batch_size=self.train_conf["batch_size"],
                drop_last=self.train_conf.get("drop_last", False),
                num_workers=0
            )
        else:
            self.valid_loader = None
        if test_dataset is not None:
            self.test_dataset = test_dataset
            if test_batch_size is None:
                self.test_batch_size = self.train_conf["batch_size"]
            else:
                self.test_batch_size = test_batch_size

        max_epochs = self.train_conf["max_epochs"]

        # training
        if not path.exists(self.final_model):
            for epoch in range(self.start_epoch, max_epochs + 1):
                if self.should_early_stop():
                    self.logger.info("Early stopping")
                    break
                self.train_dataset.set_epoch(epoch)
                self.train_one_epoch(epoch)
            self.logger.info("Training finished")
            for handler in list(self.logger.handlers):
                self.logger.removeHandler(handler)
            os.system("ln -s {} {}".format(
                os.path.abspath(self.best_model), self.final_model))
        else:
            self.logger.info(f"Exist {self.final_model}")