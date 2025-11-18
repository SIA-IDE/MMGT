import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.gesture_dataset import GestureDataset
from src.dataset.preprocess import increment_path
from src.audio2pose_model.adan import Adan
from src.audio2pose_model.diffusion import GestureDiffusion
from src.audio2pose_model.model import GestureDecoder
import torch.distributed
import numpy as np


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)

def transform_if_no_negative(x: torch.Tensor) -> torch.Tensor:
    """
    如果 x 中没有负数，则做 x = x * 2 - 1；否则不做变换。
    :param x: PyTorch Tensor
    :return: 变换后的 PyTorch Tensor
    """
    # 直接用 PyTorch 的 .any() 判断是否存在负数
    if (x < 0).any():
        # 存在负数，不做变换
        return x
    else:
        # 不存在负数，则进行 x = x*2 - 1
        return x * 2 - 1

class SMGA:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        EMA=True,
        learning_rate=2e-4,
        weight_decay=0.02,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # torch.distributed.init_process_group(backend='gloo', init_method='env://') # for RTX 4090
        
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes

        self.repr_dim = repr_dim = 402

        feature_dim = 1024 + 35 if feature_type == "wavlm" else 35

        horizon_seconds = 3.2
        FPS = 25
        self.horizon = horizon = int(horizon_seconds * FPS)

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            print('load ckpt weight'+checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.normalizer = checkpoint["normalizer"]
            print(self.normalizer)

        model = GestureDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )

        diffusion = GestureDiffusion(
            model,
            horizon,
            repr_dim,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            print('load ckpt')
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )

    def vae_encoder(self, x):
        x = self.model.vae_encoder(x)
        return x



    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        # load datasets
        print('load dataset')
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):  # already backuped pkl dataset
            print("load train dataset from pkl")
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            print("load test dataset from pkl")
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
            print('load dataset success')
        else: # no backuped pkl dataset
            print("load raw dataset")
            train_dataset = GestureDataset(
                feature_type = opt.feature_type,
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
            )
            test_dataset = GestureDataset(
                feature_type = opt.feature_type,
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
            )
            # save dataset to pkl
            # if self.accelerator.is_main_process:
            #     print("save 1/3 of dataset to pkl")
                # one_third_train = train_dataset[:len(train_dataset)//3]
                # one_third_test = test_dataset[:len(test_dataset)//3]
                # print("save dataset to pkl")
                # pickle.dump(one_third_train, open(train_tensor_dataset_path, "wb"), protocol=4)
                # pickle.dump(one_third_test, open(test_tensor_dataset_path, "wb"), protocol=4)
            if self.accelerator.is_main_process: 
                            print("save dataset to pkl")
                            pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"), protocol=4)
                            pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"), protocol=4)

        # set normalizer
        self.normalizer = test_dataset.normalizer
        print(self.normalizer)

        # data loaders
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        # test_data_loader = DataLoader(
        #     test_dataset,
        #     batch_size=opt.batch_size,
        #     shuffle=True,
        #     num_workers=2,
        #     pin_memory=True,
        #     drop_last=False,
        # )


        train_data_loader = self.accelerator.prepare(train_data_loader)

        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            print("init wandb")
            # wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name, mode="offline")
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()

        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_ploss = 0
            avg_vloss = 0
            avg_aloss = 0
            avg_hloss = 0  # 头部 loss
            avg_vhloss = 0  # 头部 velocity loss
            avg_ahloss = 0  # 头部 acceleration loss

            # train
            self.train()
            for step, (x, cond_frame, cond, wavnames) in enumerate(
                load_loop(train_data_loader)
            ):
                
                x = transform_if_no_negative(x)
                cond_frame = transform_if_no_negative(cond_frame)
                loss, (pos_loss, v_loss, a_loss, head_loss, v_head_loss, a_head_loss) = self.diffusion(
                    x, cond_frame, cond, t_override=None
                )
                self.optim.zero_grad()
                self.accelerator.backward(loss)
                self.optim.step()
                # ema update and train loss update only on main
                loss_file_path = "loss.txt"
                with open(loss_file_path, "a") as loss_file:
                    if self.accelerator.is_main_process:
                        avg_loss += loss.detach().cpu().numpy()
                        avg_ploss += pos_loss.detach().cpu().numpy()
                        avg_vloss += v_loss.detach().cpu().numpy()
                        avg_aloss += a_loss.detach().cpu().numpy()
                        
                        # 新增的头部loss处理
                        avg_hloss += head_loss.detach().cpu().numpy()
                        avg_vhloss += v_head_loss.detach().cpu().numpy()
                        avg_ahloss += a_head_loss.detach().cpu().numpy()

                        # 写入loss到文件
                        loss_file.write(f"Avg Loss: {avg_loss}, Pos Loss: {avg_ploss}, V Loss: {avg_vloss}, A Loss: {avg_aloss}, "
                                        f"Head Loss: {avg_hloss}, V Head Loss: {avg_vhloss}, A Head Loss: {avg_ahloss}\n")

                        if step % opt.ema_interval == 0:
                            self.diffusion.ema.update_model_average(
                                self.diffusion.master_model, self.diffusion.model
                            )

            # Save model
            if (epoch % opt.save_interval) == 0:
                self.accelerator.wait_for_everyone()
                
                if self.accelerator.is_main_process:
                    self.eval()
                    
                    # 计算并记录平均loss
                    avg_loss /= len(train_data_loader)
                    avg_ploss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_aloss /= len(train_data_loader)
                    
                    # 新增的头部loss的平均值记录
                    avg_hloss /= len(train_data_loader)
                    avg_vhloss /= len(train_data_loader)
                    avg_ahloss /= len(train_data_loader)

                    # 日志记录
                    log_dict = {
                        "Train Loss": avg_loss,
                        "Pos Loss": avg_ploss,
                        "V Loss": avg_vloss,
                        "A Loss": avg_aloss,
                        "Head Loss": avg_hloss,
                        "V Head Loss": avg_vhloss,
                        "A Head Loss": avg_ahloss
                    }
                    wandb.log(log_dict)

                    # 保存模型检查点
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    print(f"[MODEL SAVED at Epoch {epoch}]")

        if self.accelerator.is_main_process:
            wandb.run.finish()


    def render_sample(
            self, cond_frame, cond, last_half, mode
    ):
        
        render_count = cond_frame.shape[0]
        shape = (render_count, self.horizon, self.repr_dim)
        cond_frame_input = cond_frame.to(self.accelerator.device)
        cond_input = cond.to(self.accelerator.device)
        last_half = last_half.unsqueeze(0).to(self.accelerator.device) if last_half is not None else None
        result = self.diffusion.render_sample(
            shape,
            cond_frame_input,
            cond_input,
            # self.normalizer,
            epoch=None,
            render_out=None,
            last_half=last_half,
            mode=mode,
        ).to(self.accelerator.device)
        # result = torch.cat([cond_frame_input.unsqueeze(1), result], dim=1)

        return result

