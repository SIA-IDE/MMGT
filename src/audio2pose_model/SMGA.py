import os
import pickle
from functools import partial
from pathlib import Path
import io, struct, numpy as np, torch

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections.abc import ByteString
from src.dataset.gesture_dataset import GestureDataset
from src.dataset.preprocess import increment_path
from src.audio2pose_model.adan import Adan
from src.audio2pose_model.diffusion import GestureDiffusion
from src.audio2pose_model.model import GestureDecoder
import torch.distributed
import numpy as np

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
    


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)



class LMDM:
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
            # self.normalizer = checkpoint["normalizer"]
            # print(self.normalizer)

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
        train_lmdb_path = os.path.join(opt.processed_data_dir, "train.lmdb")
        test_lmdb_path  = os.path.join(opt.processed_data_dir, "test.lmdb")
        normalizer_path = os.path.join(opt.processed_data_dir, "normalizer.pt")

        
        if (
            not opt.no_cache
            and os.path.isdir(train_lmdb_path)
            and os.path.isdir(test_lmdb_path)
            and os.path.isfile(normalizer_path)
        ):  # already backuped pkl dataset
            print("load dataset from LMDB")
            train_dataset = LMDBGestureDataset(train_lmdb_path, split="train")
            test_dataset  = LMDBGestureDataset(test_lmdb_path,  split="test")
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
            if self.accelerator.is_main_process:
                print("dump dataset to LMDB")
                dump_dataset_to_lmdb(train_dataset, train_lmdb_path, is_main = True, split="train")
                dump_dataset_to_lmdb(test_dataset,  test_lmdb_path, is_main = True, split="test")
                torch.save(train_dataset.normalizer, os.path.join(opt.processed_data_dir, "normalizer.pt"))


            # self.normalizer = torch.load(normalizer_path)
            train_dataset = LMDBGestureDataset(train_lmdb_path, split="train")
            test_dataset  = LMDBGestureDataset(test_lmdb_path,  split="test")
        
        # set normalizer
        # self.normalizer = self.normalizer

        # data loaders
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        # 分布式训练准备
        train_data_loader = self.accelerator.prepare(train_data_loader)

        # 日志初始化
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            print("init wandb")
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()

        for epoch in range(1, opt.epochs + 1):
            metric_logger = {
                'avg_loss': 0,
                'avg_ploss': 0,
                'avg_vloss': 0,
                'avg_aloss': 0,
                'avg_hloss': 0,
                'avg_vhloss': 0,
                'avg_ahloss': 0
            }
            
            # 训练阶段
            self.train()
            progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch}") if self.accelerator.is_main_process else train_data_loader

            for batch_idx, (x, cond_frame, cond, wavnames) in enumerate(progress_bar):
                # 数据预处理
                x = transform_if_no_negative(x)
                cond_frame = transform_if_no_negative(cond_frame)
                
                # 前向计算
                with self.accelerator.autocast():
                    loss, (pos_loss, v_loss, a_loss, head_loss, v_head_loss, a_head_loss) = self.diffusion(
                        x, cond_frame, cond, t_override=None
                    )
                
                # 反向传播
                self.optim.zero_grad()
                self.accelerator.backward(loss)
                # if opt.clip_grad:
                #     self.accelerator.clip_grad_norm_(self.model.parameters(), opt.max_grad_norm)
                self.optim.step()
                
                # 指标更新
                if self.accelerator.is_main_process:
                    metric_logger['avg_loss'] += loss.detach().mean().item()
                    metric_logger['avg_ploss'] += pos_loss.detach().mean().item()
                    metric_logger['avg_vloss'] += v_loss.detach().mean().item()
                    metric_logger['avg_aloss'] += a_loss.detach().mean().item()
                    metric_logger['avg_hloss'] += head_loss.detach().mean().item()
                    metric_logger['avg_vhloss'] += v_head_loss.detach().mean().item()
                    metric_logger['avg_ahloss'] += a_head_loss.detach().mean().item()
                    
                    # EMA更新
                    if batch_idx % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, 
                            self.diffusion.model
                        )

            # 模型保存
            if (epoch % opt.save_interval) == 0 and self.accelerator.is_main_process:
                self._save_checkpoint(epoch, wdir, metric_logger, len(train_data_loader))

        # 清理资源
        if self.accelerator.is_main_process:
            wandb.run.finish()

    def _save_checkpoint(self, epoch, save_dir, metrics, num_batches):
        """保存模型检查点"""
        # 计算平均指标
        avg_metrics = {k: v / num_batches for k, v in metrics.items()}
        
        # 日志记录
        wandb.log({
            "Train Loss": avg_metrics['avg_loss'],
            "Pos Loss": avg_metrics['avg_ploss'],
            "V Loss": avg_metrics['avg_vloss'],
            "A Loss": avg_metrics['avg_aloss'],
            "Head Loss": avg_metrics['avg_hloss'],
            "V Head Loss": avg_metrics['avg_vhloss'],
            "A Head Loss": avg_metrics['avg_ahloss']
        })
        
        # 保存模型
        ckpt = {
            "ema_state_dict": self.diffusion.master_model.state_dict(),
            "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            # "normalizer": self.normalizer,
            "epoch": epoch
        }
        torch.save(ckpt, save_dir / f"epoch_{epoch}.pth")
        print(f"[Checkpoint saved at epoch {epoch}]")


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

def dump_dataset_to_lmdb(ds, lmdb_path, split, is_main, map_size=512 * 1024 ** 3):
    if not is_main:
        return

    env = lmdb.open(lmdb_path, map_size=map_size)
    with env.begin(write=True) as txn:
        for idx in tqdm(range(len(ds)), desc=f"Dump {split}", disable=not is_main):
            kpt_in, kpt_cond, wav_feat, wav_raw = ds[idx]

            txn.put(f"{split}_kptin_{idx:08d}".encode(),
                    _tensor_to_bytes(kpt_in))
            txn.put(f"{split}_kptcond_{idx:08d}".encode(),
                    _tensor_to_bytes(kpt_cond))
            txn.put(f"{split}_wavfeat_{idx:08d}".encode(),
                    _tensor_to_bytes(wav_feat))
            # wav_raw 可能是 bytes 或 ndarray；若已是 bytes 直接存
            if isinstance(wav_raw, torch.Tensor):
                payload = _tensor_to_bytes(wav_raw)
            elif isinstance(wav_raw, ByteString):          # bytes / bytearray
                payload = wav_raw
            elif isinstance(wav_raw, str):
                payload = wav_raw.encode("utf-8")           # 字符串 ⇒ bytes
            else:  # ndarray、list 等
                payload = _tensor_to_bytes(torch.as_tensor(wav_raw))

            txn.put(f"{split}_wavraw_{idx:08d}".encode(), payload)

        txn.put(f"{split}_length".encode(), str(len(ds)).encode())
    env.sync(); env.close()



class LMDBGestureDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path: str, split: str, dtype=np.float32):
        self.env  = lmdb.open(lmdb_path, readonly=True,
                              lock=False, readahead=False)
        with self.env.begin() as txn:
            raw_len = txn.get(f"{split}_length".encode())
            if raw_len is None:
                raise FileNotFoundError(f"{split}_length key missing in {lmdb_path}")
            self.length = int(raw_len.decode())
        self.split = split
        self.dtype = dtype

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            kptin_buf   = txn.get(f"{self.split}_kptin_{idx:08d}".encode())
            kptcond_buf = txn.get(f"{self.split}_kptcond_{idx:08d}".encode())
            wavfeat_buf = txn.get(f"{self.split}_wavfeat_{idx:08d}".encode())
            wavraw_buf  = txn.get(f"{self.split}_wavraw_{idx:08d}".encode())

        keypoint_input  = _bytes_to_tensor(kptin_buf,   self.dtype)
        keypoint_cond   = _bytes_to_tensor(kptcond_buf, self.dtype)
        wav_feature     = _bytes_to_tensor(wavfeat_buf, self.dtype)
        
        try:                               # 先尝试按字符串 decode
            wav_raw = wavraw_buf.decode("utf-8")
        except UnicodeDecodeError:
            wav_raw = wavraw_buf           # 非 UTF‑8，则为 bytes

        return keypoint_input, keypoint_cond, wav_feature, wav_raw


# ---------- 编码 ----------
def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    arr   = t.detach().cpu().numpy()
    ndim  = arr.ndim
    shape = arr.shape                     # tuple
    # 1) 先写 ndim (uint32)
    header = struct.pack("<I", ndim)
    # 2) 紧接着写 shape 里每个 dim (ndim * uint32)
    header += struct.pack(f"<{ndim}I", *shape)
    # 3) 你若需要保存 dtype，也可以再加一行:
    # header += struct.pack("<H", arr.dtype.num)  # 可选
    return header + arr.tobytes()

# ---------- 解码 ----------
def _bytes_to_tensor(buf: bytes, dtype=np.float32) -> torch.Tensor:
    # 1) 读 ndim
    ndim = struct.unpack_from("<I", buf, 0)[0]
    # 2) 读 shape
    shape_fmt = f"<{ndim}I"
    shape     = struct.unpack_from(shape_fmt, buf, 4)
    data_off  = 4 + 4 * ndim
    # 3) 直接用 numpy 的 zero-copy
    arr = np.frombuffer(buf[data_off:], dtype=dtype).reshape(shape)
    return torch.from_numpy(arr)