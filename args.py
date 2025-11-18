import argparse

def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="8K_batch_size_128/train", help="project/name to save ckpts") 
    parser.add_argument("--exp_name", default="head_loss", help="save to project/name") 
    parser.add_argument("--data_path", type=str, default="data", help="raw data path") 
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument("--face_g", type=bool, default=False)
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    ) 
    parser.add_argument("--feature_type", type=str, default="wavlm", help="'baseline' or 'wavlm'") 
                    # baseline = hand-crafted features
                    # wavlm = hand-crafted + wavlm features
    parser.add_argument(
        "--wandb_pj_name", type=str, default="LMDM", help="wandb project name"
    ) 
    parser.add_argument("--batch_size", type=int, default=128, help="batch size") # 256
    parser.add_argument("--epochs", type=int, default=3400)
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    ) 
    parser.add_argument(
        "--parallelism", action="store_true", help="don't reuse / cache loaded dataset"
    ) 
    parser.add_argument("--parallelism1", default=128,
                        type=int, help="Level of parallelism")
    parser.add_argument("--rank", default=0, type=int,
                        help="Rank for distributed processing")
    
    parser.add_argument(
        "--save_interval",
        type=int,
        default=200,
        help='Log model after every "save_period" epoch',
    ) 
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    # 训练检查点设置
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    ) 
    ###################################################################################
    parser.add_argument(
        "--motion_diffusion_ckpt", type=str, default="", help="motion diffusion checkpoint"
    )
    # 推理视频保存路径
    parser.add_argument(
        "--save_path", type=str, default="./8K_batch_size_128/"
    )
    parser.add_argument('--use_motion_selection', default=True, action='store_true', help='use motion selection')
    parser.add_argument(
        "--test_face", type=bool, default=None, help="motion decoupling checkpoint"
    )    
    opt = parser.parse_args()
    return opt

 # export NCCL_TIMEOUT=600000
 # nohup accelerate launch train.py > 8K_batch_size_128/train.log 2>&1 &
 # tmux attach -t session_name
 # tmux ls
 # NCCL_TIMEOUT=60000
 # tmux new -s audio2pose
 # tmux attach -t audio2pose
 # accelerate config


