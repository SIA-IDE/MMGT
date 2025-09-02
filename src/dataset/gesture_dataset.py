import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch.utils.data import Dataset
from tqdm import tqdm


class GestureDataset_lgci(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "wavlm",
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
    ):
        self.data_path = data_path

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer = normalizer
        self.data_len = data_len

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)

        print("Loading dataset...")
        data = self.load_data() 

        print(
            f"Loaded {self.name} Dataset With Dimensions: Keypoints: {data['keypoints'].shape}, Wav_features: {data['wav_features'].shape}, Wavs: {len(data['wavs'])}"
        )

        self.data = {
            "keypoints": data['keypoints'],
            "wav_features": data["wav_features"],
            "wavs": data["wavs"],
        }
        assert len(data['keypoints']) == len(data["wav_features"])
        self.length = len(data['keypoints'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        wav_feature = torch.from_numpy(self.data["wav_features"][idx])
        keypoint = self.data['keypoints'][idx]
        # seq_len = keypoint.shape[0]        
        # keypoints = keypoint.reshape(seq_len, 134, 3)
        # mask = np.ones(keypoints.shape, dtype=np.uint8) 
        # lip_indices = range(72, 92)
        # chin_indices = range(28, 37)       # 下巴
        # left_eye_indices = range(60, 66)  # 左眼
        # right_eye_indices = range(66, 72) # 右眼  

        # # 将嘴唇区域的掩码设置为 0
        # mask[:, chin_indices, :] = 0
        # mask[:, left_eye_indices, :] = 0
        # mask[:, right_eye_indices, :] = 0
        # mask[:, lip_indices, :] = 0
        # keypoint = keypoints * mask
        # keypoint = keypoint.reshape(seq_len, -1)
        keypoint_cond = torch.from_numpy(keypoint[0, :].astype(np.float32))
        # TODO mask脸部
        keypoint_input = torch.from_numpy(keypoint[0:, :].astype(np.float32))
        return keypoint_input, keypoint_cond, wav_feature, self.data["wavs"][idx], self.data['keypoint_filenames'][idx]

    def load_data(self):
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )

        keypoint_path = os.path.join(split_data_path, "keypoints")
        feature_path = os.path.join(split_data_path, f"{self.feature_type}_feats")
        baseline_path = os.path.join(split_data_path, f"baseline_feats")
        wav_path = os.path.join(split_data_path, f"wavs")
        
        # sort keypoints and sounds
        keypoints = sorted(glob.glob(os.path.join(keypoint_path, "*.npy")))
        features = sorted(glob.glob(os.path.join(feature_path, "*.npy")))
        baseline_features = sorted(glob.glob(os.path.join(baseline_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        # stack the keypoints and features together
        all_keypoints = []
        all_features = []
        all_wavs = []
        keypoint_filenames = []  # 新增列表以存储 keypoints 的文件名
        
        assert len(keypoints) == len(features) == len(baseline_features) == len(wavs)
        for keypoint, feature, baseline_feature, wav in tqdm(zip(keypoints, features, baseline_features, wavs)):
            # make sure name is matching
            k_name = os.path.splitext(os.path.basename(keypoint))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            assert k_name == f_name == w_name, str((keypoint, feature, wav))
            
            # load keypoints
            data = np.load(keypoint)
            all_keypoints.append(data)
            
            # 添加文件名
            keypoint_filenames.append(k_name)  # 存储文件名
            
            if self.feature_type != 'baseline':
                input_feature = np.concatenate((np.load(feature), np.load(baseline_feature)), axis=-1)
            else:
                input_feature = np.load(baseline_feature)
            
            all_features.append(input_feature)
            all_wavs.append(wav)

        all_keypoints = np.array(all_keypoints)
        all_features = np.array(all_features)

        print(all_keypoints.shape)
        print(all_features.shape)
        
        # 在返回的数据中增加文件名
        data = {
            "keypoints": all_keypoints,
            "wav_features": all_features,
            "wavs": all_wavs
        }
        return data



class GestureDataset_eval(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "wavlm",
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
    ):
        self.data_path = data_path

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer = normalizer
        self.data_len = data_len

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)

        print("Loading dataset...")
        data = self.load_data() 

        print(
            f"Loaded {self.name} Dataset With Dimensions: Keypoints: {data['keypoints'].shape}, Wav_features: {data['wav_features'].shape}, Wavs: {len(data['wavs'])}"
        )

        self.data = {
            "keypoints": data['keypoints'],
            "wav_features": data["wav_features"],
            "wavs": data["wavs"],
            "keypoint_filenames": data["keypoint_filenames"]
        }
        assert len(data['keypoints']) == len(data["wav_features"])
        self.length = len(data['keypoints'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        wav_feature = torch.from_numpy(self.data["wav_features"][idx])
        keypoint = self.data['keypoints'][idx]
        # seq_len = keypoint.shape[0]        
        # keypoints = keypoint.reshape(seq_len, 134, 3)
        # mask = np.ones(keypoints.shape, dtype=np.uint8) 
        # lip_indices = range(72, 92)
        # chin_indices = range(28, 37)       # 下巴
        # left_eye_indices = range(60, 66)  # 左眼
        # right_eye_indices = range(66, 72) # 右眼  

        # # 将嘴唇区域的掩码设置为 0
        # mask[:, chin_indices, :] = 0
        # mask[:, left_eye_indices, :] = 0
        # mask[:, right_eye_indices, :] = 0
        # mask[:, lip_indices, :] = 0
        # keypoint = keypoints * mask
        # keypoint = keypoint.reshape(seq_len, -1)
        keypoint_cond = torch.from_numpy(keypoint[0, :].astype(np.float32))
        # TODO mask脸部
        keypoint_input = torch.from_numpy(keypoint[0:, :].astype(np.float32))
        keypoint_filename = self.data['keypoint_filenames'][idx]


        return keypoint_input, keypoint_cond, wav_feature, self.data["wavs"][idx], keypoint_filename

    def load_data(self):
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )

        keypoint_path = os.path.join(split_data_path, "keypoints_sliced")
        feature_path = os.path.join(split_data_path, f"{self.feature_type}_feats_sliced")
        baseline_path = os.path.join(split_data_path, f"baseline_feats_sliced")
        wav_path = os.path.join(split_data_path, f"wavs_sliced")
        
        # sort keypoints and sounds
        keypoints = sorted(glob.glob(os.path.join(keypoint_path, "*.npy")))
        features = sorted(glob.glob(os.path.join(feature_path, "*.npy")))
        baseline_features = sorted(glob.glob(os.path.join(baseline_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        # stack the keypoints and features together
        all_keypoints = []
        all_features = []
        all_wavs = []
        keypoint_filenames = []  # 新增列表以存储 keypoints 的文件名

        assert len(keypoints) == len(features) == len(baseline_features) == len(wavs)
        for keypoint, feature, baseline_feature, wav in tqdm(zip(keypoints, features, baseline_features, wavs)):
            # make sure name is matching
            k_name = os.path.splitext(os.path.basename(keypoint))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            assert k_name == f_name == w_name, str((keypoint, feature, wav))
            
            # load keypoints
            data = np.load(keypoint)
            all_keypoints.append(data)
            
            # 添加文件名
            keypoint_filenames.append(k_name)  # 存储文件名
            
            if self.feature_type != 'baseline':
                input_feature = np.concatenate((np.load(feature), np.load(baseline_feature)), axis=-1)
            else:
                input_feature = np.load(baseline_feature)
            
            all_features.append(input_feature)
            all_wavs.append(wav)

        all_keypoints = np.array(all_keypoints)
        all_features = np.array(all_features)
        keypoint_filenames = np.array(keypoint_filenames)

        print(all_keypoints.shape)
        print(all_features.shape)
        
        # 在返回的数据中增加文件名
        data = {
            "keypoints": all_keypoints,
            "wav_features": all_features,
            "wavs": all_wavs,
            "keypoint_filenames": keypoint_filenames  # 添加文件名
        }
        return data
    




class GestureDataset_eval_baseline(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "wavlm",
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
    ):
        self.data_path = data_path

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer = normalizer
        self.data_len = data_len

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)

        print("Loading dataset...")
        data = self.load_data() 

        print(
            f"Loaded {self.name} Dataset With Dimensions: Keypoints: {data['keypoints'].shape}, "
            f"Wav_features: {data['wav_features'].shape}, Wavs: {len(data['wavs'])}, "
            f"Base_line_keypoints: {len(data['baseline_keypoints'])}"
        )

        self.data = {
            "keypoints": data['keypoints'],
            "wav_features": data["wav_features"],
            "wavs": data["wavs"],
            "keypoint_filenames": data["keypoint_filenames"],
            "baseline_keypoints": data['baseline_keypoints'],
            "all_gnerate_keypoints": data['all_gnerate_keypoints']
        }
        assert len(data['keypoints']) == len(data["wav_features"]) == len(data['baseline_keypoints']) == len(data['all_gnerate_keypoints'])
        self.length = len(data['keypoints'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        wav_feature = torch.from_numpy(self.data["wav_features"][idx])
        keypoint = self.data['keypoints'][idx]
        baseline_keypoint = self.data['baseline_keypoints'][idx]
        all_gnerate_keypoint = self.data['all_gnerate_keypoints'][idx]
        # seq_len = keypoint.shape[0]        
        # keypoints = keypoint.reshape(seq_len, 134, 3)
        # mask = np.ones(keypoints.shape, dtype=np.uint8) 
        # lip_indices = range(72, 92)
        # chin_indices = range(28, 37)       # 下巴
        # left_eye_indices = range(60, 66)  # 左眼
        # right_eye_indices = range(66, 72) # 右眼  

        # # 将嘴唇区域的掩码设置为 0
        # mask[:, chin_indices, :] = 0
        # mask[:, left_eye_indices, :] = 0
        # mask[:, right_eye_indices, :] = 0
        # mask[:, lip_indices, :] = 0
        # keypoint = keypoints * mask
        # keypoint = keypoint.reshape(seq_len, -1)
        keypoint_cond = torch.from_numpy(keypoint[0, :].astype(np.float32))
        baseline_keypoint_input = torch.from_numpy(baseline_keypoint[0:, :].astype(np.float32))
        all_gnerate_keypoint_input = torch.from_numpy(all_gnerate_keypoint[0:, :].astype(np.float32))
        # TODO mask脸部
        keypoint_input = torch.from_numpy(keypoint[0:, :].astype(np.float32))
        keypoint_filename = self.data['keypoint_filenames'][idx]
        return keypoint_input, keypoint_cond, wav_feature, self.data["wavs"][idx], keypoint_filename, baseline_keypoint_input, all_gnerate_keypoint_input

    def load_data(self):
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )

        keypoint_path = os.path.join(split_data_path, "keypoints_sliced")
        feature_path = os.path.join(split_data_path, f"{self.feature_type}_feats_sliced")
        baseline_path = os.path.join(split_data_path, f"baseline_feats_sliced")
        wav_path = os.path.join(split_data_path, f"wavs_sliced")

        baseline_keypoints_path = os.path.join(split_data_path,"ablation_FILM")# baseline
        # gnerate_keypoints_path = os.path.join(split_data_path, f"ours_5000")
        gnerate_keypoints_path = os.path.join(split_data_path, "test_1600")
        
        # sort keypoints and sounds
        keypoints = sorted(glob.glob(os.path.join(keypoint_path, "*.npy")))
        features = sorted(glob.glob(os.path.join(feature_path, "*.npy")))
        baseline_features = sorted(glob.glob(os.path.join(baseline_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))
        baseline_keypoints = sorted(glob.glob(os.path.join(baseline_keypoints_path, "*.npy")))
        gnerate_keypoints = sorted(glob.glob(os.path.join(gnerate_keypoints_path, "*.npy")))

        # stack the keypoints and features together
        all_keypoints = []
        all_features = []
        all_wavs = []
        keypoint_filenames = []  # 新增列表以存储 keypoints 的文件名
        all_baseline_keypoints = []
        all_gnerate_keypoints = []
        
        assert len(keypoints) == len(features) == len(baseline_features) == len(wavs) == len(baseline_keypoints) == len(gnerate_keypoints)

        for keypoint, feature, baseline_feature, wav, baseline_keypoint, gnerate_keypoint in tqdm(zip(keypoints, features, baseline_features, wavs, baseline_keypoints, gnerate_keypoints)):
            # make sure name is matching
            k_name = os.path.splitext(os.path.basename(keypoint))[0]
            b_name = os.path.splitext(os.path.basename(baseline_keypoint))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            assert k_name == f_name == w_name, str((keypoint, feature, wav))
            
            # load keypoints
            data = np.load(keypoint)
            all_keypoints.append(data)
            # load baseline keypoints
            data_baseline = np.load(baseline_keypoint)
            all_baseline_keypoints.append(data_baseline)
            # load generate keypoints
            data_generate = np.load(gnerate_keypoint)
            all_gnerate_keypoints.append(data_generate)

            # 添加文件名
            keypoint_filenames.append(k_name)  # 存储文件名
            
            if self.feature_type != 'baseline':
                input_feature = np.concatenate((np.load(feature), np.load(baseline_feature)), axis=-1)
            else:
                input_feature = np.load(baseline_feature)
            
            all_features.append(input_feature)
            all_wavs.append(wav)

        all_keypoints = np.array(all_keypoints)
        all_baseline_keypoints = np.array(all_baseline_keypoints)
        all_gnerate_keypoints = np.array(all_gnerate_keypoints)
        all_features = np.array(all_features)
        keypoint_filenames = np.array(keypoint_filenames)

        print(all_keypoints.shape)
        print(all_features.shape)
        print(all_baseline_keypoints.shape)
        print(all_gnerate_keypoints.shape)
        
        # 在返回的数据中增加文件名
        data = {
            "keypoints": all_keypoints,
            "wav_features": all_features,
            "wavs": all_wavs,
            "keypoint_filenames": keypoint_filenames,  # 添加文件名
            "baseline_keypoints": all_baseline_keypoints,
            "all_gnerate_keypoints": all_gnerate_keypoints
        }
        return data
    


class GestureDataset_train(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        parallelism: int,
        rank: int,
        feature_type: str = "wavlm",
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
    ):
        self.data_path = data_path
        self.parallelism = parallelism
        self.rank = rank
        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer = normalizer
        self.data_len = data_len

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)

        print("Loading dataset...")
        data = self.load_data() 

        print(
            f"Loaded {self.name} Dataset With Dimensions: Keypoints: {data['keypoints'].shape}, "
            f"Wav_features: {data['wav_features'].shape}, "
        )

        self.data = {
            "keypoints": data['keypoints'],
            "wav_features": data["wav_features"],
            "keypoint_filenames": data["keypoint_filenames"]
        }
        assert len(data['keypoints']) == len(data["wav_features"])
        self.length = len(data['keypoints'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        wav_feature = torch.from_numpy(self.data["wav_features"][idx])
        keypoint = self.data['keypoints'][idx]
        keypoint_cond = torch.from_numpy(keypoint[0, :].astype(np.float32))
        # TODO mask脸部
        keypoint_input = torch.from_numpy(keypoint[0:, :].astype(np.float32))
        keypoint_filename = self.data['keypoint_filenames'][idx]
        return keypoint_input, keypoint_cond, wav_feature, keypoint_filename


    def load_data(self):
        split_data_path = os.path.join(
            self.data_path, "test"
        )

        keypoint_path = os.path.join(split_data_path, "keypoints_sliced")
        feature_path = os.path.join(split_data_path, f"{self.feature_type}_feats_sliced")
        baseline_path = os.path.join(split_data_path, f"baseline_feats_sliced")
        wav_path = os.path.join(split_data_path, f"wavs_sliced")

        keypoints = get_npy_paths(keypoint_path, self.parallelism, self.rank)
        features = get_npy_paths(feature_path, self.parallelism, self.rank)
        baseline_features = get_npy_paths(baseline_path, self.parallelism, self.rank)
        wavs = get_audio_paths(wav_path, self.parallelism, self.rank)

        # keypoints = sorted(glob.glob(os.path.join(keypoint_path, "*.npy")))
        # features = sorted(glob.glob(os.path.join(feature_path, "*.npy")))
        # baseline_features = sorted(glob.glob(os.path.join(baseline_path, "*.npy")))
        # wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))
        # stack the keypoints and features together
        all_keypoints = []
        all_features = []
        all_wavs = []
        keypoint_filenames = []  # 新增列表以存储 keypoints 的文件名
        
        assert len(keypoints) == len(features) == len(baseline_features) == len(wavs)

        for keypoint, feature, baseline_feature, wav in tqdm(zip(keypoints, features, baseline_features, wavs)):
            # make sure name is matching
            k_name = os.path.splitext(os.path.basename(keypoint))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            assert k_name == f_name == w_name, str((keypoint, feature, wav))
            
            # load keypoints
            data = np.load(keypoint)
            all_keypoints.append(data[0:80, :])
            # 添加文件名
            keypoint_filenames.append(k_name)  # 存储文件名
            if self.feature_type != 'baseline':
                input_feature = np.concatenate((np.load(feature), np.load(baseline_feature)), axis=-1)
            else:
                input_feature = np.load(baseline_feature)
            
            all_features.append(input_feature)
            all_wavs.append(wav)

        # expected_shape = (81, 402)
        # mismatch_indices = []
        # for i, arr in enumerate(all_keypoints):
        #     if arr.shape != expected_shape:
        #         mismatch_indices.append(i)
        #         print(f"Index {i} has an array with shape {arr.shape}, expected {expected_shape}")
        all_keypoints = np.array(all_keypoints)
        all_features = np.array(all_features)
        keypoint_filenames = np.array(keypoint_filenames)

        print(all_keypoints.shape)
        print(all_features.shape)
        
        # 在返回的数据中增加文件名
        data = {
            "keypoints": all_keypoints,
            "wav_features": all_features,
            "keypoint_filenames": keypoint_filenames,  # 添加文件名
        }
        return data


def get_npy_paths(source_dir, parallelism, rank):
    # 将 source_dir 转换为 Path 对象
    source_dir = Path(source_dir)
    
    video_paths = [item for item in sorted(source_dir.iterdir()) if item.is_file() and item.suffix == '.npy']
    return [video_paths[i] for i in range(len(video_paths)) if i % parallelism == rank]

def get_audio_paths(source_dir, parallelism, rank):
    # 将 source_dir 转换为 Path 对象
    source_dir = Path(source_dir)
    
    video_paths = [item for item in sorted(source_dir.iterdir()) if item.is_file() and item.suffix == '.wav']
    return [video_paths[i] for i in range(len(video_paths)) if i % parallelism == rank]



class GestureDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "wavlm",
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
    ):
        self.data_path = data_path

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer = normalizer
        self.data_len = data_len

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)

        print("Loading dataset...")
        data = self.load_data() 

        print(
            f"Loaded {self.name} Dataset With Dimensions: Keypoints: {data['keypoints'].shape}, Wav_features: {data['wav_features'].shape}, Wavs: {len(data['wavs'])}"
        )

        self.data = {
            "keypoints": data['keypoints'],
            "wav_features": data["wav_features"],
            "wavs": data["wavs"],
        }
        assert len(data['keypoints']) == len(data["wav_features"])
        self.length = len(data['keypoints'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        wav_feature = torch.from_numpy(self.data["wav_features"][idx])
        keypoint = self.data['keypoints'][idx]
        keypoint_cond = torch.from_numpy(keypoint[0, :].astype(np.float32))
        keypoint_input = torch.from_numpy(keypoint[0:80, :].astype(np.float32))
        return keypoint_input, keypoint_cond, wav_feature, self.data["wavs"][idx]

    def load_data(self):
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )
    # def load_data(self):
    #     split_data_path = os.path.join(
    #         self.data_path, "test"
    #     )

        # Structure:
        # data
        #   |- train
        #   |    |- keypoints_sliced
        #   |    |- wav_sliced
        #   |    |- baseline_feats_sliced
        #   |    |- wavlm_feats_sliced
        #   |    |- keypoints
        #   |    |- wavs
        #   |    |- wavlm_feats

        keypoint_path = os.path.join(split_data_path, "keypoints_sliced")
        feature_path = os.path.join(split_data_path, f"{self.feature_type}_feats_sliced")
        baseline_path = os.path.join(split_data_path,  f"baseline_feats_sliced")
        wav_path = os.path.join(split_data_path, f"wavs_sliced")
        # sort keypoints and sounds
        keypoints = sorted(glob.glob(os.path.join(keypoint_path, "*.npy")))
        features = sorted(glob.glob(os.path.join(feature_path, "*.npy")))
        baseline_features = sorted(glob.glob(os.path.join(baseline_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        # stack the keypoints and features together
        all_keypoints = []
        all_features = []
        all_wavs = []
        print(f"Length of keypoints: {len(keypoints)}")
        print(f"Length of features: {len(features)}")
        print(f"Length of baseline_features: {len(baseline_features)}")
        print(f"Length of wavs: {len(wavs)}")
        assert len(keypoints) == len(features) == len(baseline_features) == len(wavs)
        for keypoint, feature, baseline_feature, wav in tqdm(zip(keypoints, features, baseline_features, wavs)):
            # make sure name is matching
            k_name = os.path.splitext(os.path.basename(keypoint))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            assert k_name == f_name == w_name, str((keypoint, feature, wav))
            # load keypoints
            data = np.load(keypoint)
            
            all_keypoints.append(data)
            if self.feature_type != 'baseline':
                input_feature = np.concatenate((np.load(feature), np.load(baseline_feature)), axis=-1)
            else:
                input_feature = np.load(baseline_feature)
            all_features.append(input_feature)
            all_wavs.append(wav)
        # shapes = [np.array(keypoint).shape for keypoint in all_keypoints]
        # unique_shapes = set(shapes)
        # # 只打印形状不一致的元素
        # if len(unique_shapes) > 1:
        #     print("发现形状不一致的元素：")
        #     for i, keypoint in enumerate(all_keypoints):
        #         if np.array(keypoint).shape not in unique_shapes:
        #             print(f"Index {i}: Shape {np.array(keypoint).shape}")
        # else:
        #     print("所有元素形状一致。")
        all_keypoints = np.array(all_keypoints)  
        all_features = np.array(all_features) 

        print(all_keypoints.shape)
        print(all_features.shape)
        
        data = {"keypoints": all_keypoints, "wav_features": all_features, "wavs": all_wavs}

        return data


