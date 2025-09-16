import torch
import json
import numpy as np


def save_camera_intrinsics(intrinsics: torch.Tensor, file_path: str):
    file_path = str(file_path)  # Convert PosixPath to string if needed
    assert file_path.endswith('.json'), "Only .json files are supported now"
    intrinsics = intrinsics.cpu().numpy()
    json_data = {
        'intrinsics': intrinsics.tolist()
    }
    with open(file_path, 'w') as f:
        json.dump(json_data, f)


def load_camera_intrinsics(file_path: str) -> torch.Tensor:
    file_path = str(file_path)  # Convert PosixPath to string if needed
    assert file_path.endswith('.json'), "Only .json files are supported now"
    with open(file_path, 'r') as f:
        json_data = json.load(f)
        intrinsics = torch.from_numpy(np.array(json_data['intrinsics']))
        return intrinsics

def load_camera_intrinsics_and_image_shape(file_path: str) -> tuple[torch.Tensor, tuple[int, int]]:
    file_path = str(file_path)  # Convert PosixPath to string if needed
    assert file_path.endswith('.json'), "Only .json files are supported now"
    with open(file_path, 'r') as f:
        json_data = json.load(f)
        intrinsics = torch.from_numpy(np.array(json_data['intrinsics']))
        image_shape = tuple(json_data['image_shape'])
        return intrinsics, image_shape




def test_camera_utils():
    print('test save_camera_intrinsics')
    intrinsics = torch.randn(3, 3)
    save_camera_intrinsics(intrinsics, 'test.json')
    intrinsics_loaded = load_camera_intrinsics_and_image_shape('camsets/sets0/original_intrincts.json')
    print(intrinsics)
    print(intrinsics_loaded)
    # assert torch.allclose(intrinsics, intrinsics_loaded)

if __name__ == '__main__':
    test_camera_utils()