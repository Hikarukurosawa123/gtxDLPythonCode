import scipy.io as sio
import numpy as np
import mat73
import io


def scale_data(data_dict, params):
    scaled_data_dict = {}
    for key, items in data_dict.items():
        scaled_data_dict[key] = items * params[key]
    return scaled_data_dict

def normalization(fluorescence, optical_props):
    f = (fluorescence - np.mean(fluorescence, axis=(1,2,3), keepdims=True)) / \
                   (np.std(fluorescence, axis=(1,2,3), keepdims=True) + 1e-6)
    mu_a = optical_props[..., 0]
    mu_s = optical_props[..., 1]

    mu_a_mean = np.mean(mu_a, axis=(1,2), keepdims=True)
    mu_a_std = np.std(mu_a, axis=(1,2), keepdims=True)
    mu_a_norm = (mu_a - mu_a_mean) / (mu_a_std + 1e-6)

    mu_s_mean = np.mean(mu_s, axis=(1,2), keepdims=True)
    mu_s_std = np.std(mu_s, axis=(1,2), keepdims=True)
    mu_s_norm = (mu_s - mu_s_mean) / (mu_s_std + 1e-6)
    return f, mu_a_norm, mu_s_norm

def load_split_data(file_path):
    #data = sio.loadmat(file_path)

    data = mat73.loadmat(file_path)

    data = {k: v for k, v in data.items() if not k.startswith('__')}


    splits = ['train', 'val', 'test']
    data_by_split = {split: {} for split in splits}

    for key, value in data.items():
        for split in splits:
            if key.startswith(split + '_'):
                field = key[len(split) + 1:]
                data_by_split[split][field] = value

    return data_by_split

def load_data(file_path, scale_params):
    data_by_split = load_split_data(file_path)

    result = {}

    for type in ['train', 'val', 'test']:
        print(data_by_split['train'])

    for type in ['train', 'val', 'test']:
        fluorescence = data_by_split[type]['F']
        optical_props = data_by_split[type]['OP']
        depth = data_by_split[type]['DF']
        concentration_fluor = data_by_split[type]['QF']
        reflectance = data_by_split[type]['RE']

        # f, mu_a_norm, mu_s_norm = normalization(fluorescence, optical_props)
        # data_dict = {
        #     'fluorescence': f,
        #     'reflectance': reflectance,
        #     'depth': depth, 
        #     'mu_a': mu_a_norm,
        #     'mu_s': mu_s_norm,
        #     'concentration_fluor': concentration_fluor}
        scaled_data_dict = scale_data({
            'fluorescence': fluorescence,
            'reflectance': reflectance,
            'depth': depth, 
            'mu_a': optical_props[..., 0],
            'mu_s': optical_props[..., 1],
            'concentration_fluor': concentration_fluor
            }, scale_params)
        
        result[type] = scaled_data_dict

    return result