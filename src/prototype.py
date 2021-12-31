import os
import torch

import numpy as np

import utilities as tex_syn_utilities
import model as tex_syn_model


def load_model(target_image_name, base_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(base_dir, 'models', 'VGG19_normalized_avg_pool_pytorch')

    target_image_raw = tex_syn_utilities.load_image(os.path.join(os.path.join(base_dir, 'images'), target_image_name))
    target_image = tex_syn_utilities.preprocess_image(target_image_raw)

    vgg_model = tex_syn_model.Model(os.path.join(base_dir, 'models', 'VGG19_normalized_avg_pool_pytorch'), device, target_image)    
    return vgg_model

def partial_net_at(model, last_layer_index=0):
    partial_net = torch.nn.Sequential()
    for idx, layer in enumerate(list(model.net)[0:last_layer_index+1]):
        partial_net.add_module(str(idx), layer)

    model.gram_loss_hook.clear()
    output = partial_net(model.target_image)
    output = output.detach()
    
    # transfer data back from GPU, if needed
    if torch.cuda.is_available():
        output = output.cpu()
    
    return output.numpy()

def gram_matrix(activations):
    N = activations.shape[1]
    F = activations.reshape(N, -1)
    M = F.shape[1]
    G = np.dot(F, F.T) / M
    return G

# Generates a template that is of almost double both dimensions
# of the provided feature map and puts the maximum in the center.
#
# Can be used to extract parts of the pattern for further processing
#
def build_manhattan_template(feature_maps, pixel_radius=8):
    feature_map_dimension_x = 2 * feature_maps.shape[1] - 1
    feature_map_dimension_y = 2 * feature_maps.shape[2] - 1

    distance_template = np.ones((feature_map_dimension_x, feature_map_dimension_y))
    distance_template = distance_template * pixel_radius
    
    center_pos_x = feature_map_dimension_x // 2
    center_pos_y = feature_map_dimension_y // 2

    for x in range(feature_map_dimension_x):
        for y in range(feature_map_dimension_y):
            x_dist = np.abs(x - center_pos_x)
            y_dist = np.abs(y - center_pos_y)
            
            if x_dist + y_dist <= pixel_radius:
                distance_template[x, y] = x_dist + y_dist
            
    distance_template = (np.max(distance_template) - distance_template) / np.max(distance_template)
    #distance_template[np.where(distance_template > 0)] = 1

    return distance_template

# Takes a feature map (N, Height, Width) and returns a (Height1, Width1, Height2, Width2)
# 4-dimensional distance map where the height2/width2 represent the distance to other pixels
# in the same feature map and height1/width1 correspond to the pixel on which it is centered
# around. The given function can be exchanged to provide different distances, default
# is the manhattan distance.
#
def generate_distance_map(feature_maps, pixel_radius=8, template_func=build_manhattan_template):
    distance_template = template_func(feature_maps, pixel_radius=pixel_radius)
    
    distance_map = np.zeros(feature_maps.shape[1:] + feature_maps.shape[1:])
    center_pos_x = distance_template.shape[0] // 2
    center_pos_y = distance_template.shape[1] // 2
    
    for x in range(feature_maps.shape[1]):
        for y in range(feature_maps.shape[2]):
            template_x = center_pos_x - x
            template_y = center_pos_y - y

            distance_map[x, y, :, :] = distance_template[template_x:template_x+feature_maps.shape[1], template_y:template_y+feature_maps.shape[2]]
            distance_map[x, y, :, :] /= np.sum(distance_map[x, y, :, :])
            
    return distance_map

def distanced_gram_matrix(out, distance_map):
    new = np.empty((out.shape[0],) + distance_map.shape[:2])
    for i in range(out.shape[0]):
        new[i, :, :] = np.sum(out[i, :, :] * distance_map, axis=(2,3))
    return new

