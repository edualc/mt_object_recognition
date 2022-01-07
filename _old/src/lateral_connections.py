import os
import sys

import torch
import numpy as np

base_dir = os.getcwd()
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'texture_synthesis_pytorch'))
im_dir = os.path.join(base_dir, 'images')

import utilities as tex_syn_utilities
import model as tex_syn_model
import optimize as tex_syn_opt

# Imagenet Mean (RGB)
IMAGENET_MEAN = np.array([ 0.48501961,  0.45795686, 0.40760392 ])
TARGET_IMAGE_NAME = 'unity_cube.jpg'

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(base_dir, 'models', 'VGG19_normalized_avg_pool_pytorch')

    target_image_raw = tex_syn_utilities.load_image(os.path.join(im_dir, TARGET_IMAGE_NAME))
    target_image = tex_syn_utilities.preprocess_image(target_image_raw)

    vgg_model = tex_syn_model.Model(os.path.join(base_dir, 'models', 'VGG19_normalized_avg_pool_pytorch'), device, target_image)    
    return vgg_model

def partial_net_at(model, last_layer_index=0):
    partial_net = torch.nn.Sequential()
    for idx, layer in enumerate(list(model.net)[0:last_layer_index+1]):
        partial_net.add_module(str(idx), layer)

    model.gram_loss_hook.clear()
    output = partial_net(model.target_image)
    return output.detach().numpy()

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
def build_manhattan_template(feature_maps):
    feature_map_dimension_x = 2 * feature_maps.shape[1] - 1
    feature_map_dimension_y = 2 * feature_maps.shape[2] - 1

    distance_template = np.zeros((feature_map_dimension_x, feature_map_dimension_y))
    
    center_pos_x = feature_map_dimension_x // 2
    center_pos_y = feature_map_dimension_y // 2

    for x in range(feature_map_dimension_x):
        for y in range(feature_map_dimension_y):
            distance_template[x, y] = np.abs(x - center_pos_x) + np.abs(y - center_pos_y)

    distance_template = (np.max(distance_template) - distance_template) / np.max(distance_template)
    #plt.imshow(distance_template, cmap='viridis')
    
    return distance_template
    
# Takes a feature map (N, Height, Width) and returns a (Height1, Width1, Height2, Width2)
# 4-dimensional distance map where the height2/width2 represent the distance to other pixels
# in the same feature map and height1/width1 correspond to the pixel on which it is centered
# around. The given function can be exchanged to provide different distances, default
# is the manhattan distance.
#
# Initially the full feature map is used and connected (relative to 
# their distance to the center). For each additional :num_reductions,
# this area gets halved.
#
def generate_distance_map(feature_maps, num_template_reductions=2, template_func=build_manhattan_template):
    distance_template = template_func(feature_maps)
    
    # make neighborhood smaller
    for _ in range(num_template_reductions):
        distance_template = ((2 * distance_template) - 1) / 2
        distance_template[np.where(distance_template < 0)] = 0
        distance_template *= 2
    
    distance_map = np.zeros(feature_maps.shape[1:] + feature_maps.shape[1:])
    center_pos_x = distance_template.shape[0] // 2
    center_pos_y = distance_template.shape[1] // 2
    
    for x in range(feature_maps.shape[1]):
        for y in range(feature_maps.shape[2]):
            template_x = center_pos_x - x
            template_y = center_pos_y - y

            distance_map[x, y, :, :] = distance_template[template_x:template_x+feature_maps.shape[1], template_y:template_y+feature_maps.shape[2]]
            
    return distance_map

def main():
    model = load_model()
    out_batch = partial_net_at(model, 27)
    out = out_batch[0, :, :, :]
    gr = gram_matrix(out_batch)

    distance_map = generate_distance_map(out, num_template_reductions=2)

    num_output_repetition = 4
    out_large = np.broadcast_to(out, (num_output_repetition,) + out.shape)

    import code; code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    main()
